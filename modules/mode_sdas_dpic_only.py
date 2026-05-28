"""
MODE-SDAS + DPIC only.

This file is an ablation version used to verify the independent contribution of
DPIC. It keeps the original MODE-SDAS framework and inserts the DPIC operator
after DE offspring generation and before environmental selection.

Included:
    - MODE-SDAS subspace-based evolution
    - mixed 2D scheduling encoding
    - DPIC repair on discrete task/window-selection genes

Removed:
    - external / dual archive guidance
    - adaptive subspace add/delete mechanism
    - PIC continuous/full-vector repair

Encoding:
    First D_task genes:
        0    = do not execute this task
        1..k = select the kth candidate visible time window
    Last D_task genes:
        ratio in [0, 1], controlling OTW position inside the chosen VTW
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import random
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .utils_moea import (
    non_dominated_sort,
    crowding_distance,
    fast_cal_distance,
    precompute_neighbors,
    generate_uniform_weights,
)
from .dpic_operator import dpic_repair

try:
    from pymoo.indicators.hv import Hypervolume
    from pymoo.indicators.igd import IGD
except Exception:  # pragma: no cover
    Hypervolume = None
    IGD = None


class MODE_SDAS_DPIC_ONLY:
    """
    MODE-SDAS + DPIC only ablation algorithm.

    Purpose:
        Compare this algorithm with original MODE-SDAS to verify whether DPIC
        itself contributes to performance improvement.

    Difference from full MODE-SDAS-DPIC:
        1. No external feasible archive.
        2. No dual archive.
        3. No adaptive subspace add/delete.
        4. Only DPIC is inserted into the MODE-SDAS framework.
    """

    def __init__(
        self,
        problem,
        pop_size: int = 100,
        generations: int = 100,
        Nsub: int = 21,
        L: int = 7,
        Rmem: int = 7,
        CR: float = 0.8,
        F: float = 0.35,
        probVar: float = 0.8,
        alpha: float = 0.03,
        sigma: float = 0.003,
        gamma: float = 0.85,
        ref_front_file: Optional[str] = None,
        use_dpic: bool = True,
        execute_prob: float = 0.15,
    ):
        self.problem = problem
        self.pop_size = int(pop_size)
        self.max_gen = int(generations)

        # MODE-SDAS parameters.
        self.Nsub = int(Nsub)
        self.L = int(L)
        self.Rmem = int(Rmem)
        self.CR = float(CR)
        self.F = float(F)
        self.probVar = float(probVar)

        # DPIC parameters.
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.use_dpic = bool(use_dpic)
        self.execute_prob = float(execute_prob)
        self.r1 = 0.25
        self.r2 = 0.7
        self.S = 1.0

        # Mixed 2D encoding.
        self.task_options = self._build_task_options()
        self.task_ids = list(self.task_options.keys())
        self.num_tasks = len(self.task_ids)
        self.option_counts = [len(self.task_options[t]) for t in self.task_ids]

        self.D_task = self.num_tasks
        self.D_ratio = self.num_tasks
        self.D = self.D_task + self.D_ratio

        # For DPIC: max_options[j] is the true upper bound k of gene j.
        # Valid discrete values are 0..k.
        self.max_options = self.option_counts.copy()

        # Subspace information.
        self.W = None
        self.neighbor_list = None
        self.zideal = None
        self.associate = None
        self.gen_count = 0
        self.mem_theta = np.zeros((self.Rmem, self.Nsub), dtype=float)

        # Population information.
        self.pop_dec = None
        self.pop_obj = None
        self.pop_raw = None
        self.pop_feas = None
        self.pop_cv = None

        # Indicators.
        self.hv_history: List[float] = []
        self.hv_indicator = None
        self.ideal_point = None
        self.nadir_point = None

        if ref_front_file is not None and Path(ref_front_file).exists() and IGD is not None:
            self.ref_front = np.loadtxt(ref_front_file, delimiter=",")
            self.igd_indicator = IGD(self.ref_front)
        else:
            self.ref_front = None
            self.igd_indicator = None

    # =========================================================
    # Build task options
    # =========================================================

    def _build_task_options(self):
        options = {}
        for task_id in self.problem.tasks["task_id"]:
            task_id = int(task_id)
            opts = []
            for sat_idx, tw_df in enumerate(self.problem.timewindows):
                rows = tw_df[tw_df["task_id"] == task_id].index.tolist()
                for win_idx in rows:
                    opts.append((sat_idx, win_idx))
            options[task_id] = opts
        return options

    # =========================================================
    # Decode / evaluate
    # =========================================================

    def decode(self, x):
        x = np.asarray(x, dtype=float)
        solution = []

        for i, task_id in enumerate(self.task_ids):
            opts = self.task_options[task_id]
            if len(opts) == 0:
                continue

            k = len(opts)
            choice_gene = int(round(x[i]))
            choice_gene = max(0, min(k, choice_gene))

            # 0 means this task is not executed.
            if choice_gene == 0:
                continue

            choice_idx = choice_gene - 1
            sat_idx, win_idx = opts[choice_idx]
            tw = self.problem.timewindows[sat_idx].iloc[win_idx]

            duration = float(
                self.problem.tasks.loc[
                    self.problem.tasks["task_id"] == task_id,
                    "duration",
                ].values[0]
            )
            duration_td = pd.to_timedelta(duration, unit="s")

            earliest = tw["start_time"]
            latest = tw["end_time"] - duration_td

            ratio_gene = float(x[self.D_task + i]) if self.D_task + i < len(x) else 0.0
            ratio = max(0.0, min(1.0, ratio_gene))

            if latest < earliest:
                actual_start = earliest
            else:
                available_seconds = (latest - earliest).total_seconds()
                actual_start = earliest + pd.to_timedelta(
                    ratio * available_seconds,
                    unit="s",
                )

            actual_end = actual_start + duration_td
            solution.append((task_id, win_idx, sat_idx, actual_start, actual_end))

        return solution

    def evaluate(self, x):
        sol = self.decode(x)
        raw, feasible = self.problem.evaluate_solution(sol)
        obj = {
            "profit": -raw["total_profit"],
            "load": raw["load_balance"],
            "attitude": raw["attitude_manoeuvre"],
            "quality": -raw["image_quality"],
        }
        return obj, raw, feasible

    def get_obj_matrix(self, objs: Sequence[dict]):
        return np.array(
            [[d["profit"], d["load"], d["attitude"], d["quality"]] for d in objs],
            dtype=float,
        )

    def constraint_violation(self, x):
        sol = self.decode(x)
        return self.problem.constraint_violation(sol)

    # =========================================================
    # Encoding helpers / operators
    # =========================================================

    def repair_individual(self, individual):
        individual = np.asarray(individual, dtype=float).copy()

        for d in range(self.D_task):
            k = self.option_counts[d]
            if k <= 0:
                individual[d] = 0
            else:
                individual[d] = int(round(individual[d]))
                individual[d] = max(0, min(k, individual[d]))

        for d in range(self.D_task, self.D):
            individual[d] = max(0.0, min(1.0, float(individual[d])))

        return individual

    def initialize(self):
        pop_dec = np.zeros((self.pop_size, self.D), dtype=float)

        for i in range(self.pop_size):
            # Discrete task/window-selection genes.
            for d in range(self.D_task):
                k = self.option_counts[d]
                if k <= 0:
                    pop_dec[i, d] = 0
                else:
                    if np.random.rand() < self.execute_prob:
                        pop_dec[i, d] = np.random.randint(1, k + 1)
                    else:
                        pop_dec[i, d] = 0

            # Continuous ratio genes.
            for d in range(self.D_task, self.D):
                pop_dec[i, d] = np.random.rand()

        return pop_dec

    def de_operator(self, parent_idx):
        """
        Discrete/continuous DE operator without archive guidance.
        This keeps the original MODE-SDAS mutation style and removes the
        external archive base-vector guidance.
        """

        parent_dec = self.pop_dec[parent_idx]
        sub = int(self.associate[parent_idx]) if self.associate is not None else -1

        if sub < 0 or sub >= self.Nsub:
            sub = random.randint(0, self.Nsub - 1)

        neighbors = (
            self.neighbor_list[sub]
            if self.neighbor_list is not None
            and sub < len(self.neighbor_list)
            and self.neighbor_list[sub]
            else list(range(self.Nsub))
        )

        # Prefer feasible individuals from neighboring subspaces.
        avail = [
            j for j in range(self.pop_size)
            if self.associate is not None
            and self.associate[j] in neighbors
            and self.pop_feas[j]
        ]
        if len(avail) < 2:
            avail = [j for j in range(self.pop_size) if self.pop_feas[j]]
        if len(avail) < 2:
            avail = list(range(self.pop_size))

        if len(avail) >= 2:
            r1_idx, r2_idx = random.sample(avail, 2)
        else:
            r1_idx = r2_idx = 0

        x1 = self.pop_dec[r1_idx]
        x2 = self.pop_dec[r2_idx]
        base_dec = parent_dec.copy()

        trial = np.zeros(self.D, dtype=float)

        for d in range(self.D):
            phi = base_dec[d] + self.F * (x1[d] - x2[d])

            if d < self.D_task:
                k = self.option_counts[d]
                if k <= 0:
                    trial[d] = 0
                else:
                    # Valid integer values are 0..k.
                    trial[d] = int(round(phi)) % (k + 1)
            else:
                trial[d] = max(0.0, min(1.0, phi))

        mask = np.random.rand(self.D) < self.CR
        if self.D > 0 and not np.any(mask):
            mask[random.randint(0, self.D - 1)] = True

        child_dec = parent_dec.copy()
        child_dec[mask] = trial[mask]
        return self.repair_individual(child_dec)

    # =========================================================
    # DPIC helper
    # =========================================================

    def apply_dpic_operator(self, offspring_dec, off_obj, offspring_sub, gen):
        """
        Apply DPIC only to the discrete task/window-selection genes.
        The continuous ratio genes are preserved and repaired to [0, 1].
        """

        if not self.use_dpic or len(offspring_dec) < 3:
            return offspring_dec

        offspring_dec = np.asarray(offspring_dec, dtype=float)
        task_part = offspring_dec[:, :self.D_task].copy()
        ratio_part = offspring_dec[:, self.D_task:].copy()

        old_max_options = self.max_options
        try:
            self.max_options = self.option_counts.copy()
            repaired_task, self.r1, self.r2, self.S = dpic_repair(
                self,
                task_part,
                off_obj,
                offspring_sub,
                gen,
                self.max_gen,
                ratio_part=ratio_part,
            )
            repaired_task = np.asarray(repaired_task, dtype=float)
        finally:
            self.max_options = old_max_options

        repaired_dec = np.hstack([repaired_task, ratio_part])
        repaired_dec = np.array([self.repair_individual(dec) for dec in repaired_dec])
        return repaired_dec

    # Backward-compatible name if the main script expects apply_pic_repair.
    def apply_pic_repair(self, offspring_dec, off_obj, offspring_sub, gen):
        return self.apply_dpic_operator(offspring_dec, off_obj, offspring_sub, gen)

    # =========================================================
    # Selection and indicators
    # =========================================================

    def _reassociate_population(self):
        if self.W is None or self.pop_obj is None:
            return
        obj_mat = self.get_obj_matrix(self.pop_obj)
        D_mat = fast_cal_distance(obj_mat, self.W)
        self.associate = np.argmin(D_mat, axis=1)

    def select_next_generation(
        self,
        combined_dec,
        combined_obj,
        combined_raw,
        combined_feas,
        combined_cv,
    ):
        feasible_idx = [i for i, f in enumerate(combined_feas) if f]
        infeasible_idx = [i for i, f in enumerate(combined_feas) if not f]
        selected = []

        if len(feasible_idx) >= self.pop_size:
            feasible_obj = [combined_obj[i] for i in feasible_idx]
            obj_mat = self.get_obj_matrix(feasible_obj)
            D_all = fast_cal_distance(obj_mat, self.W)
            all_assoc = np.argmin(D_all, axis=1)

            base = self.pop_size // self.Nsub
            remainder = self.pop_size - base * self.Nsub
            capacities = [base] * self.Nsub
            for i in range(remainder):
                capacities[i] += 1

            sub_inds = [[] for _ in range(self.Nsub)]
            for local_idx, sub in enumerate(all_assoc):
                sub_inds[sub].append(local_idx)

            selected_local = []
            for sub in range(self.Nsub):
                inds = sub_inds[sub]
                if not inds or capacities[sub] <= 0:
                    continue

                sub_obj = [feasible_obj[i] for i in inds]
                fronts = non_dominated_sort(sub_obj)
                filled = 0

                for front in fronts:
                    if filled >= capacities[sub]:
                        break

                    if filled + len(front) <= capacities[sub]:
                        selected_local.extend([inds[i] for i in front])
                        filled += len(front)
                    else:
                        need = capacities[sub] - filled
                        cd = crowding_distance(front, sub_obj)
                        sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                        selected_local.extend([inds[i] for i in sorted_front[:need]])
                        break

            selected = [feasible_idx[i] for i in selected_local]

            # Fill shortage caused by empty subspaces using global NSGA-II style selection.
            if len(selected) < self.pop_size:
                selected_set = set(selected)
                remaining_local = [
                    i for i in range(len(feasible_idx))
                    if feasible_idx[i] not in selected_set
                ]
                if remaining_local:
                    remaining_obj = [feasible_obj[i] for i in remaining_local]
                    fronts = non_dominated_sort(remaining_obj)
                    for front in fronts:
                        if len(selected) >= self.pop_size:
                            break
                        if len(selected) + len(front) <= self.pop_size:
                            selected.extend([feasible_idx[remaining_local[i]] for i in front])
                        else:
                            need = self.pop_size - len(selected)
                            cd = crowding_distance(front, remaining_obj)
                            sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                            selected.extend([
                                feasible_idx[remaining_local[i]]
                                for i in sorted_front[:need]
                            ])
                            break
        else:
            selected.extend(feasible_idx)
            remain = self.pop_size - len(selected)
            if remain > 0 and infeasible_idx:
                sorted_inf = sorted(infeasible_idx, key=lambda i: combined_cv[i])
                selected.extend(sorted_inf[:remain])

        if len(selected) < self.pop_size:
            need = self.pop_size - len(selected)
            if selected:
                selected.extend(np.random.choice(selected, need, replace=True).tolist())
            else:
                all_idx = list(range(len(combined_dec)))
                selected.extend(np.random.choice(all_idx, need, replace=True).tolist())

        selected = selected[:self.pop_size]

        self.pop_dec = np.array([combined_dec[i] for i in selected], dtype=float)
        self.pop_obj = [combined_obj[i] for i in selected]
        self.pop_raw = [combined_raw[i] for i in selected]
        self.pop_feas = [combined_feas[i] for i in selected]
        self.pop_cv = [combined_cv[i] for i in selected]

    def compute_hv(self):
        if Hypervolume is None:
            return 0.0

        feasible_obj = [self.pop_obj[i] for i, f in enumerate(self.pop_feas) if f]
        if not feasible_obj:
            return 0.0

        F = self.get_obj_matrix(feasible_obj)
        fronts = non_dominated_sort(feasible_obj)
        nd_F = F[fronts[0]]

        if self.ideal_point is None:
            self.ideal_point = np.min(nd_F, axis=0)
        else:
            self.ideal_point = np.minimum(self.ideal_point, np.min(nd_F, axis=0))

        if self.nadir_point is None:
            self.nadir_point = np.max(nd_F, axis=0)
        else:
            self.nadir_point = np.maximum(self.nadir_point, np.max(nd_F, axis=0))

        denom = self.nadir_point - self.ideal_point
        denom[denom < 1e-12] = 1.0
        norm_nd = (nd_F - self.ideal_point) / denom

        if self.hv_indicator is None:
            self.hv_indicator = Hypervolume(ref_point=np.ones(4))

        return float(self.hv_indicator(norm_nd))

    def _evaluate_population(self, pop_dec):
        objs, raws, feas, cvs = [], [], [], []
        for dec in pop_dec:
            dec = self.repair_individual(dec)
            obj, raw, feasible = self.evaluate(dec)
            objs.append(obj)
            raws.append(raw)
            feas.append(feasible)
            cvs.append(0.0 if feasible else float(self.constraint_violation(dec)))
        return objs, raws, feas, cvs

    # =========================================================
    # Run
    # =========================================================

    def run(self):
        self.W = generate_uniform_weights(self.Nsub, 4)
        self.neighbor_list = precompute_neighbors(
            self.W,
            min(self.L, max(1, self.Nsub - 1)),
        )

        self.pop_dec = self.initialize()
        self.pop_obj, self.pop_raw, self.pop_feas, self.pop_cv = self._evaluate_population(self.pop_dec)

        obj_mat = self.get_obj_matrix(self.pop_obj)
        self.zideal = np.min(obj_mat, axis=0)
        D_mat = fast_cal_distance(obj_mat, self.W)
        self.associate = np.argmin(D_mat, axis=1)

        hv = self.compute_hv()
        self.hv_history.append(hv)
        print(f"Init: HV={hv:.4f}, feasible ratio={sum(self.pop_feas) / self.pop_size:.2f}")

        for gen in range(self.max_gen):
            # -------------------------------------------------
            # Adaptive subspace-selection probability in MODE-SDAS.
            # This is not subspace add/delete; it only allocates more evolution
            # chances to promising fixed subspaces.
            # -------------------------------------------------
            if self.gen_count < self.Rmem:
                prob_sub = np.ones(self.Nsub) / self.Nsub
            else:
                theta_sum = np.sum(self.mem_theta, axis=0)
                if np.all(theta_sum <= 1e-12):
                    prob_sub = np.ones(self.Nsub) / self.Nsub
                else:
                    sigma_val = np.mean(theta_sum) / max(1, self.Nsub)
                    denom = np.sum(theta_sum) + self.Nsub * sigma_val
                    if denom > 1e-12:
                        prob_sub = (theta_sum + sigma_val) / denom
                    else:
                        prob_sub = np.ones(self.Nsub) / self.Nsub
                    prob_sub = np.clip(prob_sub, 0, None)
                    prob_sub = prob_sub / prob_sub.sum()

            n_evo = np.zeros(self.Nsub, dtype=int)
            cum_prob = np.cumsum(prob_sub)

            for _ in range(self.pop_size):
                sub = int(np.searchsorted(cum_prob, random.random()))
                if sub >= self.Nsub:
                    sub = random.randint(0, self.Nsub - 1)
                n_evo[sub] += 1

            # -------------------------------------------------
            # DE offspring generation.
            # -------------------------------------------------
            offspring_dec = []
            offspring_sub = []

            for sub in range(self.Nsub):
                if n_evo[sub] == 0:
                    continue

                idx_sub = [i for i in range(len(self.associate)) if self.associate[i] == sub]
                if not idx_sub:
                    idx_sub = list(range(len(self.associate)))

                for _ in range(n_evo[sub]):
                    parent_idx = random.choice(idx_sub)
                    offspring_dec.append(self.de_operator(parent_idx))
                    offspring_sub.append(sub)

            offspring_dec = (
                np.array(offspring_dec, dtype=float)
                if offspring_dec
                else np.empty((0, self.D), dtype=float)
            )

            off_obj, off_raw, off_feas, off_cv = self._evaluate_population(offspring_dec)

            if len(offspring_dec) > 0:
                off_obj_mat = self.get_obj_matrix(off_obj)
                self.zideal = np.minimum(self.zideal, np.min(off_obj_mat, axis=0))

            # -------------------------------------------------
            # DPIC insertion point: after DE and before selection.
            # -------------------------------------------------
            if self.use_dpic and len(offspring_dec) >= 3:
                offspring_dec = self.apply_dpic_operator(
                    offspring_dec,
                    off_obj,
                    offspring_sub,
                    gen,
                )
                off_obj, off_raw, off_feas, off_cv = self._evaluate_population(offspring_dec)

                if len(offspring_dec) > 0:
                    off_obj_mat = self.get_obj_matrix(off_obj)
                    self.zideal = np.minimum(self.zideal, np.min(off_obj_mat, axis=0))

            # -------------------------------------------------
            # Merge and environmental selection.
            # -------------------------------------------------
            if len(offspring_dec) > 0:
                combined_dec = np.vstack([self.pop_dec, offspring_dec])
                combined_obj = self.pop_obj + off_obj
                combined_raw = self.pop_raw + off_raw
                combined_feas = self.pop_feas + off_feas
                combined_cv = self.pop_cv + off_cv
            else:
                combined_dec = self.pop_dec.copy()
                combined_obj = self.pop_obj.copy()
                combined_raw = self.pop_raw.copy()
                combined_feas = self.pop_feas.copy()
                combined_cv = self.pop_cv.copy()

            self.select_next_generation(
                combined_dec,
                combined_obj,
                combined_raw,
                combined_feas,
                combined_cv,
            )

            self._reassociate_population()

            # -------------------------------------------------
            # Update push-distance memory using feasible offspring only.
            # -------------------------------------------------
            theta_sub = np.zeros(self.Nsub, dtype=float)

            for idx, sub in enumerate(offspring_sub):
                if idx >= len(off_obj) or idx >= len(off_feas) or not off_feas[idx]:
                    continue
                if sub < 0 or sub >= self.Nsub:
                    continue

                sub_pop_idx = [
                    i for i in range(len(self.pop_obj))
                    if self.associate[i] == sub and self.pop_feas[i]
                ]

                if sub_pop_idx:
                    sub_objs = self.get_obj_matrix([self.pop_obj[i] for i in sub_pop_idx])
                    dist_to_ideal = np.sqrt(np.sum((sub_objs - self.zideal) ** 2, axis=1))
                    child_arr = self.get_obj_matrix([off_obj[idx]])[0]
                    child_dist = np.sqrt(np.sum((child_arr - self.zideal) ** 2))
                    theta_sub[sub] += max(0.0, float(np.min(dist_to_ideal) - child_dist))

            if self.mem_theta.shape[1] != self.Nsub:
                new_mem = np.zeros((self.Rmem, self.Nsub), dtype=float)
                cols = min(self.mem_theta.shape[1], self.Nsub)
                new_mem[:, :cols] = self.mem_theta[:, :cols]
                self.mem_theta = new_mem

            if self.gen_count < self.Rmem:
                self.mem_theta[self.gen_count, :] = theta_sub
            else:
                self.mem_theta[:-1, :] = self.mem_theta[1:, :]
                self.mem_theta[-1, :] = theta_sub

            self.gen_count += 1

            hv = self.compute_hv()
            self.hv_history.append(hv)

            profits = [raw["total_profit"] for raw in self.pop_raw]
            loads = [raw["load_balance"] for raw in self.pop_raw]
            attitudes = [raw["attitude_manoeuvre"] for raw in self.pop_raw]
            qualities = [raw["image_quality"] for raw in self.pop_raw]
            task_counts = [len(self.decode(dec)) for dec in self.pop_dec]
            fea_ratio = sum(self.pop_feas) / self.pop_size

            print(
                f"Gen {gen + 1:3d}: "
                f"profit_max={max(profits):.2f}, profit_avg={np.mean(profits):.2f}, "
                f"load_min={min(loads):.2f}, att_min={min(attitudes):.2f}, "
                f"quality_max={max(qualities):.2f}, quality_avg={np.mean(qualities):.2f}, "
                f"tasks_avg={np.mean(task_counts):.2f}, HV={hv:.4f}, "
                f"feasible ratio={fea_ratio:.2f}, Nsub={self.Nsub}"
            )

        return self.pop_dec, self.pop_raw, self.pop_obj, self.hv_history


# Backward-compatible aliases.
MODE_SDAS_DPIC_ABLATION = MODE_SDAS_DPIC_ONLY
MODE_SDAS_DPIC_ONLY_ABLATION = MODE_SDAS_DPIC_ONLY
