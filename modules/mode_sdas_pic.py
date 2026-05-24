import os
os.environ["OMP_NUM_THREADS"] = "1"

import random
import numpy as np
import pandas as pd

from .utils_moea import (
    non_dominated_sort,
    crowding_distance,
    fast_cal_distance,
    precompute_neighbors,
    generate_uniform_weights,
)
from .pic_operator import pic_repair
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD


class MODE_SDAS_PIC:
    """
    MODE-SDAS-PIC with mixed 2D encoding.

    Encoding:
        - First D_task genes: task execution/window selection.
              0     = do not execute this task
              1..k  = select kth candidate visible time window
        - Last D_task genes: OTW position ratio in [0, 1].
              actual_start = earliest + ratio * (latest - earliest)

    PICEO note:
        pic_repair is applied only to the discrete task-selection part.
        The continuous ratio genes are kept and then repaired to [0, 1].
    """

    

    def __init__(self, problem, pop_size=100, generations=100,
                 Nsub=9, L=10, Rmem=5, CR=0.5, F=0.5, probVar=0.8,
                 alpha=0.15, sigma=0.01, gamma=0.7,
                 ref_front_file=None, elite_prob=0.6, archive_size=50,
                 execute_prob=0.15):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = generations
        self.Nsub = Nsub
        self.initial_Nsub = Nsub
        self.L = L
        self.Rmem = Rmem
        self.CR = CR
        self.F = F
        self.probVar = probVar
        self.alpha = alpha
        self.sigma = sigma
        self.gamma = gamma
        self.elite_prob = elite_prob
        self.archive_size = archive_size
        self.execute_prob = float(execute_prob)

        self.task_options = self._build_task_options()
        self.task_ids = list(self.task_options.keys())
        self.num_tasks = len(self.task_ids)
        self.option_counts = [len(self.task_options[t]) for t in self.task_ids]

        self.D_task = self.num_tasks
        self.D_ratio = self.num_tasks
        self.D = self.D_task + self.D_ratio

        # Kept for compatibility. For the full individual, use option_counts
        # for the first half and [0,1] bounds for the second half.
        self.max_options = self.option_counts

        self.W = None
        self.neighbor_list = None
        self.zideal = None

        # PICEO internal variables.
        self.r1 = 0.6
        self.r2 = 0.85
        self.S = 1.0

        # Subspace adaptive selection memory.
        self.gen_count = 0
        self.mem_theta = np.zeros((self.Rmem, self.Nsub))

        # Adaptive weight maintenance.
        self.check_interval = 25                    #每隔多少代检查一次子空间是否需要增删
        self.weight_deletion_threshold = 5          #子空间连续多少次/多少代没有个体关联后才考虑删除
        self.max_subspace_num = int(Nsub * 1.5)     # 最大子空间数量。
        self.min_subspace_num = max(2, Nsub // 2)   # 最小子空间数量。
        self.last_assoc_gen = np.full(Nsub, -1)
        self.crowd_factor = 1.5                     # 拥挤区域触发新增子空间的阈值因子。

        self.pop_dec = None
        self.pop_obj = None
        self.pop_raw = None
        self.pop_feas = None
        self.pop_cv = None
        self.associate = None

        # External feasible elite archive.
        self.external_archive_dec = []
        self.external_archive_obj = []

        self.hv_history = []
        self.hv_indicator = None
        self.ideal_point = None
        self.nadir_point = None

        if ref_front_file is not None and os.path.exists(ref_front_file):
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
        solution = []

        for i, task_id in enumerate(self.task_ids):
            opts = self.task_options[task_id]
            if len(opts) == 0:
                continue

            k = len(opts)
            choice_gene = int(round(x[i]))
            choice_gene = max(0, min(k, choice_gene))
            if choice_gene == 0:
                continue

            choice_idx = choice_gene - 1
            sat_idx, win_idx = opts[choice_idx]
            tw = self.problem.timewindows[sat_idx].iloc[win_idx]

            duration = float(
                self.problem.tasks.loc[
                    self.problem.tasks["task_id"] == task_id, "duration"
                ].values[0]
            )
            duration_td = pd.to_timedelta(duration, unit="s")

            earliest = tw["start_time"]
            latest = tw["end_time"] - duration_td

            ratio_gene = float(x[self.D_task + i])
            ratio = max(0.0, min(1.0, ratio_gene))

            if latest < earliest:
                actual_start = earliest
            else:
                available_seconds = (latest - earliest).total_seconds()
                actual_start = earliest + pd.to_timedelta(
                    ratio * available_seconds, unit="s"
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

    def get_obj_matrix(self, objs):
        return np.array([
            [d["profit"], d["load"], d["attitude"], d["quality"]]
            for d in objs
        ])

    def constraint_violation(self, x):
        sol = self.decode(x)
        return self.problem.constraint_violation(sol)

    # =========================================================
    # Encoding helpers / operators
    # =========================================================

    def repair_individual(self, individual):
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
            for d in range(self.D_task):
                k = self.option_counts[d]
                if k <= 0:
                    pop_dec[i, d] = 0
                else:
                    if np.random.rand() < self.execute_prob:
                        pop_dec[i, d] = np.random.randint(1, k + 1)
                    else:
                        pop_dec[i, d] = 0

            for d in range(self.D_task, self.D):
                pop_dec[i, d] = np.random.rand()

        return pop_dec

    def de_operator(self, parent_idx):
        parent_dec = self.pop_dec[parent_idx]
        sub = self.associate[parent_idx]

        if sub < 0 or sub >= self.Nsub:
            sub = random.randint(0, self.Nsub - 1)

        neighbors = (
            self.neighbor_list[sub]
            if sub < len(self.neighbor_list) and self.neighbor_list[sub]
            else list(range(self.Nsub))
        )

        # Archive-guided base vector.
        use_archive = (
            random.random() < self.elite_prob and len(self.external_archive_dec) > 0
        )
        if use_archive:
            base_dec = np.array(random.choice(self.external_archive_dec)).copy()
        else:
            base_dec = parent_dec.copy()

        # Prefer feasible individuals in selected neighboring subspaces.
        avail = [
            j for j in range(self.pop_size)
            if self.associate[j] in neighbors and self.pop_feas[j]
        ]
        if len(avail) < 2:
            avail = [j for j in range(self.pop_size) if self.pop_feas[j]]
        if len(avail) < 2:
            avail = list(range(self.pop_size))

        r1, r2 = random.sample(avail, 2) if len(avail) >= 2 else (0, 0)
        x1 = self.pop_dec[r1]
        x2 = self.pop_dec[r2]

        trial = np.zeros(self.D)
        for d in range(self.D):
            phi = base_dec[d] + self.F * (x1[d] - x2[d])

            if d < self.D_task:
                k = self.option_counts[d]
                if k <= 0:
                    trial[d] = 0
                else:
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
    # PICEO repair helper
    # =========================================================

    def apply_pic_repair(self, offspring_dec, off_obj, offspring_sub, gen):
        """
        Apply pic_repair only on the discrete task-selection genes.
        This avoids corrupting continuous ratio genes and avoids compatibility
        issues if pic_operator assumes a discrete vector of length D.
        """
        if len(offspring_dec) < 3:
            return offspring_dec

        ratio_part = offspring_dec[:, self.D_task:].copy()
        task_part = offspring_dec[:, :self.D_task].copy()

        # Temporarily expose a discrete-only view to pic_repair.
        old_D = self.D
        old_max_options = self.max_options
        old_option_counts = self.option_counts

        try:
            self.D = self.D_task
            # For 0..k task-choice genes, the number of values is k+1.
            self.max_options = [k + 1 for k in self.option_counts]

            repaired_task, self.r1, self.r2, self.S = pic_repair(
                self, task_part, off_obj, offspring_sub, gen, self.max_gen
            )

            repaired_task = np.asarray(repaired_task, dtype=float)

        finally:
            self.D = old_D
            self.max_options = old_max_options
            self.option_counts = old_option_counts

        repaired_dec = np.hstack([repaired_task, ratio_part])
        for dec in repaired_dec:
            self.repair_individual(dec)
        return repaired_dec

    # =========================================================
    # Adaptive weight maintenance
    # =========================================================

    def adapt_weights(self, gen):
        if gen % self.check_interval != 0 or gen == 0:
            return
        if self.associate is None or self.W is None:
            return

        counts = np.zeros(self.Nsub, dtype=int)
        for sub in self.associate:
            if 0 <= sub < self.Nsub:
                counts[sub] += 1

        for sub in range(self.Nsub):
            if counts[sub] > 0:
                self.last_assoc_gen[sub] = gen

        to_delete = []
        for sub in range(self.Nsub):
            if gen - self.last_assoc_gen[sub] >= self.weight_deletion_threshold:
                if self.Nsub - len(to_delete) > self.min_subspace_num:
                    to_delete.append(sub)

        if to_delete:
            for sub in reversed(to_delete):
                self.W = np.delete(self.W, sub, axis=0)
                self.last_assoc_gen = np.delete(self.last_assoc_gen, sub)
                self.mem_theta = np.delete(self.mem_theta, sub, axis=1)
            self.Nsub = self.W.shape[0]
            self.neighbor_list = precompute_neighbors(self.W, self.L)

        counts = np.zeros(self.Nsub, dtype=int)
        for sub in self.associate:
            if 0 <= sub < self.Nsub:
                counts[sub] += 1

        if self.Nsub <= 0:
            return

        avg_count = np.mean(counts) if np.sum(counts) > 0 else 0
        new_weights = []
        for sub in range(self.Nsub):
            if counts[sub] > self.crowd_factor * avg_count and counts[sub] > 1:
                neighbors = self.neighbor_list[sub] if self.neighbor_list else []
                if not neighbors:
                    continue
                nb = random.choice(neighbors)
                w_new = 0.5 * self.W[sub] + 0.5 * self.W[nb]
                norm = np.linalg.norm(w_new)
                if norm > 1e-12:
                    new_weights.append(w_new / norm)

        num_to_add = min(len(new_weights), self.max_subspace_num - self.Nsub)
        if num_to_add > 0:
            added = np.array(new_weights[:num_to_add])
            self.W = np.vstack([self.W, added])
            self.last_assoc_gen = np.append(self.last_assoc_gen, np.full(num_to_add, gen))
            self.mem_theta = np.hstack([self.mem_theta, np.zeros((self.Rmem, num_to_add))])

        self.Nsub = self.W.shape[0]
        self.neighbor_list = precompute_neighbors(self.W, self.L)

    # =========================================================
    # Selection, archive, HV
    # =========================================================

    def select_next_generation(self, combined_dec, combined_obj, combined_raw,
                               combined_feas, combined_cv):
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
                if not inds:
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
        self.pop_dec = np.array([combined_dec[i] for i in selected])
        self.pop_obj = [combined_obj[i] for i in selected]
        self.pop_raw = [combined_raw[i] for i in selected]
        self.pop_feas = [combined_feas[i] for i in selected]
        self.pop_cv = [combined_cv[i] for i in selected]

    def update_archive(self):
        if not self.pop_obj:
            return

        feasible_idx = [i for i, f in enumerate(self.pop_feas) if f]
        if not feasible_idx:
            return

        cur_dec = [self.pop_dec[i].copy() for i in feasible_idx]
        cur_obj = [self.pop_obj[i] for i in feasible_idx]

        combined_obj = self.external_archive_obj + cur_obj
        combined_dec = self.external_archive_dec + cur_dec
        if not combined_obj:
            return

        fronts = non_dominated_sort(combined_obj)
        new_archive_obj = []
        new_archive_dec = []
        count = 0

        for front in fronts:
            if count >= self.archive_size:
                break
            if count + len(front) <= self.archive_size:
                chosen = front
            else:
                need = self.archive_size - count
                cd = crowding_distance(front, combined_obj)
                chosen = sorted(front, key=lambda i: cd[i], reverse=True)[:need]

            for i in chosen:
                new_archive_obj.append(combined_obj[i])
                new_archive_dec.append(np.array(combined_dec[i]).copy())

            count += len(chosen)

        self.external_archive_obj = new_archive_obj
        self.external_archive_dec = new_archive_dec

    def compute_hv(self):
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

        return self.hv_indicator(norm_nd)

    def _evaluate_population(self, pop_dec):
        objs, raws, feas, cvs = [], [], [], []
        for dec in pop_dec:
            self.repair_individual(dec)
            obj, raw, feasible = self.evaluate(dec)
            objs.append(obj)
            raws.append(raw)
            feas.append(feasible)
            cvs.append(0.0 if feasible else self.constraint_violation(dec))
        return objs, raws, feas, cvs

    # =========================================================
    # Run
    # =========================================================

    def run(self):
        self.W = generate_uniform_weights(self.Nsub, 4)
        self.neighbor_list = precompute_neighbors(self.W, self.L)

        self.pop_dec = self.initialize()
        self.pop_obj, self.pop_raw, self.pop_feas, self.pop_cv = self._evaluate_population(self.pop_dec)

        obj_mat = self.get_obj_matrix(self.pop_obj)
        self.zideal = np.min(obj_mat, axis=0)
        D_mat = fast_cal_distance(obj_mat, self.W)
        self.associate = np.argmin(D_mat, axis=1)

        self.update_archive()

        hv = self.compute_hv()
        self.hv_history.append(hv)
        print(f"Init: HV={hv:.4f}, feasible ratio={sum(self.pop_feas) / self.pop_size:.2f}")

        for gen in range(self.max_gen):
            # Adaptive subspace-selection probability.
            if self.gen_count < self.Rmem:
                prob_sub = np.ones(self.Nsub) / self.Nsub
            else:
                theta_sum = np.sum(self.mem_theta, axis=0)
                if np.all(theta_sum <= 1e-12):
                    prob_sub = np.ones(self.Nsub) / self.Nsub
                else:
                    sigma_val = np.mean(theta_sum) / max(1, self.Nsub)
                    denom = np.sum(theta_sum) + self.Nsub * sigma_val
                    prob_sub = (theta_sum + sigma_val) / denom if denom > 1e-12 else np.ones(self.Nsub) / self.Nsub
                    prob_sub = np.clip(prob_sub, 0, None)
                    prob_sub = prob_sub / prob_sub.sum()

            n_evo = np.zeros(self.Nsub, dtype=int)
            cum_prob = np.cumsum(prob_sub)
            for _ in range(self.pop_size):
                sub = np.searchsorted(cum_prob, random.random())
                if sub >= self.Nsub:
                    sub = random.randint(0, self.Nsub - 1)
                n_evo[sub] += 1

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

            offspring_dec = np.array(offspring_dec) if offspring_dec else np.empty((0, self.D))
            off_obj, off_raw, off_feas, off_cv = self._evaluate_population(offspring_dec)

            if len(offspring_dec) > 0:
                off_obj_mat = self.get_obj_matrix(off_obj)
                self.zideal = np.minimum(self.zideal, np.min(off_obj_mat, axis=0))

            # PICEO repair on task-choice part only.
            if len(offspring_dec) >= 3:
                offspring_dec = self.apply_pic_repair(offspring_dec, off_obj, offspring_sub, gen)
                off_obj, off_raw, off_feas, off_cv = self._evaluate_population(offspring_dec)

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

            self.select_next_generation(combined_dec, combined_obj, combined_raw,
                                        combined_feas, combined_cv)

            obj_mat_new = self.get_obj_matrix(self.pop_obj)
            D_new = fast_cal_distance(obj_mat_new, self.W)
            self.associate = np.argmin(D_new, axis=1)

            self.update_archive()

            # Update push-distance memory using feasible offspring only.
            theta_sub = np.zeros(self.Nsub)
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
                    theta_sub[sub] += max(0.0, np.min(dist_to_ideal) - child_dist)

            if self.mem_theta.shape[1] != self.Nsub:
                new_mem = np.zeros((self.Rmem, self.Nsub))
                cols = min(self.mem_theta.shape[1], self.Nsub)
                new_mem[:, :cols] = self.mem_theta[:, :cols]
                self.mem_theta = new_mem

            if self.gen_count < self.Rmem:
                self.mem_theta[self.gen_count, :] = theta_sub
            else:
                self.mem_theta[:-1, :] = self.mem_theta[1:, :]
                self.mem_theta[-1, :] = theta_sub

            self.gen_count += 1
            self.adapt_weights(gen)

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
