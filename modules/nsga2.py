"""
Plain NSGA-II baseline for the satellite scheduling problem.

This version is intended as a common/basic NSGA-II comparison baseline.
It keeps only the standard NSGA-II components:
    - random initialization in decision-variable bounds
    - binary tournament selection
    - SBX crossover
    - polynomial mutation
    - non-dominated sorting
    - crowding-distance environmental selection

It does NOT include MODE-SDAS, DPIC/PIC, dual archive, adaptive subspace,
problem-specific insertion/repair/local-search operators, or elite archive guidance.

The only problem-dependent parts kept are the necessary decode/evaluate functions,
because any algorithm must map a chromosome to a satellite scheduling solution.
"""

import random
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from .utils_moea import non_dominated_sort, crowding_distance

try:
    from pymoo.indicators.hv import Hypervolume
except Exception:  # pragma: no cover
    Hypervolume = None


class NSGA2_PLAIN:
    """
    Plain constrained NSGA-II.

    Encoding
    --------
    The same mixed chromosome is used only so that the algorithm can evaluate
    the same scheduling model:
        first D_task genes:
            continuous values clipped to [0, k], decoded by round()
            0    = do not execute this task
            1..k = select the kth candidate window
        last D_task genes:
            continuous ratio in [0, 1]

    Important
    ---------
    This is still a valid scheduling optimizer, but it has no scheduling-specific
    search operator.  Constraint handling uses the generic Deb feasibility rule:
        feasible dominates infeasible;
        among infeasible solutions, lower constraint violation is better.
    """

    def __init__(
        self,
        problem,
        pop_size: int = 100,
        generations: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: Optional[float] = None,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
        ref_front_file: Optional[str] = None,
    ):
        self.problem = problem
        self.pop_size = int(pop_size)
        self.max_gen = int(generations)
        self.crossover_prob = float(crossover_prob)
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)

        self.task_options = self._build_task_options()
        self.task_ids = list(self.task_options.keys())
        self.num_tasks = len(self.task_ids)
        self.option_counts = [len(self.task_options[t]) for t in self.task_ids]

        self.D_task = self.num_tasks
        self.D_ratio = self.num_tasks
        self.D = self.D_task + self.D_ratio

        self.mutation_prob = float(mutation_prob) if mutation_prob is not None else 1.0 / max(1, self.D)

        self.lower = np.zeros(self.D, dtype=float)
        self.upper = np.ones(self.D, dtype=float)
        for d in range(self.D_task):
            self.upper[d] = max(0, self.option_counts[d])
        for d in range(self.D_task, self.D):
            self.upper[d] = 1.0

        self.pop_dec = None
        self.pop_obj = None
        self.pop_raw = None
        self.pop_feas = None
        self.pop_cv = None

        self.hv_history: List[float] = []
        self.hv_indicator = None
        self.ideal_point = None
        self.nadir_point = None

        self.ref_front_file = ref_front_file

    # =========================================================
    # Problem-dependent decode/evaluate interface
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

            ratio_gene = float(x[self.D_task + i]) if self.D_task + i < len(x) else 0.0
            ratio = max(0.0, min(1.0, ratio_gene))

            if latest < earliest:
                actual_start = earliest
            else:
                available_seconds = (latest - earliest).total_seconds()
                actual_start = earliest + pd.to_timedelta(ratio * available_seconds, unit="s")

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

    def constraint_violation(self, x):
        sol = self.decode(x)
        return float(self.problem.constraint_violation(sol))

    def get_obj_matrix(self, objs):
        return np.array(
            [[d["profit"], d["load"], d["attitude"], d["quality"]] for d in objs],
            dtype=float,
        )

    # =========================================================
    # Standard NSGA-II operators
    # =========================================================

    def repair_individual(self, x):
        """Only bound clipping.  No scheduling-specific repair is used."""
        x = np.asarray(x, dtype=float).copy()
        return np.minimum(np.maximum(x, self.lower), self.upper)

    def initialize(self):
        pop = np.zeros((self.pop_size, self.D), dtype=float)
        for i in range(self.pop_size):
            # Generic random initialization inside variable bounds.
            pop[i] = self.lower + np.random.rand(self.D) * (self.upper - self.lower)
        return pop

    def sbx_crossover(self, p1, p2):
        c1 = np.asarray(p1, dtype=float).copy()
        c2 = np.asarray(p2, dtype=float).copy()

        if random.random() > self.crossover_prob:
            return c1, c2

        for i in range(self.D):
            if random.random() > 0.5:
                continue

            y1 = min(p1[i], p2[i])
            y2 = max(p1[i], p2[i])
            lb = self.lower[i]
            ub = self.upper[i]

            if abs(y1 - y2) <= 1e-14 or ub - lb <= 1e-14:
                continue

            rand = random.random()
            beta = 1.0 + (2.0 * (y1 - lb) / (y2 - y1))
            alpha = 2.0 - beta ** (-(self.eta_c + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (self.eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1.0))
            child1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

            beta = 1.0 + (2.0 * (ub - y2) / (y2 - y1))
            alpha = 2.0 - beta ** (-(self.eta_c + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (self.eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1.0))
            child2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

            child1 = min(max(child1, lb), ub)
            child2 = min(max(child2, lb), ub)

            if random.random() <= 0.5:
                c1[i] = child2
                c2[i] = child1
            else:
                c1[i] = child1
                c2[i] = child2

        return self.repair_individual(c1), self.repair_individual(c2)

    def polynomial_mutation(self, x):
        y = np.asarray(x, dtype=float).copy()
        for i in range(self.D):
            if random.random() > self.mutation_prob:
                continue

            lb = self.lower[i]
            ub = self.upper[i]
            if ub - lb <= 1e-14:
                y[i] = lb
                continue

            delta1 = (y[i] - lb) / (ub - lb)
            delta2 = (ub - y[i]) / (ub - lb)
            rand = random.random()
            mut_pow = 1.0 / (self.eta_m + 1.0)

            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.eta_m + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.eta_m + 1.0))
                deltaq = 1.0 - val ** mut_pow

            y[i] = y[i] + deltaq * (ub - lb)
            y[i] = min(max(y[i], lb), ub)

        return self.repair_individual(y)

    # =========================================================
    # Ranking / selection
    # =========================================================

    def _rank_and_crowding(self, objs, feas, cvs):
        n = len(objs)
        ranks = np.full(n, 10**9, dtype=int)
        crowd = np.zeros(n, dtype=float)

        feasible_idx = [i for i, f in enumerate(feas) if f]
        if feasible_idx:
            feasible_obj = [objs[i] for i in feasible_idx]
            fronts = non_dominated_sort(feasible_obj)
            for r, front in enumerate(fronts):
                original_front = [feasible_idx[i] for i in front]
                for idx in original_front:
                    ranks[idx] = r
                cd_local = crowding_distance(front, feasible_obj)
                for local_i in front:
                    crowd[feasible_idx[local_i]] = cd_local[local_i]

        # Infeasible solutions are ranked after feasible solutions and compared by CV.
        return ranks, crowd

    def _better(self, a, b, ranks, crowd):
        fa = self.pop_feas[a]
        fb = self.pop_feas[b]

        if fa and not fb:
            return a
        if fb and not fa:
            return b

        if not fa and not fb:
            if self.pop_cv[a] < self.pop_cv[b]:
                return a
            if self.pop_cv[b] < self.pop_cv[a]:
                return b
            return a if random.random() < 0.5 else b

        if ranks[a] < ranks[b]:
            return a
        if ranks[b] < ranks[a]:
            return b
        if crowd[a] > crowd[b]:
            return a
        if crowd[b] > crowd[a]:
            return b
        return a if random.random() < 0.5 else b

    def tournament_select(self, ranks, crowd):
        a, b = random.sample(range(self.pop_size), 2)
        return self._better(a, b, ranks, crowd)

    def select_next_generation(self, combined_dec, combined_obj, combined_raw, combined_feas, combined_cv):
        selected = []
        feasible_idx = [i for i, f in enumerate(combined_feas) if f]
        infeasible_idx = [i for i, f in enumerate(combined_feas) if not f]

        if len(feasible_idx) >= self.pop_size:
            feasible_obj = [combined_obj[i] for i in feasible_idx]
            fronts = non_dominated_sort(feasible_obj)
            for front in fronts:
                if len(selected) + len(front) <= self.pop_size:
                    selected.extend([feasible_idx[i] for i in front])
                else:
                    need = self.pop_size - len(selected)
                    cd = crowding_distance(front, feasible_obj)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    selected.extend([feasible_idx[i] for i in sorted_front[:need]])
                    break
        else:
            selected.extend(feasible_idx)
            remain = self.pop_size - len(selected)
            if remain > 0 and infeasible_idx:
                sorted_inf = sorted(infeasible_idx, key=lambda i: combined_cv[i])
                selected.extend(sorted_inf[:remain])

        if len(selected) < self.pop_size:
            all_idx = list(range(len(combined_dec)))
            need = self.pop_size - len(selected)
            selected.extend(np.random.choice(all_idx, need, replace=True).tolist())

        selected = selected[:self.pop_size]
        self.pop_dec = np.array([combined_dec[i] for i in selected], dtype=float)
        self.pop_obj = [combined_obj[i] for i in selected]
        self.pop_raw = [combined_raw[i] for i in selected]
        self.pop_feas = [combined_feas[i] for i in selected]
        self.pop_cv = [combined_cv[i] for i in selected]

    # =========================================================
    # Evaluation / HV
    # =========================================================

    def _evaluate_population(self, pop_dec):
        objs, raws, feas, cvs = [], [], [], []
        for dec in pop_dec:
            dec = self.repair_individual(dec)
            obj, raw, feasible = self.evaluate(dec)
            objs.append(obj)
            raws.append(raw)
            feas.append(feasible)
            cvs.append(0.0 if feasible else self.constraint_violation(dec))
        return objs, raws, feas, cvs

    def compute_hv(self):
        if Hypervolume is None:
            return 0.0

        feasible_obj = [self.pop_obj[i] for i, f in enumerate(self.pop_feas) if f]
        if not feasible_obj:
            return 0.0

        fronts = non_dominated_sort(feasible_obj)
        if not fronts or not fronts[0]:
            return 0.0

        F = self.get_obj_matrix(feasible_obj)
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
        F_norm = (nd_F - self.ideal_point) / denom
        F_norm = np.clip(F_norm, 0.0, 1.2)

        if self.hv_indicator is None:
            self.hv_indicator = Hypervolume(ref_point=np.ones(4) * 1.1)

        return float(self.hv_indicator(F_norm))

    # =========================================================
    # Main run
    # =========================================================

    def run(self):
        self.pop_dec = self.initialize()
        self.pop_obj, self.pop_raw, self.pop_feas, self.pop_cv = self._evaluate_population(self.pop_dec)

        hv = self.compute_hv()
        self.hv_history.append(hv)
        print(f"Init: HV={hv:.4f}, feasible ratio={sum(self.pop_feas) / self.pop_size:.2f}")

        for gen in range(self.max_gen):
            ranks, crowd = self._rank_and_crowding(self.pop_obj, self.pop_feas, self.pop_cv)

            offspring = []
            while len(offspring) < self.pop_size:
                p1_idx = self.tournament_select(ranks, crowd)
                p2_idx = self.tournament_select(ranks, crowd)
                p1 = self.pop_dec[p1_idx]
                p2 = self.pop_dec[p2_idx]

                c1, c2 = self.sbx_crossover(p1, p2)
                c1 = self.polynomial_mutation(c1)
                c2 = self.polynomial_mutation(c2)

                offspring.append(c1)
                if len(offspring) < self.pop_size:
                    offspring.append(c2)

            offspring_dec = np.array(offspring, dtype=float)
            off_obj, off_raw, off_feas, off_cv = self._evaluate_population(offspring_dec)

            combined_dec = np.vstack([self.pop_dec, offspring_dec])
            combined_obj = self.pop_obj + off_obj
            combined_raw = self.pop_raw + off_raw
            combined_feas = self.pop_feas + off_feas
            combined_cv = self.pop_cv + off_cv

            self.select_next_generation(
                combined_dec,
                combined_obj,
                combined_raw,
                combined_feas,
                combined_cv,
            )

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
                f"profit_max={max(profits):.2f} "
                f"load_min={min(loads):.2f}, att_min={min(attitudes):.2f}, "
                f"quality_max={max(qualities):.2f} "
                f"HV={hv:.4f}, "
                f"feasible ratio={fea_ratio:.2f}"
            )

        return self.pop_dec, self.pop_raw, self.pop_obj, self.hv_history
