import os
os.environ["OMP_NUM_THREADS"] = "1"

import random
import numpy as np
import pandas as pd

from .utils_moea import non_dominated_sort, crowding_distance
from pymoo.indicators.hv import Hypervolume


class NSGA2:
    """
    NSGA-II for satellite scheduling with mixed 2D encoding.

    Encoding:
        - First D_task genes: task execution/window selection.
              0     = do not execute this task
              1..k  = select the kth candidate visible time window
        - Last D_task genes: OTW position ratio in [0, 1].
              actual_start = earliest + ratio * (latest - earliest)

    Decoded solution item:
        (task_id, win_idx, sat_idx, actual_start, actual_end)
    """

    def __init__(self, problem, pop_size=100, generations=100,
                 crossover_prob=0.9, mutation_prob=0.1,
                 ref_front_file=None):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.task_options = self._build_task_options()
        self.task_ids = list(self.task_options.keys())
        self.num_tasks = len(self.task_ids)
        self.option_counts = [len(self.task_options[t]) for t in self.task_ids]

        self.D_task = self.num_tasks
        self.D_ratio = self.num_tasks
        self.D = self.D_task + self.D_ratio

        # Kept for compatibility with old code that may read max_options.
        self.max_options = self.option_counts

        self.pop_dec = None
        self.pop_obj = None
        self.pop_raw = None
        self.pop_feas = None
        self.pop_cv = None

        self.hv_history = []
        self.hv_indicator = None
        self.ideal_point = None
        self.nadir_point = None

        if ref_front_file is not None and os.path.exists(ref_front_file):
            from pymoo.indicators.igd import IGD
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

            # 0 = not executed; 1..k = choose a VTW.
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
    # Operators
    # =========================================================

    def initialize(self):
        pop_dec = np.zeros((self.pop_size, self.D), dtype=float)

        for i in range(self.pop_size):
            # Task selection genes.
            for d in range(self.D_task):
                k = self.option_counts[d]
                if k <= 0:
                    pop_dec[i, d] = 0
                else:
                    # 0 means not scheduled. Use a moderate skip probability
                    # to avoid all tasks being scheduled by construction.
                    if random.random() < 0.2:
                        pop_dec[i, d] = 0
                    else:
                        pop_dec[i, d] = random.randint(1, k)

            # OTW ratio genes.
            for d in range(self.D_task, self.D):
                pop_dec[i, d] = random.random()

        return pop_dec

    def crossover(self, parent1, parent2):
        child1 = parent1.copy()
        child2 = parent2.copy()

        if random.random() < self.crossover_prob:
            mask = np.random.rand(self.D) < 0.5
            child1[mask] = parent2[mask]
            child2[mask] = parent1[mask]

        self.repair_individual(child1)
        self.repair_individual(child2)
        return child1, child2

    def mutate(self, individual):
        # Discrete task-selection genes.
        for d in range(self.D_task):
            if random.random() < self.mutation_prob:
                k = self.option_counts[d]
                individual[d] = 0 if k <= 0 else random.randint(0, k)

        # Continuous ratio genes.
        for d in range(self.D_task, self.D):
            if random.random() < self.mutation_prob:
                individual[d] += random.gauss(0.0, 0.1)

        self.repair_individual(individual)
        return individual

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

    # =========================================================
    # Selection
    # =========================================================

    def tournament_select(self, fronts, cd):
        pool = random.sample(range(self.pop_size), 2)
        a, b = pool[0], pool[1]

        if self.pop_feas[a] and not self.pop_feas[b]:
            return a
        if self.pop_feas[b] and not self.pop_feas[a]:
            return b

        if (not self.pop_feas[a]) and (not self.pop_feas[b]):
            if self.pop_cv[a] < self.pop_cv[b]:
                return a
            if self.pop_cv[b] < self.pop_cv[a]:
                return b

        rank_a = next(fi for fi, f in enumerate(fronts) if a in f)
        rank_b = next(fi for fi, f in enumerate(fronts) if b in f)

        if rank_a < rank_b:
            return a
        if rank_b < rank_a:
            return b

        return a if cd.get(a, 0.0) > cd.get(b, 0.0) else b

    def select_next_generation(self, combined_dec, combined_obj, combined_raw,
                               combined_feas, combined_cv):
        feasible_idx = [i for i, f in enumerate(combined_feas) if f]
        infeasible_idx = [i for i, f in enumerate(combined_feas) if not f]
        selected = []

        if len(feasible_idx) >= self.pop_size:
            feasible_obj = [combined_obj[i] for i in feasible_idx]
            fronts = non_dominated_sort(feasible_obj)

            selected_local = []
            for front in fronts:
                if len(selected_local) + len(front) <= self.pop_size:
                    selected_local.extend(front)
                else:
                    remain = self.pop_size - len(selected_local)
                    cd_dict = crowding_distance(front, feasible_obj)
                    sorted_front = sorted(front, key=lambda i: cd_dict[i], reverse=True)
                    selected_local.extend(sorted_front[:remain])
                    break

            selected = [feasible_idx[i] for i in selected_local]
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

    # =========================================================
    # Indicators / run
    # =========================================================

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

    def run(self):
        self.pop_dec = self.initialize()
        self.pop_obj, self.pop_raw, self.pop_feas, self.pop_cv = self._evaluate_population(self.pop_dec)

        hv = self.compute_hv()
        self.hv_history.append(hv)
        print(f"Init: HV={hv:.4f}, feasible ratio={sum(self.pop_feas) / self.pop_size:.2f}")

        for gen in range(self.max_gen):
            fronts = non_dominated_sort(self.pop_obj)
            cd = {}
            for front in fronts:
                cd.update(crowding_distance(front, self.pop_obj))

            offspring_dec = []
            for _ in range(self.pop_size // 2):
                p1 = self.tournament_select(fronts, cd)
                p2 = self.tournament_select(fronts, cd)
                c1, c2 = self.crossover(self.pop_dec[p1], self.pop_dec[p2])
                offspring_dec.append(self.mutate(c1))
                offspring_dec.append(self.mutate(c2))

            if len(offspring_dec) < self.pop_size:
                p1 = self.tournament_select(fronts, cd)
                p2 = self.tournament_select(fronts, cd)
                c1, _ = self.crossover(self.pop_dec[p1], self.pop_dec[p2])
                offspring_dec.append(self.mutate(c1))

            offspring_dec = np.array(offspring_dec[:self.pop_size])
            off_obj, off_raw, off_feas, off_cv = self._evaluate_population(offspring_dec)

            combined_dec = np.vstack([self.pop_dec, offspring_dec])
            combined_obj = self.pop_obj + off_obj
            combined_raw = self.pop_raw + off_raw
            combined_feas = self.pop_feas + off_feas
            combined_cv = self.pop_cv + off_cv

            self.select_next_generation(combined_dec, combined_obj, combined_raw,
                                        combined_feas, combined_cv)

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
                f"quality_max={max(qualities):.2f} "
                f"tasks_avg={np.mean(task_counts):.2f}, HV={hv:.4f}, "
                f"feasible ratio={fea_ratio:.2f}"
            )

        return self.pop_dec, self.pop_raw, self.pop_obj, self.hv_history
