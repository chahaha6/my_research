import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2 as PymooNSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting



OBJ_KEYS = ["profit", "load", "attitude", "quality"]


class MixedScheduleRepair(Repair):
    """
    Repair operator for the mixed satellite scheduling chromosome.

    First D_task genes:
        discrete task/window choice, valid values are integers in [0, k].
        0 means the task is not scheduled; 1..k selects one candidate VTW.

    Last D_task genes:
        continuous OTW position ratio in [0, 1].
    """

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X, dtype=float).copy()

        if X.ndim == 1:
            X = X.reshape(1, -1)

        D_task = problem.D_task
        option_counts = problem.option_counts

        for i in range(X.shape[0]):
            for d in range(D_task):
                k = int(option_counts[d])
                if k <= 0:
                    X[i, d] = 0
                else:
                    X[i, d] = int(round(X[i, d]))
                    X[i, d] = np.clip(X[i, d], 0, k)

            X[i, D_task:] = np.clip(X[i, D_task:], 0.0, 1.0)

        return X


class SparseScheduleSampling(Sampling):
    """
    Sparse initial sampling for the mixed satellite scheduling chromosome.

    execute_prob is used only here to control the initial task activation rate.
    Later crossover, mutation, repair, and decoding use the standard encoding
    directly without any execute-probability gate.
    """

    def __init__(self, execute_prob=0.15):
        super().__init__()
        self.execute_prob = float(execute_prob)

    def _do(self, problem, n_samples, **kwargs):
        D_task = problem.D_task
        D = problem.n_var
        option_counts = problem.option_counts

        X = np.zeros((n_samples, D), dtype=float)

        for i in range(n_samples):
            for d in range(D_task):
                k = int(option_counts[d])
                if k <= 0:
                    X[i, d] = 0
                else:
                    if np.random.rand() < self.execute_prob:
                        X[i, d] = np.random.randint(1, k + 1)
                    else:
                        X[i, d] = 0

            X[i, D_task:] = np.random.rand(D_task)

        return X


class SatelliteSchedulingPymooProblem(ElementwiseProblem):
    """
    Pymoo problem wrapper for the satellite scheduling model.

    This wrapper only defines encoding, decoding, objectives and constraints.
    The NSGA-II search mechanism itself is provided by pymoo.
    """

    def __init__(self, scheduling_problem):
        self.problem = scheduling_problem

        self.task_options = self._build_task_options()
        self.task_ids = list(self.task_options.keys())
        self.num_tasks = len(self.task_ids)
        self.option_counts = [len(self.task_options[t]) for t in self.task_ids]

        self.D_task = self.num_tasks
        self.D_ratio = self.num_tasks
        D = self.D_task + self.D_ratio

        xl = np.zeros(D, dtype=float)
        xu = np.zeros(D, dtype=float)

        for d in range(self.D_task):
            xu[d] = self.option_counts[d]

        xu[self.D_task:] = 1.0

        # Compatible with different pymoo versions.
        try:
            super().__init__(
                n_var=D,
                n_obj=4,
                n_ieq_constr=1,
                xl=xl,
                xu=xu,
            )
        except TypeError:
            super().__init__(
                n_var=D,
                n_obj=4,
                n_constr=1,
                xl=xl,
                xu=xu,
            )

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

    def repair_vector(self, x):
        x = np.asarray(x, dtype=float).copy()

        for d in range(self.D_task):
            k = int(self.option_counts[d])
            if k <= 0:
                x[d] = 0
            else:
                x[d] = int(round(x[d]))
                x[d] = np.clip(x[d], 0, k)

        x[self.D_task:] = np.clip(x[self.D_task:], 0.0, 1.0)
        return x

    def decode(self, x):
        x = self.repair_vector(x)
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
                    self.problem.tasks["task_id"] == task_id,
                    "duration",
                ].values[0]
            )
            duration_td = pd.to_timedelta(duration, unit="s")

            earliest = tw["start_time"]
            latest = tw["end_time"] - duration_td

            ratio = float(x[self.D_task + i])
            ratio = max(0.0, min(1.0, ratio))

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

    def evaluate_x(self, x):
        x = self.repair_vector(x)
        sol = self.decode(x)

        raw, feasible = self.problem.evaluate_solution(sol)

        obj = {
            "profit": -raw["total_profit"],
            "load": raw["load_balance"],
            "attitude": raw["attitude_manoeuvre"],
            "quality": -raw["image_quality"],
        }

        try:
            cv = float(self.problem.constraint_violation(sol))
        except Exception:
            cv = 0.0 if feasible else 1.0

        if not feasible and cv <= 0.0:
            cv = 1.0

        return obj, raw, bool(feasible), max(0.0, cv)

    def _evaluate(self, x, out, *args, **kwargs):
        obj, raw, feasible, cv = self.evaluate_x(x)

        out["F"] = np.array(
            [obj["profit"], obj["load"], obj["attitude"], obj["quality"]],
            dtype=float,
        )

        # In pymoo, inequality constraints are feasible if G <= 0.
        out["G"] = np.array([cv], dtype=float)


class GenerationPrintCallback(Callback):
    """
    Print per-generation statistics during pymoo NSGA-II optimization.

    Statistics are computed from the current population:
        - profit_max / profit_avg: converted from F[:, 0] because profit is minimized as -profit
        - load_min: minimum load objective
        - att_min: minimum attitude manoeuvre objective
        - quality_max / quality_avg: converted from F[:, 3] because quality is minimized as -quality
        - HV: dynamic HV calculated only from feasible individuals
        - feasible ratio: number of feasible individuals / population size
    """

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.hv_history = []

    def notify(self, algorithm):
        pop = algorithm.pop
        F = pop.get("F")
        G = pop.get("G")

        gen = int(algorithm.n_gen)

        if F is None or len(F) == 0:
            self.hv_history.append(0.0)
            print(
                f"Gen {gen:3d}: "
                f"profit_max=0.00, profit_avg=0.00, "
                f"load_min=0.00, att_min=0.00, "
                f"quality_max=0.00, quality_avg=0.00, "
                f"HV=0.0000, feasible ratio=0.00",
                flush=True,
            )
            return

        F = np.asarray(F, dtype=float)

        if G is not None:
            G = np.asarray(G, dtype=float)
            if G.ndim == 1:
                feasible_mask = G <= 1e-12
            else:
                feasible_mask = np.all(G <= 1e-12, axis=1)
        else:
            feasible_mask = np.ones(F.shape[0], dtype=bool)

        feasible_count = int(np.sum(feasible_mask))
        feasible_ratio = feasible_count / max(1, F.shape[0])

        F_for_stats = F[feasible_mask] if feasible_count > 0 else F
        F_for_hv = F[feasible_mask] if feasible_count > 0 else np.empty((0, F.shape[1]))

        hv = self.owner._compute_dynamic_hv_from_F(F_for_hv)
        self.hv_history.append(hv)

        profits = -F_for_stats[:, 0]
        loads = F_for_stats[:, 1]
        attitudes = F_for_stats[:, 2]
        qualities = -F_for_stats[:, 3]

        print(
            f"Gen {gen:3d}: "
            f"profit_max={np.max(profits):.2f}, "
            f"profit_avg={np.mean(profits):.2f}, "
            f"load_min={np.min(loads):.2f}, "
            f"att_min={np.min(attitudes):.2f}, "
            f"quality_max={np.max(qualities):.2f}, "
            f"quality_avg={np.mean(qualities):.2f}, "
            f"HV={hv:.4f}, "
            f"feasible ratio={feasible_ratio:.2f}",
            flush=True,
        )


class NSGA2_PYMOO:
    """
    Standard pymoo NSGA-II baseline.

    The algorithm core uses pymoo's NSGA2 implementation. This class only wraps
    the satellite scheduling problem so that it can be called by your existing
    main.py interface:
        final_dec, final_raw, final_obj, hv_history = alg.run()
    """

    def __init__(
        self,
        problem,
        pop_size=100,
        generations=100,
        ref_front_file=None,
        seed=None,
        crossover_prob=0.9,
        crossover_eta=15,
        mutation_eta=20,
        execute_prob=0.15,
    ):
        self.problem = problem
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.ref_front_file = ref_front_file
        self.seed = seed
        self.crossover_prob = float(crossover_prob)
        self.crossover_eta = int(crossover_eta)
        self.mutation_eta = int(mutation_eta)
        self.execute_prob = float(execute_prob)

        self.hv_indicator = Hypervolume(ref_point=np.ones(4))
        self.ideal_point = None
        self.nadir_point = None

    @staticmethod
    def _obj_dict_to_array(obj_list):
        if not obj_list:
            return np.empty((0, 4), dtype=float)

        return np.array(
            [[obj[k] for k in OBJ_KEYS] for obj in obj_list],
            dtype=float,
        )

    def _compute_dynamic_hv_from_F(self, F):
        if F is None or len(F) == 0:
            return 0.0

        F = np.asarray(F, dtype=float)
        if F.ndim == 1:
            F = F.reshape(1, -1)

        if F.shape[0] == 0:
            return 0.0

        nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[nd_idx]

        if nd_F.shape[0] == 0:
            return 0.0

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
        norm_nd = np.clip(norm_nd, 0.0, 1.0)

        hv_value = float(self.hv_indicator(norm_nd))
        return max(0.0, min(1.0, hv_value))

    def _history_to_hv(self, history):
        hv_history = []

        for entry in history:
            pop = entry.pop
            F = pop.get("F")
            G = pop.get("G")

            if F is None or len(F) == 0:
                hv_history.append(0.0)
                continue

            F = np.asarray(F, dtype=float)

            if G is not None:
                G = np.asarray(G, dtype=float)
                if G.ndim == 1:
                    feasible_mask = G <= 1e-12
                else:
                    feasible_mask = np.all(G <= 1e-12, axis=1)
                F_feas = F[feasible_mask]
            else:
                F_feas = F

            hv_history.append(self._compute_dynamic_hv_from_F(F_feas))

        return hv_history

    def _evaluate_final_population(self, pymoo_problem, X):
        if X is None:
            return [], [], [], []

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        final_dec = []
        final_raw = []
        final_obj = []
        feasible_flags = []

        for x in X:
            x = pymoo_problem.repair_vector(x)
            obj, raw, feasible, cv = pymoo_problem.evaluate_x(x)

            final_dec.append(x)
            final_raw.append(raw)
            final_obj.append(obj)
            feasible_flags.append(feasible)

        if any(feasible_flags):
            idx = [i for i, f in enumerate(feasible_flags) if f]
            final_dec = [final_dec[i] for i in idx]
            final_raw = [final_raw[i] for i in idx]
            final_obj = [final_obj[i] for i in idx]
        else:
            final_dec = []
            final_raw = []
            final_obj = []

        return (
            np.array(final_dec, dtype=float),
            final_raw,
            final_obj,
            feasible_flags,
        )

    def run(self):
        pymoo_problem = SatelliteSchedulingPymooProblem(self.problem)

        mutation_prob = 1.0 / max(1, pymoo_problem.n_var)

        algorithm = PymooNSGA2(
            pop_size=self.pop_size,
            sampling=SparseScheduleSampling(execute_prob=self.execute_prob),
            crossover=SBX(prob=self.crossover_prob, eta=self.crossover_eta),
            mutation=PM(prob=mutation_prob, eta=self.mutation_eta),
            repair=MixedScheduleRepair(),
            eliminate_duplicates=True,
        )

        # Reset dynamic HV bounds for this independent run.
        self.ideal_point = None
        self.nadir_point = None

        callback = GenerationPrintCallback(self)

        print("开始运行 pymoo NSGA-II ...", flush=True)

        res = minimize(
            pymoo_problem,
            algorithm,
            ("n_gen", self.generations),
            seed=self.seed,
            verbose=False,
            save_history=False,
            callback=callback,
        )

        print("pymoo NSGA-II 运行结束。", flush=True)

        hv_history = callback.hv_history

        # Use final population rather than only res.X, so that your existing
        # summary and final_objs saving logic remain consistent with other
        # algorithms.
        X_final = res.pop.get("X") if res.pop is not None else res.X

        final_dec, final_raw, final_obj, feasible_flags = self._evaluate_final_population(
            pymoo_problem,
            X_final,
        )
        self.last_final_feasible_flags = feasible_flags

        if len(final_raw) > 0:
            profits = [raw["total_profit"] for raw in final_raw]
            loads = [raw["load_balance"] for raw in final_raw]
            attitudes = [raw["attitude_manoeuvre"] for raw in final_raw]
            qualities = [raw["image_quality"] for raw in final_raw]
            fea_ratio = sum(feasible_flags) / len(feasible_flags) if feasible_flags else 0.0

            print(
                f"Final pymoo NSGA-II: "
                f"profit_max={max(profits):.2f}, "
                f"profit_avg={np.mean(profits):.2f}, "
                f"load_min={min(loads):.2f}, "
                f"att_min={min(attitudes):.2f}, "
                f"quality_max={max(qualities):.2f}, "
                f"quality_avg={np.mean(qualities):.2f}, "
                f"HV={hv_history[-1] if hv_history else 0.0:.4f}, "
                f"feasible ratio={fea_ratio:.2f}"
            )
        else:
            print("Final pymoo NSGA-II: no final solution returned.")

        return final_dec, final_raw, final_obj, hv_history


# Backward-compatible aliases.
NSGA2 = NSGA2_PYMOO
NSGA2_PLAIN = NSGA2_PYMOO
