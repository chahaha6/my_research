import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from modules.problem_model import SatelliteSchedulingProblem
from modules.mode_sdas import MODE_SDAS
from modules.mode_sdas_dpic import MODE_SDAS_DPIC
from modules.mode_sdas_dpic_only import MODE_SDAS_DPIC_ONLY
from modules.mode_sdas_dual_archive import MODE_SDAS_DUAL_ARCHIVE
from modules.mode_sdas_adaptive_subspace import MODE_SDAS_ADAPTIVE_SUBSPACE
from modules.mode_sdas_dpic_archive import MODE_SDAS_DPIC_DUAL_ARCHIVE
from modules.nsga2_pymoo import NSGA2_PYMOO as NSGA2


# =========================================================
# 统一 HV 工具
# 优先使用 utils_moea.py 里的函数；
# 若尚未添加，则使用 main.py 内置版本。
# =========================================================

OBJ_KEYS = ["profit", "load", "attitude", "quality"]

try:
    from modules.utils_moea import compute_global_bounds, compute_unified_hv
except Exception:
    from pymoo.indicators.hv import Hypervolume
    from modules.utils_moea import non_dominated_sort

    def obj_list_to_matrix(obj_list, obj_keys=OBJ_KEYS):
        if obj_list is None or len(obj_list) == 0:
            return np.empty((0, len(obj_keys)))

        return np.array(
            [[obj[k] for k in obj_keys] for obj in obj_list],
            dtype=float
        )

    def compute_global_bounds(all_obj_lists, obj_keys=OBJ_KEYS):
        all_F = []

        for obj_list in all_obj_lists:
            if obj_list is None or len(obj_list) == 0:
                continue

            F = obj_list_to_matrix(obj_list, obj_keys)
            if F.size > 0:
                all_F.append(F)

        if not all_F:
            raise ValueError("No objective values found for unified HV calculation.")

        all_F = np.vstack(all_F)

        ideal = np.min(all_F, axis=0)
        nadir = np.max(all_F, axis=0)

        denom = nadir - ideal
        denom[denom < 1e-12] = 1.0

        return ideal, nadir, denom

    def compute_unified_hv(obj_list, ideal, denom, obj_keys=OBJ_KEYS, ref_point=None):
        if obj_list is None or len(obj_list) == 0:
            return 0.0

        fronts = non_dominated_sort(obj_list)
        if not fronts or len(fronts[0]) == 0:
            return 0.0

        nd_obj = [obj_list[i] for i in fronts[0]]
        F = obj_list_to_matrix(nd_obj, obj_keys)

        F_norm = (F - ideal) / denom
        F_norm = np.clip(F_norm, 0.0, 1.2)

        if ref_point is None:
            ref_point = np.ones(F_norm.shape[1]) * 1.1

        hv = Hypervolume(ref_point=ref_point)
        return float(hv(F_norm))


# =========================================================
# 运行开关：一次一般只跑一个算法
# 需要跑哪个算法，就把对应开关改成 True。
# =========================================================

#MODE_SDAS原始算法
RUN_MODE_SDAS = False
#MODE_SDAS + 双档案
RUN_MODE_SDAS_DUAL_ARCHIVE = True
#MODE_SDAS + 自适应子空间
RUN_MODE_SDAS_ADAPTIVE_SUBSPACE = False
#MODE_SDAS + DPIC only
RUN_MODE_SDAS_DPIC_ONLY = True
#MODE-SDAS-DPIC Full
RUN_MODE_SDAS_DPIC = False
#MODE-SDAS + DPIC + 双档案
RUN_MODE_SDAS_DPIC_DUAL_ARCHIVE = True
# NSGA-II 基线算法
RUN_NSGA2 = False


# =========================================================
# 数据规模与通用实验参数
# 以后只需要改这里的 NUM_SATS / NUM_TASKS 即可。
# 算法内部参数请在对应 modules/*.py 文件中修改。
# =========================================================

NUM_SATS = 2          # 卫星数量：读取 outputattitude_1~NUM_SATS / outputtimewindow_1~NUM_SATS
NUM_TASKS = 500      # 任务数量：用于结果目录命名；实际任务数由 CSV_Data/task_info_standard.csv 决定

CASE_TAG = f"s{NUM_SATS}_t{NUM_TASKS}"

POP_SIZE = 100
GENERATIONS = 60
NUM_RUNS = 4

# Shared MODE-SDAS-family parameters for fair comparison on the same case.
MODE_NSUB = 21
MODE_L = 7
MODE_RMEM = 10
MODE_CR = 0.85
MODE_F = 0.45
MODE_EXECUTE_PROB = 0.03

DPIC_ALPHA = 0.02
DPIC_SIGMA = 0.003
DPIC_GAMMA = 0.85

ARCHIVE_PROB = 0.15
FEASIBLE_ARCHIVE_SIZE = 200
INFEASIBLE_ARCHIVE_SIZE = 200
ELITE_PROB = 0.25
ELITE_ARCHIVE_SIZE = 200

MODE_ADAPT_CHECK_INTERVAL = 100
MODE_WEIGHT_DELETION_THRESHOLD = 100
MODE_SEMANTIC_ADD_PROB = 0.35
MODE_SEMANTIC_ADD_TRIALS = 30
MODE_SEMANTIC_ADD_MAX = 1
MODE_SEMANTIC_DROP_PROB = 0.01
MODE_SEMANTIC_WINDOW_PROB = 0.15
MODE_FINAL_GREEDY_ADD_MAX = 15
MODE_FINAL_GREEDY_ADD_TRIALS = 120


# =========================================================
# 保存 final_obj，用于之后统一 HV
# =========================================================

def save_final_objs(final_objs_all_runs, results_dir, algo_tag):
    rows = []
    columns = ["Run", "Index"] + OBJ_KEYS

    for run_idx, final_obj in enumerate(final_objs_all_runs, start=1):
        for obj_idx, obj in enumerate(final_obj):
            row = {
                "Run": run_idx,
                "Index": obj_idx,
            }
            for key in OBJ_KEYS:
                row[key] = obj[key]
            rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    out_file = results_dir / f"final_objs_{algo_tag}.csv"
    df.to_csv(out_file, index=False)
    print(f"Final objective values saved to {out_file}")


def load_saved_final_objs(results_dir):
    saved = {}

    for file in sorted(results_dir.glob("final_objs_*.csv")):
        algo_tag = file.stem.replace("final_objs_", "")
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            continue

        obj_list = []
        for _, row in df.iterrows():
            obj = {key: float(row[key]) for key in OBJ_KEYS}
            obj_list.append(obj)

        if obj_list:
            saved[algo_tag] = obj_list

    return saved


def compute_unified_hv_from_saved(results_dir):
    saved_objs = load_saved_final_objs(results_dir)

    if len(saved_objs) < 2:
        print("\n当前 results/ 中少于 2 个算法的 final_objs 文件，暂不计算统一 HV。")
        print("至少需要两个文件，例如：")
        print("  final_objs_mode_sdas.csv")
        print("  final_objs_mode_sdas_dpic_dual_archive.csv")
        return

    ideal, nadir, denom = compute_global_bounds(list(saved_objs.values()))

    rows = []
    for algo_tag, obj_list in saved_objs.items():
        hv_value = compute_unified_hv(obj_list, ideal, denom)
        rows.append({
            "Algorithm": algo_tag,
            "Unified_HV": hv_value
        })

    df_hv = pd.DataFrame(rows)
    hv_file = results_dir / "unified_hv_from_saved_final_objs.csv"
    df_hv.to_csv(hv_file, index=False)

    bounds_df = pd.DataFrame({
        "Objective": OBJ_KEYS,
        "Ideal": ideal,
        "Nadir": nadir,
        "Denom": denom
    })
    bounds_file = results_dir / "unified_hv_bounds.csv"
    bounds_df.to_csv(bounds_file, index=False)

    print(f"\nUnified HV saved to {hv_file}")
    print(f"Unified HV bounds saved to {bounds_file}")
    print(df_hv)


# =========================================================
# 单算法多轮运行
# =========================================================

def filter_feasible_final_results(alg, final_dec, final_raw, final_obj):
    """
    Keep only feasible final solutions for summary and final HV calculation.
    """
    if not hasattr(alg, "decode") or not hasattr(alg, "problem"):
        return final_dec, final_raw, final_obj, None

    if final_dec is None or len(final_raw) == 0:
        return final_dec, final_raw, final_obj, []

    dec_array = np.asarray(final_dec, dtype=float)
    if dec_array.ndim == 1:
        dec_array = dec_array.reshape(1, -1)

    feasible_flags = []
    for dec in dec_array:
        sol = alg.decode(dec)
        _, feasible = alg.problem.evaluate_solution(sol)
        feasible_flags.append(bool(feasible))

    if len(feasible_flags) != len(final_raw):
        raise ValueError(
            "Final feasibility flags do not match final objective count: "
            f"{len(feasible_flags)} vs {len(final_raw)}"
        )

    feasible_idx = [i for i, flag in enumerate(feasible_flags) if flag]
    if not feasible_idx:
        return dec_array[:0], [], [], feasible_flags

    return (
        dec_array[feasible_idx],
        [final_raw[i] for i in feasible_idx],
        [final_obj[i] for i in feasible_idx],
        feasible_flags,
    )


def safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def safe_nanstd(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanstd(arr))


def run_one_algorithm(algo_tag, algo_name, build_algorithm, results_dir):
    print(f"\n运行算法：{algo_name}")

    hv_histories = []
    best_profits = []
    best_loads = []
    best_attitudes = []
    best_qualities = []
    final_objs_all_runs = []
    final_feasible_counts = []
    final_candidate_counts = []

    for run_idx in range(1, NUM_RUNS + 1):
        print(f"\n========== {algo_name} Run {run_idx}/{NUM_RUNS} ==========")

        alg = build_algorithm()

        final_dec, final_raw, final_obj, hv_history = alg.run()

        hv_histories.append(np.array(hv_history, dtype=float))

        final_dec, final_raw, final_obj, feasible_flags = filter_feasible_final_results(
            alg,
            final_dec,
            final_raw,
            final_obj,
        )

        if feasible_flags is None:
            stored_flags = getattr(alg, "last_final_feasible_flags", None)
            if stored_flags is None:
                feasible_count = len(final_raw)
                candidate_count = len(final_raw)
            else:
                feasible_count = int(sum(stored_flags))
                candidate_count = len(stored_flags)
        else:
            feasible_count = int(sum(feasible_flags))
            candidate_count = len(feasible_flags)

        final_feasible_counts.append(feasible_count)
        final_candidate_counts.append(candidate_count)
        final_objs_all_runs.append(final_obj)

        print(
            f"Final feasible solutions used for summary/HV: "
            f"{feasible_count}/{candidate_count}"
        )

        if len(final_raw) == 0:
            best_profits.append(np.nan)
            best_loads.append(np.nan)
            best_attitudes.append(np.nan)
            best_qualities.append(np.nan)
            print(
                f"Run {run_idx} best: no feasible final solution; "
                "summary metrics set to NaN"
            )
            continue

        profits = [raw["total_profit"] for raw in final_raw]
        loads = [raw["load_balance"] for raw in final_raw]
        attitudes = [raw["attitude_manoeuvre"] for raw in final_raw]
        qualities = [raw["image_quality"] for raw in final_raw]

        best_profits.append(max(profits))
        best_loads.append(min(loads))
        best_attitudes.append(min(attitudes))
        best_qualities.append(max(qualities))

        print(
            f"Run {run_idx} best: "
            f"profit={best_profits[-1]:.1f}, "
            f"load={best_loads[-1]:.2f}, "
            f"att={best_attitudes[-1]:.2f}, "
            f"quality={best_qualities[-1]:.2f}"
        )

    # ================= 保存 HV 平均曲线 =================
    hv_array = np.array(hv_histories)
    mean_hv = np.mean(hv_array, axis=0)

    hv_avg_file = results_dir / f"hv_avg_{NUM_RUNS}runs_{algo_tag}.csv"
    np.savetxt(hv_avg_file, mean_hv, delimiter=",")
    print(f"\nAverage HV history saved to {hv_avg_file}")

    # ================= 保存 summary =================
    avg_profit = safe_nanmean(best_profits)
    avg_load = safe_nanmean(best_loads)
    avg_att = safe_nanmean(best_attitudes)
    avg_quality = safe_nanmean(best_qualities)

    std_profit = safe_nanstd(best_profits)
    std_load = safe_nanstd(best_loads)
    std_att = safe_nanstd(best_attitudes)
    std_quality = safe_nanstd(best_qualities)

    summary_dict = {
        "Run": list(range(1, NUM_RUNS + 1)) + ["Average", "Std"],
        "Profit_max": best_profits + [avg_profit, std_profit],
        "Load_min": best_loads + [avg_load, std_load],
        "Attitude_min": best_attitudes + [avg_att, std_att],
        "Quality_max": best_qualities + [avg_quality, std_quality],
        "Final_feasible_count": final_feasible_counts + [
            safe_nanmean(final_feasible_counts),
            safe_nanstd(final_feasible_counts),
        ],
        "Final_candidate_count": final_candidate_counts + [
            safe_nanmean(final_candidate_counts),
            safe_nanstd(final_candidate_counts),
        ],
    }

    df_summary = pd.DataFrame(summary_dict)
    summary_file = results_dir / f"summary_{NUM_RUNS}runs_{algo_tag}.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"Summary of best objectives saved to {summary_file}")

    # ================= 保存 final_obj =================
    save_final_objs(final_objs_all_runs, results_dir, algo_tag)

    # ================= 绘制该算法单独 HV 曲线 =================
    plt.figure(figsize=(8, 5))
    gens = range(1, len(mean_hv) + 1)
    label = algo_tag.replace("_", " ").upper()

    plt.plot(
        gens,
        mean_hv,
        "s-",
        label=f"Average HV ({label}, {NUM_RUNS} runs)",
        markersize=4
    )

    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title(f"Average HV Convergence ({label})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = results_dir / f"hv_avg_convergence_{algo_tag}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Average convergence plot saved to {plot_path}")


# =========================================================
# main
# =========================================================

def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "CSV_Data" / CASE_TAG
    local_data_dir = base_dir / "Local_Data" / f"area_{CASE_TAG}"

    task_file = data_dir / "task_info_standard.csv"
    attitude_files = [data_dir / f"outputattitude_{i}.csv" for i in range(1, NUM_SATS + 1)]
    timewindow_files = [data_dir / f"outputtimewindow_{i}.csv" for i in range(1, NUM_SATS + 1)]

    missing_files = [f for f in attitude_files + timewindow_files + [task_file] if not f.exists()]
    if missing_files:
        raise FileNotFoundError(
            "以下数据文件不存在，请先运行数据预处理脚本：\n"
            + "\n".join(str(f) for f in missing_files)
        )

    problem = SatelliteSchedulingProblem(
        str(task_file),
        [str(f) for f in attitude_files],
        [str(f) for f in timewindow_files]
    )

    # 不同数据规模单独保存，避免 s10_t100 和 s5_t1000 的 final_objs 混在一起计算统一 HV
    results_dir = base_dir / "results" / CASE_TAG
    results_dir.mkdir(parents=True, exist_ok=True)

    ref_front_file = results_dir / "reference_front.csv"
    if not ref_front_file.exists():
        ref_front_file = None
    else:
        ref_front_file = str(ref_front_file)

    # =====================================================
    # MODE-SDAS
    # =====================================================
    if RUN_MODE_SDAS:
        def build_mode_sdas():
            return MODE_SDAS(
                problem,
                pop_size=POP_SIZE,
                generations=GENERATIONS,
                Nsub=MODE_NSUB,
                L=MODE_L,
                CR=MODE_CR,
                F=MODE_F,
                execute_prob=MODE_EXECUTE_PROB,
                ref_front_file=ref_front_file
            )

        run_one_algorithm(
            algo_tag="mode_sdas",
            algo_name="MODE-SDAS",
            build_algorithm=build_mode_sdas,
            results_dir=results_dir
        )

    # =====================================================
    # MODE-SDAS + Dual Archive
    # =====================================================
    if RUN_MODE_SDAS_DUAL_ARCHIVE:
        def build_mode_sdas_dual_archive():
            return MODE_SDAS_DUAL_ARCHIVE(
                problem,
                pop_size=POP_SIZE,
                generations=GENERATIONS,
                Nsub=MODE_NSUB,
                L=MODE_L,
                CR=MODE_CR,
                F=MODE_F,
                archive_prob=ARCHIVE_PROB,
                feasible_archive_size=FEASIBLE_ARCHIVE_SIZE,
                infeasible_archive_size=INFEASIBLE_ARCHIVE_SIZE,
                execute_prob=MODE_EXECUTE_PROB,
                ref_front_file=ref_front_file
            )

        run_one_algorithm(
            algo_tag="mode_sdas_dual_archive",
            algo_name="MODE-SDAS + Dual Archive",
            build_algorithm=build_mode_sdas_dual_archive,
            results_dir=results_dir
        )

    # =====================================================
    # MODE-SDAS + Adaptive Subspace
    # =====================================================
    if RUN_MODE_SDAS_ADAPTIVE_SUBSPACE:
        def build_mode_sdas_adaptive_subspace():
            return MODE_SDAS_ADAPTIVE_SUBSPACE(
                problem,
                pop_size=POP_SIZE,
                generations=GENERATIONS,
                Nsub=MODE_NSUB,
                L=MODE_L,
                Rmem=MODE_RMEM,
                CR=MODE_CR,
                F=MODE_F,
                execute_prob=MODE_EXECUTE_PROB,
                ref_front_file=ref_front_file
            )

        run_one_algorithm(
            algo_tag="mode_sdas_adaptive_subspace",
            algo_name="MODE-SDAS + Adaptive Subspace",
            build_algorithm=build_mode_sdas_adaptive_subspace,
            results_dir=results_dir
        )

    # =====================================================
    # MODE-SDAS + DPIC only
    # =====================================================
    if RUN_MODE_SDAS_DPIC_ONLY:
        def build_mode_sdas_dpic_only():
            return MODE_SDAS_DPIC_ONLY(
                problem,
                pop_size=POP_SIZE,
                generations=GENERATIONS,
                Nsub=MODE_NSUB,
                L=MODE_L,
                Rmem=MODE_RMEM,
                CR=MODE_CR,
                F=MODE_F,
                alpha=DPIC_ALPHA,
                sigma=DPIC_SIGMA,
                gamma=DPIC_GAMMA,
                execute_prob=MODE_EXECUTE_PROB,
                ref_front_file=ref_front_file
            )

        run_one_algorithm(
            algo_tag="mode_sdas_dpic_only",
            algo_name="MODE-SDAS + DPIC only",
            build_algorithm=build_mode_sdas_dpic_only,
            results_dir=results_dir
        )

    # =====================================================
    # MODE-SDAS-DPIC Full
    # =====================================================
    if RUN_MODE_SDAS_DPIC:
        def build_mode_sdas_dpic():
            return MODE_SDAS_DPIC(
                problem,
                pop_size=POP_SIZE,
                generations=GENERATIONS,
                Nsub=MODE_NSUB,
                L=MODE_L,
                Rmem=MODE_RMEM,
                CR=MODE_CR,
                F=MODE_F,
                alpha=DPIC_ALPHA,
                sigma=DPIC_SIGMA,
                gamma=DPIC_GAMMA,
                elite_prob=ELITE_PROB,
                archive_size=ELITE_ARCHIVE_SIZE,
                execute_prob=MODE_EXECUTE_PROB,
                check_interval=MODE_ADAPT_CHECK_INTERVAL,
                weight_deletion_threshold=MODE_WEIGHT_DELETION_THRESHOLD,
                semantic_add_prob=MODE_SEMANTIC_ADD_PROB,
                semantic_add_trials=MODE_SEMANTIC_ADD_TRIALS,
                semantic_add_max=MODE_SEMANTIC_ADD_MAX,
                semantic_drop_prob=MODE_SEMANTIC_DROP_PROB,
                semantic_window_prob=MODE_SEMANTIC_WINDOW_PROB,
                final_greedy_add_max=MODE_FINAL_GREEDY_ADD_MAX,
                final_greedy_add_trials=MODE_FINAL_GREEDY_ADD_TRIALS,
                ref_front_file=ref_front_file
            )

        run_one_algorithm(
            algo_tag="mode_sdas_dpic",
            algo_name="MODE-SDAS-DPIC Full",
            build_algorithm=build_mode_sdas_dpic,
            results_dir=results_dir
        )

    # =====================================================
    # MODE-SDAS + DPIC + Dual Archive
    # =====================================================
    if RUN_MODE_SDAS_DPIC_DUAL_ARCHIVE:
        def build_mode_sdas_dpic_dual_archive():
            return MODE_SDAS_DPIC_DUAL_ARCHIVE(
                problem,
                pop_size=POP_SIZE,
                generations=GENERATIONS,
                Nsub=MODE_NSUB,
                L=MODE_L,
                CR=MODE_CR,
                F=MODE_F,
                archive_prob=ARCHIVE_PROB,
                feasible_archive_size=FEASIBLE_ARCHIVE_SIZE,
                infeasible_archive_size=INFEASIBLE_ARCHIVE_SIZE,
                alpha=DPIC_ALPHA,
                sigma=DPIC_SIGMA,
                gamma=DPIC_GAMMA,
                execute_prob=MODE_EXECUTE_PROB,
                ref_front_file=ref_front_file
            )

        run_one_algorithm(
            algo_tag="mode_sdas_dpic_dual_archive",
            algo_name="MODE-SDAS + DPIC + Dual Archive",
            build_algorithm=build_mode_sdas_dpic_dual_archive,
            results_dir=results_dir
        )

    # =====================================================
    # NSGA-II
    # =====================================================
    if RUN_NSGA2:
        def build_nsga2():
            return NSGA2(
                problem,
                pop_size=POP_SIZE,
                generations=GENERATIONS,
                execute_prob=MODE_EXECUTE_PROB,
                ref_front_file=ref_front_file
            )

        run_one_algorithm(
            algo_tag="nsga2",
            algo_name="NSGA-II",
            build_algorithm=build_nsga2,
            results_dir=results_dir
        )

    # =====================================================
    # 统一 HV：读取 results 中已有 final_objs_*.csv 计算
    # =====================================================
    compute_unified_hv_from_saved(results_dir)

    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()
