from pathlib import Path

import numpy as np
import pandas as pd

from modules.mode_sdas_dpic_archive import MODE_SDAS_DPIC_DUAL_ARCHIVE
from modules.mode_sdas_dpic_only import MODE_SDAS_DPIC_ONLY
from modules.mode_sdas_dual_archive import MODE_SDAS_DUAL_ARCHIVE
from modules.problem_model import SatelliteSchedulingProblem
from modules.utils_moea import OBJ_KEYS, compute_global_bounds, compute_unified_hv


TEST_CASES = [
    (5, 100),
    (5, 300),
]


RUN_MODE_SDAS_DUAL_ARCHIVE = False
RUN_MODE_SDAS_DPIC_ONLY = True
RUN_MODE_SDAS_DPIC_DUAL_ARCHIVE = True


POP_SIZE = 100
GENERATIONS = 100
NUM_RUNS = 8


MODE_NSUB = 21
MODE_L = 7
MODE_RMEM = 10
MODE_CR = 0.85
MODE_F = 0.45
MODE_EXECUTE_PROB = 0.03

"""
DPIC_ALPHA = 0.02
DPIC_SIGMA = 0.003
DPIC_GAMMA = 0.85

"""

DPIC_PARAM_CONFIGS = [
    {
        "name": "a0p01_s0p0015_g0p90",
        "alpha": 0.01,
        "sigma": 0.0015,
        "gamma": 0.90,
    },
    {
        "name": "a0p02_s0p002_g0p90",
        "alpha": 0.05,
        "sigma": 0.002,
        "gamma": 0.90,
    },
    {
        "name": "paper_a0p08_s0p01_g0p90",
        "alpha": 0.08,
        "sigma": 0.0025,
        "gamma": 0.90,
    },
    {
        "name": "paper_a0p1_s0p01_g0p90",
        "alpha": 0.1,
        "sigma": 0.003,
        "gamma": 0.90,
    },
]


ARCHIVE_PROB = 0.15
FEASIBLE_ARCHIVE_SIZE = 200
INFEASIBLE_ARCHIVE_SIZE = 200


BASELINE_BUDGET = "8\u8f6e100\u4ee3"


def budget_label():
    return f"{NUM_RUNS}\u8f6e{GENERATIONS}\u4ee3"


def algo_label(algo_tag):
    if algo_tag.lower() == "nsga2":
        return "NSGA2"
    return algo_tag


def file_tag(algo_tag, param_config=None):
    tag = f"{algo_label(algo_tag)}-{budget_label()}"
    if param_config is not None:
        tag = f"{tag}-{param_config['name']}"
    return tag


def parse_final_objs_file(path):
    stem = path.stem.replace("final_objs_", "", 1)
    if "-" in stem:
        algorithm, budget = stem.rsplit("-", 1)
    else:
        algorithm, budget = stem, "unknown"
    return algorithm, budget


def load_final_obj_file(path):
    df = pd.read_csv(path)
    return [
        {key: float(row[key]) for key in OBJ_KEYS}
        for _, row in df.iterrows()
    ]


def load_baseline_records(files):
    records = []
    for file in sorted(files):
        algorithm, budget = parse_final_objs_file(file)
        obj_list = load_final_obj_file(file)
        if obj_list:
            records.append({
                "Algorithm": algorithm,
                "Budget": budget,
                "Source": "baseline",
                "Param_Set": "baseline",
                "DPIC_ALPHA": "",
                "DPIC_SIGMA": "",
                "DPIC_GAMMA": "",
                "ObjList": obj_list,
            })
    return records


def compute_hv_from_records(records, output_file):
    saved_objs = {}
    meta = {}

    for record in records:
        obj_list = record["ObjList"]
        if not obj_list:
            continue

        key = (
            f"{record['Algorithm']}-"
            f"{record['Budget']}-"
            f"{record['Source']}-"
            f"{record['Param_Set']}"
        )
        saved_objs[key] = obj_list
        meta[key] = {
            "Algorithm": record["Algorithm"],
            "Budget": record["Budget"],
            "Source": record["Source"],
            "Param_Set": record["Param_Set"],
            "DPIC_ALPHA": record["DPIC_ALPHA"],
            "DPIC_SIGMA": record["DPIC_SIGMA"],
            "DPIC_GAMMA": record["DPIC_GAMMA"],
        }

    if len(saved_objs) < 2:
        print(f"Skip unified HV for {output_file}: fewer than 2 result records.")
        return

    ideal, nadir, denom = compute_global_bounds(list(saved_objs.values()))

    rows = []
    for key, obj_list in saved_objs.items():
        row = dict(meta[key])
        row["Unified_HV"] = compute_unified_hv(obj_list, ideal, denom)
        rows.append(row)

    df_hv = pd.DataFrame(rows)
    df_hv = df_hv.sort_values("Unified_HV", ascending=False)
    df_hv.to_csv(output_file, index=False, encoding="utf-8-sig")

    bounds_file = output_file.with_name(output_file.stem + "_bounds.csv")
    pd.DataFrame({
        "Objective": OBJ_KEYS,
        "Ideal": ideal,
        "Nadir": nadir,
        "Denom": denom,
    }).to_csv(bounds_file, index=False, encoding="utf-8-sig")

    print(f"Unified HV saved to {output_file}")
    print(f"Unified HV bounds saved to {bounds_file}")
    print(df_hv)


def compute_test_and_comparison_hv(case_results_dir, output_dir, test_records):
    test_budget = budget_label()

    compute_hv_from_records(
        test_records,
        output_dir / f"unified_hv_from_saved_final_objs-all_params-{test_budget}.csv",
    )

    baseline_files = list(case_results_dir.glob(f"final_objs_*-{BASELINE_BUDGET}.csv"))
    baseline_records = load_baseline_records(baseline_files)
    if test_budget == BASELINE_BUDGET:
        compare_name = f"unified_hv_compare_all_params_vs_baseline_{BASELINE_BUDGET}.csv"
    else:
        compare_name = f"unified_hv_compare_all_params_{test_budget}_vs_{BASELINE_BUDGET}.csv"
    compute_hv_from_records(
        baseline_records + test_records,
        output_dir / compare_name,
    )


def filter_feasible_final_results(alg, final_dec, final_raw, final_obj):
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


def run_one_algorithm(algo_tag, algo_name, build_algorithm, results_dir, param_config):
    print(f"\nRunning algorithm: {algo_name}")
    print(
        "Param set: "
        f"{param_config['name']} "
        f"(alpha={param_config['alpha']}, "
        f"sigma={param_config['sigma']}, "
        f"gamma={param_config['gamma']})"
    )

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
        final_dec, final_raw, final_obj, _ = alg.run()

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

        print(f"Final feasible solutions used for summary/HV: {feasible_count}/{candidate_count}")

        if len(final_raw) == 0:
            best_profits.append(np.nan)
            best_loads.append(np.nan)
            best_attitudes.append(np.nan)
            best_qualities.append(np.nan)
            print("Run best: no feasible final solution; summary metrics set to NaN")
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

    tag = file_tag(algo_tag, param_config)
    summary_dict = {
        "Run": list(range(1, NUM_RUNS + 1)) + ["Average", "Std"],
        "Param_Set": [param_config["name"]] * (NUM_RUNS + 2),
        "DPIC_ALPHA": [param_config["alpha"]] * (NUM_RUNS + 2),
        "DPIC_SIGMA": [param_config["sigma"]] * (NUM_RUNS + 2),
        "DPIC_GAMMA": [param_config["gamma"]] * (NUM_RUNS + 2),
        "Profit_max": best_profits + [safe_nanmean(best_profits), safe_nanstd(best_profits)],
        "Load_min": best_loads + [safe_nanmean(best_loads), safe_nanstd(best_loads)],
        "Attitude_min": best_attitudes + [safe_nanmean(best_attitudes), safe_nanstd(best_attitudes)],
        "Quality_max": best_qualities + [safe_nanmean(best_qualities), safe_nanstd(best_qualities)],
        "Final_feasible_count": final_feasible_counts + [
            safe_nanmean(final_feasible_counts),
            safe_nanstd(final_feasible_counts),
        ],
        "Final_candidate_count": final_candidate_counts + [
            safe_nanmean(final_candidate_counts),
            safe_nanstd(final_candidate_counts),
        ],
    }

    summary_file = results_dir / f"summary_{tag}.csv"
    pd.DataFrame(summary_dict).to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"Summary of best objectives saved to {summary_file}")

    flat_final_objs = [
        obj
        for final_objs in final_objs_all_runs
        for obj in final_objs
    ]
    return {
        "Algorithm": algo_label(algo_tag),
        "Budget": budget_label(),
        "Source": "test",
        "Param_Set": param_config["name"],
        "DPIC_ALPHA": param_config["alpha"],
        "DPIC_SIGMA": param_config["sigma"],
        "DPIC_GAMMA": param_config["gamma"],
        "ObjList": flat_final_objs,
    }


def build_problem(base_dir, num_sats, num_tasks):
    case_tag = f"s{num_sats}_t{num_tasks}"
    data_dir = base_dir / "CSV_Data" / case_tag
    task_file = data_dir / "task_info_standard.csv"
    attitude_files = [data_dir / f"outputattitude_{i}.csv" for i in range(1, num_sats + 1)]
    timewindow_files = [data_dir / f"outputtimewindow_{i}.csv" for i in range(1, num_sats + 1)]

    missing_files = [f for f in attitude_files + timewindow_files + [task_file] if not f.exists()]
    if missing_files:
        raise FileNotFoundError("Missing data files:\n" + "\n".join(str(f) for f in missing_files))

    problem = SatelliteSchedulingProblem(
        str(task_file),
        [str(f) for f in attitude_files],
        [str(f) for f in timewindow_files],
    )
    return case_tag, problem


def run_case(base_dir, num_sats, num_tasks):
    case_tag, problem = build_problem(base_dir, num_sats, num_tasks)
    case_results_dir = base_dir / "results" / case_tag
    output_dir = case_results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_front_file = case_results_dir / "reference_front.csv"
    ref_front_file = str(ref_front_file) if ref_front_file.exists() else None

    print(f"\n========== TEST CASE {case_tag} ==========")
    print(f"Outputs: {output_dir}")
    print(f"Budget: {budget_label()}")

    test_records = []

    for param_config in DPIC_PARAM_CONFIGS:
        print(
            f"\n----- PARAM SET {param_config['name']} "
            f"(alpha={param_config['alpha']}, "
            f"sigma={param_config['sigma']}, "
            f"gamma={param_config['gamma']}) -----"
        )

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
                    ref_front_file=ref_front_file,
                )

            record = run_one_algorithm(
                "mode_sdas_dual_archive",
                "MODE-SDAS + Dual Archive",
                build_mode_sdas_dual_archive,
                output_dir,
                param_config,
            )
            test_records.append(record)

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
                    alpha=param_config["alpha"],
                    sigma=param_config["sigma"],
                    gamma=param_config["gamma"],
                    execute_prob=MODE_EXECUTE_PROB,
                    ref_front_file=ref_front_file,
                )

            record = run_one_algorithm(
                "mode_sdas_dpic_only",
                "MODE-SDAS + DPIC only",
                build_mode_sdas_dpic_only,
                output_dir,
                param_config,
            )
            test_records.append(record)

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
                    alpha=param_config["alpha"],
                    sigma=param_config["sigma"],
                    gamma=param_config["gamma"],
                    execute_prob=MODE_EXECUTE_PROB,
                    ref_front_file=ref_front_file,
                )

            record = run_one_algorithm(
                "mode_sdas_dpic_dual_archive",
                "MODE-SDAS + DPIC + Dual Archive",
                build_mode_sdas_dpic_dual_archive,
                output_dir,
                param_config,
            )
            test_records.append(record)

    compute_test_and_comparison_hv(case_results_dir, output_dir, test_records)


def main():
    base_dir = Path(__file__).parent
    for num_sats, num_tasks in TEST_CASES:
        run_case(base_dir, num_sats, num_tasks)
    print("\nAll test tasks completed.")


if __name__ == "__main__":
    main()
