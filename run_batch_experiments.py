import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ALGORITHM_FLAGS = {
    "mode_sdas": "RUN_MODE_SDAS",
    "mode_sdas_dual_archive": "RUN_MODE_SDAS_DUAL_ARCHIVE",
    "mode_sdas_adaptive_subspace": "RUN_MODE_SDAS_ADAPTIVE_SUBSPACE",
    "mode_sdas_dpic_only": "RUN_MODE_SDAS_DPIC_ONLY",
    "mode_sdas_dpic": "RUN_MODE_SDAS_DPIC",
    "mode_sdas_dpic_dual_archive": "RUN_MODE_SDAS_DPIC_DUAL_ARCHIVE",
    "nsga2": "RUN_NSGA2",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", type=int, default=[100, 300, 500, 1000])
    parser.add_argument("--algorithms", nargs="+", default=["all"])
    parser.add_argument("--num-runs", type=int)
    parser.add_argument("--pop-size", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--param", action="append", default=[])
    parser.add_argument("--log-root", type=Path, default=Path("experiment_logs"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--clean-results", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_algorithms(raw):
    if not raw or raw == ["all"] or "all" in raw:
        return list(ALGORITHM_FLAGS.keys())
    unknown = [name for name in raw if name not in ALGORITHM_FLAGS]
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}")
    return raw


def parse_params(items):
    params = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--param requires KEY=VALUE, got {item}")
        key, value = item.split("=", 1)
        params[key.strip()] = value.strip()
    return params


def preflight(base_dir, cases):
    missing = []
    for size in cases:
        data_dir = base_dir / "CSV_Data" / f"s5_t{size}"
        expected = [data_dir / "task_info_standard.csv"]
        expected.extend(data_dir / f"outputattitude_{i}.csv" for i in range(1, 6))
        expected.extend(data_dir / f"outputtimewindow_{i}.csv" for i in range(1, 6))
        missing.extend(path for path in expected if not path.exists())
    if missing:
        raise FileNotFoundError("\n".join(str(path) for path in missing))


def make_base_env(args):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    env["RESULTS_ROOT"] = str(args.results_root)

    optional = {
        "NUM_RUNS": args.num_runs,
        "POP_SIZE": args.pop_size,
        "GENERATIONS": args.generations,
    }
    for key, value in optional.items():
        if value is not None:
            env[key] = str(value)

    env.update(parse_params(args.param))
    return env


def env_for_run(base_env, num_tasks, algorithm):
    env = base_env.copy()
    env["NUM_SATS"] = "5"
    env["NUM_TASKS"] = str(num_tasks)
    for flag in ALGORITHM_FLAGS.values():
        env[flag] = "0"
    env[ALGORITHM_FLAGS[algorithm]] = "1"
    return env


def clean_case_results(base_dir, results_root, cases, algorithms):
    for size in cases:
        case_dir = base_dir / results_root / f"s5_t{size}"
        if not case_dir.exists():
            continue
        for algorithm in algorithms:
            patterns = [
                f"summary_*runs_{algorithm}.csv",
                f"hv_avg_*runs_{algorithm}.csv",
                f"final_objs_{algorithm}.csv",
                f"hv_avg_convergence_{algorithm}.png",
            ]
            for pattern in patterns:
                for path in case_dir.glob(pattern):
                    path.unlink()
        for filename in ("unified_hv_from_saved_final_objs.csv", "unified_hv_bounds.csv"):
            path = case_dir / filename
            if path.exists():
                path.unlink()


def summary_exists(base_dir, results_root, num_tasks, algorithm):
    case_dir = base_dir / results_root / f"s5_t{num_tasks}"
    return any(case_dir.glob(f"summary_*runs_{algorithm}.csv"))


def write_status_header(path):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "algorithm", "started_at", "finished_at", "return_code", "log_file"])


def append_status(path, case_tag, algorithm, started_at, finished_at, return_code, log_file):
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([case_tag, algorithm, started_at, finished_at, return_code, log_file])


def run_one(base_dir, log_dir, status_file, base_env, num_tasks, algorithm):
    case_tag = f"s5_t{num_tasks}"
    case_log_dir = log_dir / case_tag
    case_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = case_log_dir / f"{algorithm}.log"

    started_at = datetime.now().isoformat(timespec="seconds")
    print(f"[{started_at}] START {case_tag} / {algorithm}", flush=True)
    print(f"  log: {log_file}", flush=True)

    env = env_for_run(base_env, num_tasks, algorithm)
    with log_file.open("w", encoding="utf-8", newline="") as f:
        f.write(f"START {case_tag} / {algorithm}: {started_at}\n")
        f.write(f"COMMAND {sys.executable} -u main.py\n\n")
        f.flush()
        completed = subprocess.run(
            [sys.executable, "-u", "main.py"],
            cwd=base_dir,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        finished_at = datetime.now().isoformat(timespec="seconds")
        f.write(f"\nEND {case_tag} / {algorithm}: {finished_at}\n")
        f.write(f"RETURN_CODE {completed.returncode}\n")

    append_status(status_file, case_tag, algorithm, started_at, finished_at, completed.returncode, log_file)
    print(f"[{finished_at}] END {case_tag} / {algorithm} return_code={completed.returncode}", flush=True)
    return completed.returncode


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    algorithms = selected_algorithms(args.algorithms)
    preflight(base_dir, args.cases)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = base_dir / args.log_root / run_id
    status_file = log_dir / "status.csv"

    print("Batch experiment plan", flush=True)
    print(f"  cases: {', '.join(f's5_t{x}' for x in args.cases)}", flush=True)
    print(f"  algorithms: {', '.join(algorithms)}", flush=True)
    print(f"  log_dir: {log_dir}", flush=True)
    print(f"  results_root: {base_dir / args.results_root}", flush=True)
    print(f"  clean_results: {args.clean_results}", flush=True)
    print(f"  skip_existing: {args.skip_existing}", flush=True)

    if args.dry_run:
        return

    if args.clean_results:
        clean_case_results(base_dir, args.results_root, args.cases, algorithms)

    log_dir.mkdir(parents=True, exist_ok=True)
    write_status_header(status_file)
    base_env = make_base_env(args)

    for num_tasks in args.cases:
        for algorithm in algorithms:
            if args.skip_existing and summary_exists(base_dir, args.results_root, num_tasks, algorithm):
                print(f"SKIP s5_t{num_tasks} / {algorithm}: summary exists", flush=True)
                continue
            return_code = run_one(base_dir, log_dir, status_file, base_env, num_tasks, algorithm)
            if return_code != 0 and not args.keep_going:
                raise SystemExit(return_code)

    print("Batch experiment completed.", flush=True)
    print(f"status_file: {status_file}", flush=True)


if __name__ == "__main__":
    main()
