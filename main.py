import numpy as np
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from modules.problem_model import SatelliteSchedulingProblem
from modules.nsga2 import NSGA2
from modules.mode_sdas_pic import MODE_SDAS_PIC
from modules.utils_moea import non_dominated_sort

# 统一参数（可调整）
POP_SIZE = 20
GENERATIONS = 30

def run_algorithm(alg, name):
    """运行算法，返回 final_dec, final_raw, final_obj, hv_history"""
    print(f"\n========== Running {name} ==========")
    final_dec, final_raw, final_obj, hv_history = alg.run()
    return final_dec, final_raw, final_obj, hv_history

def save_results(alg, final_dec, final_raw, hv_history, name, results_dir):
    """保存 HV 历史、最优解、IGD（如果可能）"""
    # 保存 HV 历史
    hv_file = results_dir / f"hv_history_{name}.csv"
    np.savetxt(hv_file, np.array(hv_history), delimiter=',')
    print(f"[{name}] HV history saved to {hv_file}")

    # 保存最优解（按收益）
    best_idx = np.argmax([raw['total_profit'] for raw in final_raw])
    best_dec = final_dec[best_idx]
    best_sol = alg.decode(best_dec)
    df = pd.DataFrame(best_sol, columns=["task_id", "window_idx", "sat_idx"])
    sol_file = results_dir / f"final_solution_{name}.csv"
    df.to_csv(sol_file, index=False)
    print(f"[{name}] Final solution saved to {sol_file}")

    # 若算法包含 IGD 指示器，则计算
    if hasattr(alg, 'igd_indicator') and alg.igd_indicator is not None:
        min_objs = [alg.evaluate(dec)[0] for dec in final_dec]
        fronts = non_dominated_sort(min_objs)
        nd_F = alg.get_obj_matrix([min_objs[i] for i in fronts[0]])
        igd_val = alg.igd_indicator(nd_F)
        print(f"[{name}] IGD = {igd_val:.4f}")
        with open(results_dir / f"igd_{name}.txt", 'w') as f:
            f.write(f"{igd_val}\n")

def main():
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "CSV_Data"

    # 数据路径
    task_file = DATA_DIR / "task_info_standard.csv"
    attitude_files = [DATA_DIR / f"outputattitude_{i}.csv" for i in range(1, 11)]
    timewindow_files = [DATA_DIR / f"outputtimewindow_{i}.csv" for i in range(1, 11)]

    problem = SatelliteSchedulingProblem(str(task_file),
                                         [str(f) for f in attitude_files],
                                         [str(f) for f in timewindow_files])

    # 可选 IGD 参考前沿
    ref_front_file = BASE_DIR / "results" / "reference_front.csv"
    if not ref_front_file.exists():
        ref_front_file = None

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # -------------------- NSGA-II --------------------
    nsga2 = NSGA2(problem,
                  pop_size=POP_SIZE,
                  generations=GENERATIONS,
                  crossover_prob=0.9,
                  mutation_prob=0.1,
                  ref_front_file=ref_front_file)
    nsga2_final_dec, nsga2_final_raw, nsga2_final_obj, nsga2_hv = run_algorithm(nsga2, "NSGA-II")
    save_results(nsga2, nsga2_final_dec, nsga2_final_raw, nsga2_hv, "nsga2", results_dir)

    # -------------------- MODE-SDAS-PIC --------------------
    mode_pic = MODE_SDAS_PIC(problem,
                             pop_size=POP_SIZE,
                             generations=GENERATIONS,
                             Nsub=5, L=3, Rmem=5, CR=0.5, F=0.5,
                             alpha=0.15, sigma=0.01, gamma=0.7,
                             ref_front_file=ref_front_file)
    mode_final_dec, mode_final_raw, mode_final_obj, mode_hv = run_algorithm(mode_pic, "MODE-SDAS-PIC")
    save_results(mode_pic, mode_final_dec, mode_final_raw, mode_hv, "mode_sdas_pic", results_dir)

    # -------------------- 绘制 HV 对比图 --------------------
    plt.figure(figsize=(8, 5))
    gens = range(1, len(nsga2_hv) + 1)
    plt.plot(gens, nsga2_hv, 'o-', label='NSGA-II', markersize=4)
    plt.plot(gens, mode_hv, 's-', label='MODE-SDAS-PIC', markersize=4)
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.title('HV Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = results_dir / "hv_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()