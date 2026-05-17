import numpy as np
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from modules.problem_model import SatelliteSchedulingProblem
from modules.mode_sdas_pic import MODE_SDAS_PIC
from modules.nsga2 import NSGA2
from modules.utils_moea import non_dominated_sort

# ----------------------------------------------------------------------
# 统一参数（可在此处修改，或通过函数参数传入）
# ----------------------------------------------------------------------
POP_SIZE = 100
GENERATIONS = 30

# ----------------------------------------------------------------------
# MODE‑SDAS‑PIC 完整运行函数
# ----------------------------------------------------------------------
def run_mode_sdas_pic(problem, pop_size=POP_SIZE, generations=GENERATIONS,
                      Nsub=10, L=3, Rmem=5, CR=0.5, F=0.5,
                      alpha=0.15, sigma=0.01, gamma=0.7,
                      ref_front_file=None, results_dir=None):
    """
    运行 MODE‑SDAS‑PIC 算法，保存 HV 历史、最优解、收敛曲线，可选计算 IGD。
    参数：
        problem: SatelliteSchedulingProblem 实例
        pop_size, generations: 种群大小与进化代数
        Nsub, L, Rmem, CR, F: 算法特有参数
        alpha, sigma, gamma: PICEO 相关参数
        ref_front_file: IGD 参考前沿文件路径 (Path 或 None)
        results_dir: 结果保存目录 (Path 或 None，默认为 my_research/results)
    返回：
        hv_history: HV 历史列表
        final_raw: 最终种群原始目标值列表
        final_dec: 最终种群决策变量数组
        final_obj: 最终种群最小化目标字典列表
    """
    if results_dir is None:
        BASE_DIR = Path(__file__).parent
        results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # 初始化算法
    alg = MODE_SDAS_PIC(problem,
                        pop_size=pop_size,
                        generations=generations,
                        Nsub=Nsub, L=L, Rmem=Rmem, CR=CR, F=F,
                        alpha=alpha, sigma=sigma, gamma=gamma,
                        ref_front_file=ref_front_file)

    print("\n========== Running MODE‑SDAS‑PIC ==========")
    final_dec, final_raw, final_obj, hv_history = alg.run()

    # 保存 HV 历史
    hv_file = results_dir / "hv_history_mode_sdas_pic.csv"
    np.savetxt(hv_file, np.array(hv_history), delimiter=',')
    print(f"[MODE‑SDAS‑PIC] HV history saved to {hv_file}")

    # 保存最优解（按收益最大）
    best_idx = np.argmax([raw['total_profit'] for raw in final_raw])
    best_dec = final_dec[best_idx]
    best_sol = alg.decode(best_dec)
    df = pd.DataFrame(best_sol, columns=["task_id", "window_idx", "sat_idx"])
    sol_file = results_dir / "final_solution_mode_sdas_with_one_change_pic.csv"
    df.to_csv(sol_file, index=False)
    print(f"[MODE‑SDAS‑PIC] Final solution saved to {sol_file}")

    # 若存在参考前沿，计算 IGD
    if alg.igd_indicator is not None:
        min_objs = [alg.evaluate(dec)[0] for dec in final_dec]
        fronts = non_dominated_sort(min_objs)
        nd_F = alg.get_obj_matrix([min_objs[i] for i in fronts[0]])
        igd_val = alg.igd_indicator(nd_F)
        print(f"[MODE‑SDAS‑PIC] IGD = {igd_val:.4f}")
        with open(results_dir / "igd_mode_sdas_with_one_change_pic.txt", 'w') as f:
            f.write(f"{igd_val}\n")

    # 绘制 HV 收敛曲线
    plt.figure(figsize=(8, 5))
    gens = range(1, len(hv_history) + 1)
    plt.plot(gens, hv_history, 's-', label='MODE‑SDAS‑PIC', markersize=4)
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.title('HV Convergence (MODE‑SDAS‑PIC)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = results_dir / "hv_convergence_mode_sdas_pic.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()                     # 弹出图形窗口
    print(f"[MODE‑SDAS‑PIC] Convergence plot saved to {plot_path}")

    return hv_history, final_raw, final_dec, final_obj


# ----------------------------------------------------------------------
# NSGA‑II 完整运行函数
# ----------------------------------------------------------------------
def run_nsga2(problem, pop_size=POP_SIZE, generations=GENERATIONS,
              crossover_prob=0.9, mutation_prob=0.1,
              ref_front_file=None, results_dir=None):
    """
    运行 NSGA‑II 算法，保存 HV 历史、最优解、收敛曲线，可选计算 IGD。
    参数：
        problem: SatelliteSchedulingProblem 实例
        pop_size, generations: 种群大小与进化代数
        crossover_prob, mutation_prob: 交叉和变异概率
        ref_front_file: IGD 参考前沿文件路径 (Path 或 None)
        results_dir: 结果保存目录 (Path 或 None，默认为 my_research/results)
    返回：
        hv_history, final_raw, final_dec, final_obj
    """
    if results_dir is None:
        BASE_DIR = Path(__file__).parent
        results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # 初始化算法
    alg = NSGA2(problem,
                pop_size=pop_size,
                generations=generations,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
                ref_front_file=ref_front_file)

    print("\n========== Running NSGA‑II ==========")
    final_dec, final_raw, final_obj, hv_history = alg.run()

    # 保存 HV 历史
    hv_file = results_dir / "hv_history_nsga2.csv"
    np.savetxt(hv_file, np.array(hv_history), delimiter=',')
    print(f"[NSGA‑II] HV history saved to {hv_file}")

    # 保存最优解（按收益最大）
    best_idx = np.argmax([raw['total_profit'] for raw in final_raw])
    best_dec = final_dec[best_idx]
    best_sol = alg.decode(best_dec)
    df = pd.DataFrame(best_sol, columns=["task_id", "window_idx", "sat_idx"])
    sol_file = results_dir / "final_solution_nsga2.csv"
    df.to_csv(sol_file, index=False)
    print(f"[NSGA‑II] Final solution saved to {sol_file}")

    # 若存在参考前沿，计算 IGD
    if alg.igd_indicator is not None:
        min_objs = [alg.evaluate(dec)[0] for dec in final_dec]
        fronts = non_dominated_sort(min_objs)
        nd_F = alg.get_obj_matrix([min_objs[i] for i in fronts[0]])
        igd_val = alg.igd_indicator(nd_F)
        print(f"[NSGA‑II] IGD = {igd_val:.4f}")
        with open(results_dir / "igd_nsga2.txt", 'w') as f:
            f.write(f"{igd_val}\n")

    # 绘制 HV 收敛曲线
    plt.figure(figsize=(8, 5))
    gens = range(1, len(hv_history) + 1)
    plt.plot(gens, hv_history, 'o-', label='NSGA‑II', markersize=4)
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.title('HV Convergence (NSGA‑II)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = results_dir / "hv_convergence_nsga2.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()                     # 弹出图形窗口
    print(f"[NSGA‑II] Convergence plot saved to {plot_path}")

    return hv_history, final_raw, final_dec, final_obj


# ----------------------------------------------------------------------
# 主程序入口
# ----------------------------------------------------------------------
def main():
    # 项目根目录与数据目录
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "CSV_Data"

    # 数据文件路径
    task_file = DATA_DIR / "task_info_standard.csv"
    attitude_files = [DATA_DIR / f"outputattitude_{i}.csv" for i in range(1, 11)]
    timewindow_files = [DATA_DIR / f"outputtimewindow_{i}.csv" for i in range(1, 11)]

    # 构建问题实例（数据由外部 CSV 提供）
    problem = SatelliteSchedulingProblem(str(task_file),
                                         [str(f) for f in attitude_files],
                                         [str(f) for f in timewindow_files])

    # 可选 IGD 参考前沿文件
    ref_front_file = BASE_DIR / "results" / "reference_front.csv"
    if not ref_front_file.exists():
        ref_front_file = None

    # 可根据需要只运行其中一个，或两个都运行
    # 运行 MODE‑SDAS‑PIC
    hv_hist_m, final_raw_m, final_dec_m, final_obj_m = run_mode_sdas_pic(
        problem,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        Nsub=10, L=3, Rmem=5, CR=0.5, F=0.5,
        alpha=0.15, sigma=0.01, gamma=0.7,
        ref_front_file=ref_front_file
    )

    # 运行 NSGA‑II（如果您希望运行，取消注释即可）
    # hv_hist_n, final_raw_n, final_dec_n, final_obj_n = run_nsga2(
    #     problem,
    #     pop_size=POP_SIZE,
    #     generations=GENERATIONS,
    #     crossover_prob=0.9,
    #     mutation_prob=0.1,
    #     ref_front_file=ref_front_file
    # )

    # 如需对比两个算法的 HV 曲线，可以在此处额外绘制合并图
    # （省略，您可根据需要自行添加）

    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()