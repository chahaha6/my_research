import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from modules.problem_model import SatelliteSchedulingProblem
from modules.mode_sdas import MODE_SDAS
from modules.mode_sdas_pic import MODE_SDAS_PIC
from modules.nsga2 import NSGA2

# ================= 通用参数 =================
POP_SIZE = 100
GENERATIONS = 50          # 每轮代数
NUM_RUNS = 3              # 运行轮数

# ================= 算法专属参数 =================
# MODE‑SDAS / MODE‑SDAS‑PIC 参数
N_SUB = 10
L = 3
RMEM = 5
CR = 0.5
F = 0.5
ALPHA = 0.1
SIGMA = 0.01
GAMMA = 0.7
ELITE_PROB = 0.6
ARCHIVE_SIZE = 50

# NSGA‑II 参数
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.1


def main():
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "CSV_Data"

    # 数据文件路径
    task_file = DATA_DIR / "task_info_standard.csv"
    attitude_files = [DATA_DIR / f"outputattitude_{i}.csv" for i in range(1, 11)]
    timewindow_files = [DATA_DIR / f"outputtimewindow_{i}.csv" for i in range(1, 11)]

    problem = SatelliteSchedulingProblem(
        str(task_file),
        [str(f) for f in attitude_files],
        [str(f) for f in timewindow_files]
    )

    ref_front_file = BASE_DIR / "results" / "reference_front.csv"
    if not ref_front_file.exists():
        ref_front_file = None

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # 存储多轮结果
    hv_histories = []
    best_profits, best_loads, best_attitudes, best_qualities = [], [], [], []

    # =================== MODE‑SDAS‑PIC ===================
    # 取消注释即可运行 MODE‑SDAS‑PIC
    
    # print("运行算法：MODE‑SDAS‑PIC")
    # for run_idx in range(1, NUM_RUNS + 1):
    #     print(f"\n========== MODE‑SDAS‑PIC Run {run_idx}/{NUM_RUNS} ==========")

    #     alg = MODE_SDAS_PIC(
    #         problem,
    #         pop_size=POP_SIZE,
    #         generations=GENERATIONS,
    #         Nsub=N_SUB, L=L, Rmem=RMEM, CR=CR, F=F,
    #         alpha=ALPHA, sigma=SIGMA, gamma=GAMMA,
    #         elite_prob=ELITE_PROB, archive_size=ARCHIVE_SIZE,
    #         ref_front_file=ref_front_file
    #     )

    #     final_dec, final_raw, final_obj, hv_history = alg.run()
    #     hv_histories.append(np.array(hv_history))

    #     profits = [raw['total_profit'] for raw in final_raw]
    #     loads = [raw['load_balance'] for raw in final_raw]
    #     attitudes = [raw['attitude_manoeuvre'] for raw in final_raw]
    #     qualities = [raw['image_quality'] for raw in final_raw]

    #     best_profits.append(max(profits))
    #     best_loads.append(min(loads))
    #     best_attitudes.append(min(attitudes))
    #     best_qualities.append(max(qualities))

    #     print(f"Run {run_idx} best: profit={best_profits[-1]:.1f}, "
    #           f"load={best_loads[-1]:.2f}, att={best_attitudes[-1]:.2f}, "
    #           f"quality={best_qualities[-1]:.2f}")

    # algo_tag = "mode_sdas_pic"
    

    # =================== MODE‑SDAS ===================
    # 取消注释即可运行 MODE‑SDAS
    
    # print("运行算法：MODE‑SDAS")
    # for run_idx in range(1, NUM_RUNS + 1):
    #     print(f"\n========== MODE‑SDAS Run {run_idx}/{NUM_RUNS} ==========")

    #     alg = MODE_SDAS(
    #         problem,
    #         pop_size=POP_SIZE,
    #         generations=GENERATIONS,
    #         Nsub=N_SUB,
    #         L=L,
    #         CR=CR,
    #         F=F,
    #         ref_front_file=ref_front_file
    #     )

    #     final_dec, final_raw, final_obj, hv_history = alg.run()
    #     hv_histories.append(np.array(hv_history))

    #     profits = [raw['total_profit'] for raw in final_raw]
    #     loads = [raw['load_balance'] for raw in final_raw]
    #     attitudes = [raw['attitude_manoeuvre'] for raw in final_raw]
    #     qualities = [raw['image_quality'] for raw in final_raw]

    #     best_profits.append(max(profits))
    #     best_loads.append(min(loads))
    #     best_attitudes.append(min(attitudes))
    #     best_qualities.append(max(qualities))

    #     print(f"Run {run_idx} best: profit={best_profits[-1]:.1f}, "
    #           f"load={best_loads[-1]:.2f}, att={best_attitudes[-1]:.2f}, "
    #           f"quality={best_qualities[-1]:.2f}")

    # algo_tag = "mode_sdas"
    

    # =================== NSGA‑II ===================
    # 取消注释即可运行 NSGA‑II
    
    print("运行算法：NSGA‑II")
    for run_idx in range(1, NUM_RUNS + 1):
        print(f"\n========== NSGA‑II Run {run_idx}/{NUM_RUNS} ==========")

        alg = NSGA2(
            problem,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            crossover_prob=CROSSOVER_PROB,
            mutation_prob=MUTATION_PROB,
            ref_front_file=ref_front_file
        )

        final_dec, final_raw, final_obj, hv_history = alg.run()
        hv_histories.append(np.array(hv_history))

        profits = [raw['total_profit'] for raw in final_raw]
        loads = [raw['load_balance'] for raw in final_raw]
        attitudes = [raw['attitude_manoeuvre'] for raw in final_raw]
        qualities = [raw['image_quality'] for raw in final_raw]

        best_profits.append(max(profits))
        best_loads.append(min(loads))
        best_attitudes.append(min(attitudes))
        best_qualities.append(max(qualities))

        print(f"Run {run_idx} best: profit={best_profits[-1]:.1f}, "
              f"load={best_loads[-1]:.2f}, att={best_attitudes[-1]:.2f}, "
              f"quality={best_qualities[-1]:.2f}")

    algo_tag = "nsga2"
    

    # =================== 保存和绘图 ===================
    if hv_histories:
        hv_array = np.array(hv_histories)
        mean_hv = np.mean(hv_array, axis=0)

        avg_profit = np.mean(best_profits)
        avg_load = np.mean(best_loads)
        avg_att = np.mean(best_attitudes)
        avg_quality = np.mean(best_qualities)

        # 保存平均 HV 曲线数据
        hv_avg_file = results_dir / f"hv_avg_3runs_{algo_tag}.csv"
        np.savetxt(hv_avg_file, mean_hv, delimiter=',')
        print(f"\nAverage HV history saved to {hv_avg_file}")

        # 保存每轮最优目标值及平均值
        summary_dict = {
            'Run': list(range(1, NUM_RUNS + 1)) + ['Average'],
            'Profit_max': best_profits + [avg_profit],
            'Load_min': best_loads + [avg_load],
            'Attitude_min': best_attitudes + [avg_att],
            'Quality_max': best_qualities + [avg_quality]
        }
        df_summary = pd.DataFrame(summary_dict)
        summary_file = results_dir / f"summary_3runs_{algo_tag}.csv"
        df_summary.to_csv(summary_file, index=False)
        print(f"Summary of best objectives saved to {summary_file}")

        # 绘制平均 HV 收敛曲线
        plt.figure(figsize=(8, 5))
        gens = range(1, len(mean_hv) + 1)
        label = algo_tag.replace("_", " ").upper()
        plt.plot(gens, mean_hv, 's-', label=f'Average HV ({label}, {NUM_RUNS} runs)', markersize=4)
        plt.xlabel('Generation')
        plt.ylabel('Hypervolume')
        plt.title(f'Average HV Convergence ({label})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = results_dir / f"hv_avg_convergence_{algo_tag}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Average convergence plot saved to {plot_path}")

    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()