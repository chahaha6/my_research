from pathlib import Path
from modules.problem_model import SatelliteSchedulingProblem

# test.py 所在目录 = my_research/
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "CSV_Data"

task_file = DATA_DIR / "task_info_standard.csv"
attitude_files = [DATA_DIR / f"outputattitude_{i}.csv" for i in range(1, 11)]
timewindow_files = [DATA_DIR / f"outputtimewindow_{i}.csv" for i in range(1, 11)]

problem = SatelliteSchedulingProblem(
    str(task_file),
    [str(f) for f in attitude_files],
    [str(f) for f in timewindow_files]
)

solution = [
    (0, 0, 0),   # 任务0用窗口0
    (1, 1, 0),   # 任务1用窗口1（假设窗口1存在）
    (2, 0, 1),
    (3, 0, 2)
]

objectives, feasible = problem.evaluate_solution(solution)

print("目标值：", objectives)
print("是否可行：", feasible)