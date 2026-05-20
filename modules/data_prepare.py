import os
import pandas as pd
from datetime import datetime


LOCAL_DATA_DIR = "Loacal_Data"
CSV_DATA_DIR = "CSV_Data"

os.makedirs(CSV_DATA_DIR, exist_ok=True)


def parse_datetime(parts, offset):
    """
    从 parts[offset:offset+6] 解析时间：
    year month day hour minute second
    支持小数秒。
    """
    year = int(parts[offset])
    month = int(parts[offset + 1])
    day = int(parts[offset + 2])
    hour = int(parts[offset + 3])
    minute = int(parts[offset + 4])

    sec_float = float(parts[offset + 5])
    sec = int(sec_float)
    micro = int(round((sec_float - sec) * 1_000_000))

    return datetime(year, month, day, hour, minute, sec, micro)


# =========================================================
# 1. 处理 tasklist.txt
# =========================================================

task_file = os.path.join(LOCAL_DATA_DIR, "tasklist.txt")


task_cols = [
    "task_id",
    "latitude",
    "longitude",
    "profit",
    "duration",
    "extra_attr"
]

task_data = pd.read_csv(
    task_file,
    sep=r"\s+",
    names=task_cols
)

task_data["task_id"] = task_data["task_id"].astype(int)
task_data["profit"] = task_data["profit"].astype(float)
task_data["duration"] = task_data["duration"].astype(float)

task_output = os.path.join(CSV_DATA_DIR, "task_info_standard.csv")
task_data.to_csv(task_output, index=False)

print(f"任务信息已保存：{task_output}")


# =========================================================
# 2. 处理 outputattitude_i.txt
# =========================================================

for i in range(1, 11):
    input_file = os.path.join(LOCAL_DATA_DIR, f"outputattitude_{i}.txt")
    output_file = os.path.join(CSV_DATA_DIR, f"outputattitude_{i}.csv")

    if not os.path.exists(input_file):
        print(f"{input_file} 不存在，跳过")
        continue

    rows = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()

            if len(parts) < 9:
                continue

            try:
                dt = parse_datetime(parts, 0)

                # 这里沿用你原来的角度列
                angle_x = float(parts[6])
                angle_y = float(parts[7])
                angle_z = float(parts[8])

                rows.append([dt, angle_x, angle_y, angle_z])

            except Exception as e:
                print(f"姿态行解析失败：{line.strip()}，错误：{e}")

    df = pd.DataFrame(
        rows,
        columns=["dt", "angle_x", "angle_y", "angle_z"]
    )

    df.to_csv(output_file, index=False)
    print(f"{input_file} -> {output_file}")


# =========================================================
# 3. 处理 outputtimewindow_i.txt
# =========================================================

for i in range(1, 11):
    input_file = os.path.join(LOCAL_DATA_DIR, f"outputtimewindow_{i}.txt")
    output_file = os.path.join(CSV_DATA_DIR, f"outputtimewindow_{i}.csv")

    if not os.path.exists(input_file):
        print(f"{input_file} 不存在，跳过")
        continue

    window_list = []
    current_task = None

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # 跳过 s1 / s2 这种卫星标识
            if line.startswith("s"):
                continue

            # 任务编号行
            if line.isdigit():
                current_task = int(line)
                continue

            parts = line.split()

            if len(parts) >= 13 and current_task is not None:
                try:
                    start_time = parse_datetime(parts, 0)
                    end_time = parse_datetime(parts, 6)
                    window_length = float(parts[12])

                    # VTW 中点，也就是示例代码里的 middle
                    best_time = start_time + (end_time - start_time) / 2.0

                    window_list.append([
                        current_task,
                        start_time,
                        end_time,
                        window_length,
                        best_time
                    ])

                except Exception as e:
                    print(f"窗口行解析失败：{line}，错误：{e}")

    if window_list:
        df = pd.DataFrame(
            window_list,
            columns=[
                "task_id",
                "start_time",
                "end_time",
                "window_length",
                "best_time"
            ]
        )

        df.to_csv(output_file, index=False)
        print(f"{input_file} -> {output_file}")

    else:
        print(f"{input_file} 没有有效窗口")