import os
import pandas as pd
from datetime import datetime


# =========================================================
# 数据规模配置
# 后续你只需要改 NUM_SATS / NUM_TASKS 即可。
# =========================================================

NUM_SATS = 5
NUM_TASKS = 1000

CASE_TAG = f"s{NUM_SATS}_t{NUM_TASKS}"
LOCAL_DATA_ROOT = "Local_Data"
LOCAL_DATA_DIR = os.path.join(LOCAL_DATA_ROOT, f"area_{CASE_TAG}")
CSV_DATA_ROOT = "CSV_Data"
CSV_DATA_DIR = os.path.join(CSV_DATA_ROOT, CASE_TAG)

# 是否删除 CSV_Data 中旧的 outputattitude_*.csv / outputtimewindow_*.csv，
# 建议 True，避免不同卫星数量的数据混在一起。
CLEAN_OLD_SAT_CSV = True

if not os.path.isdir(LOCAL_DATA_DIR):
    raise FileNotFoundError(
        f"Local dataset directory not found: {LOCAL_DATA_DIR}\n"
        f"Expected a folder like Local_Data/area_s{NUM_SATS}_t{NUM_TASKS}."
    )

os.makedirs(CSV_DATA_DIR, exist_ok=True)

print(f"Using local dataset: {LOCAL_DATA_DIR}")
print(f"Writing CSV dataset: {CSV_DATA_DIR}")


# =========================================================
# 工具函数
# =========================================================

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

    # 处理 round 后 micro 进位的极端情况
    if micro >= 1_000_000:
        sec += 1
        micro -= 1_000_000

    return datetime(year, month, day, hour, minute, sec, micro)


def clean_old_satellite_csv():
    if not CLEAN_OLD_SAT_CSV:
        return

    for name in os.listdir(CSV_DATA_DIR):
        if (
            (name.startswith("outputattitude_") or name.startswith("outputtimewindow_"))
            and name.endswith(".csv")
        ):
            try:
                os.remove(os.path.join(CSV_DATA_DIR, name))
            except OSError:
                pass


clean_old_satellite_csv()


# =========================================================
# 1. 处理 tasklist.txt
# =========================================================

task_file = os.path.join(LOCAL_DATA_DIR, "tasklist.txt")

# tasklist.txt 中正常任务行应为 6 列：
# task_id latitude longitude profit duration extra_attr
# 数据中会隔行出现类似 “2 45.0 45.0 0.0” 的 4 列辅助/异常行，必须跳过，
# 否则 duration、extra_attr 会变成 NaN，后续解码和评价会异常。
task_rows = []
skipped_short_rows = 0
skipped_bad_rows = 0

with open(task_file, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # 只接受完整任务行，过滤隔行出现的 4 列行
        if len(parts) != 6:
            skipped_short_rows += 1
            continue

        try:
            task_id = int(parts[0])
            latitude = float(parts[1])
            longitude = float(parts[2])
            profit = float(parts[3])
            duration = float(parts[4])
            extra_attr = float(parts[5])

            # duration 必须为正；profit 为 0 的任务也可以保留，
            # 但如果你后续想过滤低价值任务，可在这里加 profit > 0 条件。
            if duration <= 0:
                skipped_bad_rows += 1
                continue

            task_rows.append([
                task_id,
                latitude,
                longitude,
                profit,
                duration,
                extra_attr
            ])

        except Exception as e:
            skipped_bad_rows += 1
            print(f"任务行解析失败：第{line_no}行：{line}，错误：{e}")

if not task_rows:
    raise ValueError(
        "tasklist.txt 中没有解析到有效任务行。请检查任务文件是否为 6 列格式。"
    )

task_data = pd.DataFrame(
    task_rows,
    columns=[
        "task_id",
        "latitude",
        "longitude",
        "profit",
        "duration",
        "extra_attr"
    ]
)

# 去掉重复 task_id，避免异常重复任务干扰编码。
before_dedup = len(task_data)
task_data = task_data.drop_duplicates(subset=["task_id"], keep="first")
dedup_count = before_dedup - len(task_data)

# 先过滤异常行，再取前 NUM_TASKS 个有效任务。
if NUM_TASKS is not None and NUM_TASKS > 0:
    task_data = task_data.head(NUM_TASKS).copy()

selected_task_ids = set(task_data["task_id"].astype(int).tolist())

task_output = os.path.join(CSV_DATA_DIR, "task_info_standard.csv")
task_data.to_csv(task_output, index=False)

print(f"任务信息已保存：{task_output}")
print(f"当前任务数量：{len(task_data)}")
print(f"当前卫星数量：{NUM_SATS}")
print(f"已跳过非 6 列任务行：{skipped_short_rows}")
print(f"已跳过异常任务行：{skipped_bad_rows}")
print(f"已删除重复 task_id：{dedup_count}")
print("任务表空值统计：")
print(task_data.isna().sum())


# =========================================================
# 2. 处理 outputattitude_i.txt
# =========================================================

for i in range(1, NUM_SATS + 1):
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
    print(f"{input_file} -> {output_file}，行数={len(df)}")


# =========================================================
# 3. 处理 outputtimewindow_i.txt
# =========================================================

for i in range(1, NUM_SATS + 1):
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

            # 只保留选定任务数量范围内的任务窗口
            if current_task not in selected_task_ids:
                continue

            parts = line.split()

            if len(parts) >= 13 and current_task is not None:
                try:
                    start_time = parse_datetime(parts, 0)
                    end_time = parse_datetime(parts, 6)
                    window_length = float(parts[12])

                    # VTW 中点，也就是 image_quality 里可作为最佳成像时刻的 best_time
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
        print(f"{input_file} -> {output_file}，窗口数={len(df)}")

    else:
        print(f"{input_file} 没有有效窗口")

print("\n数据预处理完成。")
