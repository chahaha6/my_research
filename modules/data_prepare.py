import pandas as pd
from datetime import datetime
import os


#读取原 tasklist 文件
task_file = "Loacal_Data/tasklist.txt"
cols = ['task_id', 'latitude', 'longitude', 'profit', 'duration', 'min_quality']
task_data = pd.read_csv(task_file, delim_whitespace=True, names=cols)

# 保存为标准 CSV
task_data.to_csv("CSV_Data/task_info_standard.csv", index=False)
print("任务信息已处理为标准 CSV，保留最低质量列: task_info_standard.csv")


attitude_input_folder = "Loacal_Data/"
output_folder = "CSV_Data"

os.makedirs(output_folder, exist_ok=True)

for i in range(1, 11):

    input_file = os.path.join(attitude_input_folder, f"outputattitude_{i}.txt")
    output_file = os.path.join(output_folder, f"outputattitude_{i}.csv")

    if not os.path.exists(input_file):
        print(f"{input_file} 不存在，跳过")
        continue

    cols = [
        'year','month','day',
        'hour','minute','second',
        'angle_x','angle_y','angle_z',
        'val1','val2','val3','val4','val5'
    ]

    # 读取TXT
    df = pd.read_csv(
        input_file,
        sep=r'\s+',
        names=cols
    )

    # 只保留必要列
    df = df[[
        'year','month','day',
        'hour','minute','second',
        'angle_x','angle_y','angle_z'
    ]]

    # datetime列
    df['dt'] = df.apply(
        lambda r: datetime(
            int(r.year),
            int(r.month),
            int(r.day),
            int(r.hour),
            int(r.minute),
            int(r.second)
        ),
        axis=1
    )

    # 保存CSV
    df.to_csv(output_file, index=False)


timewindow_input_folder = "Loacal_Data/"
time_window_output_folder = "CSV_Data"

os.makedirs(time_window_output_folder, exist_ok=True)

for i in range(1, 11):
    input_file = os.path.join(timewindow_input_folder, f"outputtimewindow_{i}.txt")
    output_file = os.path.join(time_window_output_folder, f"outputtimewindow_{i}.csv")

    if not os.path.exists(input_file):
        print(f"{input_file} 不存在，跳过")
        continue

    window_list = []
    current_task = None

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 如果整行是任务编号
            if line.isdigit():
                current_task = int(line)
                continue

            # 窗口行
            parts = line.split()
            if len(parts) >= 13:
                try:
                    # 将秒列 float 转 int，直接构造 datetime
                    start = datetime(
                        int(parts[0]), int(parts[1]), int(parts[2]),
                        int(parts[3]), int(parts[4]), int(float(parts[5]))
                    )
                    end = datetime(
                        int(parts[6]), int(parts[7]), int(parts[8]),
                        int(parts[9]), int(parts[10]), int(float(parts[11]))
                    )
                    window_length = float(parts[12])
                    window_list.append([current_task, start, end, window_length])
                except Exception as e:
                    print(f"警告：行解析失败 -> {line}, 错误: {e}")
                    continue
            else:
                # 任务编号存在但没有窗口行，直接跳过
                continue

    # 生成 DataFrame
    if window_list:
        df = pd.DataFrame(window_list, columns=['task_id', 'start_time', 'end_time', 'window_length'])
        df.to_csv(output_file, index=False)
        print(f"{input_file} -> {output_file} （只包含有窗口的任务）")
    else:
        print(f"{input_file} 没有任何有效窗口，未生成 CSV")