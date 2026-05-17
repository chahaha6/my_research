%% prepare_aeoss_data.m
% 将真实任务和可见窗口数据转换为 MOAEOSSP 问题模型所需的格式
% 输出文件: real_aeoss_data.mat

clear; clc;

% ------------------------------
% 1. 读取任务列表 tasklist.txt
% ------------------------------
fid = fopen('tasklist.txt', 'r');
if fid == -1
    error('无法打开 tasklist.txt');
end
lines = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
lines = lines{1};
lines = lines(~cellfun('isempty', lines));

nTask = 100;
tasks_raw = zeros(nTask, 6);
for i = 1:nTask
    parts = sscanf(lines{i}, '%f %f %f %f %f %f');
    tasks_raw(i, :) = parts';
end

% tasks 表格式: [task_id, priority, longitude, latitude, duration]
% tasklist.txt 列顺序: ID, 纬度, 经度, 优先级, 观测时长(秒), 最低质量
tasks = zeros(nTask, 5);
tasks(:,1) = tasks_raw(:,1) + 1;                     % task_id (从1开始)
tasks(:,2) = tasks_raw(:,4);                         % priority
tasks(:,3) = tasks_raw(:,3);                         % longitude (注意顺序：经度在前)
tasks(:,4) = tasks_raw(:,2);                         % latitude
tasks(:,5) = tasks_raw(:,5);                         % duration (秒)

% ------------------------------
% 2. 读取10个时间窗口文件并构建全局窗口表
% ------------------------------
num_satellites = 10;
window_id = 1;
windows = [];  % [window_id, task_id, satellite_id, orbit_id, start_time, end_time, duration]

% 基准时间：所有窗口均在同一天，取第一个窗口的日期作为基准
% 先扫描一个文件获取参考日期
sat = 1;
filename = sprintf('outputtimewindow_%d.txt', sat);
fid = fopen(filename, 'r');
content = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
file_lines = content{1};
ref_date = [];
idx = 2;
while idx <= length(file_lines) && isempty(ref_date)
    line = strtrim(file_lines{idx});
    task_id = str2double(line);
    if isnan(task_id)
        idx = idx + 1;
        continue;
    end
    idx = idx + 1;
    while idx <= length(file_lines)
        next_line = strtrim(file_lines{idx});
        if ~isempty(next_line) && all(isstrprop(next_line, 'digit') | next_line == ' ')
            break;
        end
        parts = strsplit(next_line);
        if length(parts) >= 12
            % 修正秒字段可能缺失前导零
            if startsWith(parts{6}, '.')
                parts{6} = ['0', parts{6}];
            end
            start_str = strjoin(parts(1:6), ' ');
            start_dt = datetime(start_str, 'InputFormat', 'yyyy MM dd HH mm ss.SSS');
            ref_date = dateshift(start_dt, 'start', 'day');
            break;
        end
        idx = idx + 1;
    end
    if ~isempty(ref_date), break; end
end
if isempty(ref_date)
    error('无法从文件中提取参考日期。');
end

% 正式读取所有窗口
for sat = 1:num_satellites
    filename = sprintf('outputtimewindow_%d.txt', sat);
    fid = fopen(filename, 'r');
    if fid == -1
        error('无法打开文件: %s', filename);
    end
    content = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    file_lines = content{1};
    
    idx = 2;  % 跳过第一行 'sX'
    while idx <= length(file_lines)
        line = strtrim(file_lines{idx});
        task_id_raw = str2double(line);
        if isnan(task_id_raw)
            idx = idx + 1;
            continue;
        end
        task_id = task_id_raw + 1;   % MATLAB 下标从1开始
        idx = idx + 1;
        
        while idx <= length(file_lines)
            next_line = strtrim(file_lines{idx});
            if ~isempty(next_line) && all(isstrprop(next_line, 'digit') | next_line == ' ')
                break;
            end
            parts = strsplit(next_line);
            if length(parts) >= 13   % 包含持续时长
                % 修正秒字段
                if startsWith(parts{6}, '.')
                    parts{6} = ['0', parts{6}];
                end
                if startsWith(parts{12}, '.')
                    parts{12} = ['0', parts{12}];
                end
                
                start_str = strjoin(parts(1:6), ' ');
                start_dt = datetime(start_str, 'InputFormat', 'yyyy MM dd HH mm ss.SSS');
                end_str = strjoin(parts(7:12), ' ');
                end_dt = datetime(end_str, 'InputFormat', 'yyyy MM dd HH mm ss.SSS');
                
                % 转为相对秒数
                start_sec = seconds(start_dt - ref_date);
                end_sec = seconds(end_dt - ref_date);
                duration = str2double(parts{13});   % 文件中的持续时长（秒）
                
                % 添加到窗口表
                windows(window_id, :) = [window_id, task_id, sat, 1, start_sec, end_sec, duration];
                window_id = window_id + 1;
            end
            idx = idx + 1;
        end
    end
end

nWindow = size(windows, 1);
fprintf('共读取 %d 个可见窗口\n', nWindow);

% ------------------------------
% 3. 生成卫星参数（沿用原模型随机生成，固定种子）
% ------------------------------
rng(2024);  % 固定随机种子保证可重复
nSat = num_satellites;
sats = zeros(nSat, 5);
sats(:,1) = 1:nSat;                     % 卫星ID
sats(:,2) = 10000 + 5000*rand(nSat,1);  % 能量容量
sats(:,3) = 500 + 200*rand(nSat,1);     % 存储容量 (MB)
sats(:,4) = 3 + 2*rand(nSat,1);         % 姿态转移速度 (度/秒)
sats(:,5) = 2 + 0.5*rand(nSat,1);       % 单位时间姿态能耗

% ------------------------------
% 4. 保存为 .mat 文件
% ------------------------------
save('real_aeoss_data.mat', 'tasks', 'windows', 'sats', 'nSat', 'nTask', 'nWindow');
fprintf('数据转换完成，已保存为 real_aeoss_data.mat\n');