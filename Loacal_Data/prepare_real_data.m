%% prepare_real_data.m
% 将 tasklist.txt 和 outputtimewindow_X.txt 转换为 MO_SRSP 可用的数据格式
% 输出文件：real_srsp_data.mat

clear; clc;

% ------------------------------
% 参数设置（与 MO_SRSP 一致）
% ------------------------------
num_tasks = 100;
num_satellites = 10;       % 卫星数量
num_antennas = 10;         % 天线数量（这里与卫星一一对应）
time_horizon = 24;         % 时间范围（小时）

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

tasks_raw = zeros(num_tasks, 6);
for i = 1:num_tasks
    parts = sscanf(lines{i}, '%f %f %f %f %f %f');
    tasks_raw(i, :) = parts';
end

% ------------------------------
% 2. 初始化任务数据结构体
% ------------------------------
T = struct();
for i = 1:num_tasks
    T(i).si = [];
    T(i).pi = tasks_raw(i, 4);
    T(i).di = tasks_raw(i, 5) / 3600;   % 秒转小时
    T(i).ei = rand() * 0.5;
    T(i).uesi = 24;
    T(i).ulei = 0;
    T(i).available_windows = [];
end

% ------------------------------
% 3. 读取10个时间窗口文件
% ------------------------------
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
        task_id = str2double(line);
        if isnan(task_id)
            idx = idx + 1;
            continue;
        end
        task_idx = task_id + 1;
        idx = idx + 1;
        
        while idx <= length(file_lines)
            next_line = strtrim(file_lines{idx});
            if ~isempty(next_line) && all(isstrprop(next_line, 'digit') | next_line == ' ')
                break;
            end
            parts = strsplit(next_line);
            if length(parts) >= 12
                % 修正秒字段缺失前导零的问题
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
                
                ref_date = dateshift(start_dt, 'start', 'day');
                start_hour = hours(start_dt - ref_date);
                end_hour = hours(end_dt - ref_date);
                
                win.start = start_hour;
                win.finish = end_hour;
                win.antenna = sat;
                
                if isempty(T(task_idx).available_windows)
                    T(task_idx).available_windows = win;
                else
                    T(task_idx).available_windows(end+1) = win;
                end
                
                T(task_idx).uesi = min(T(task_idx).uesi, start_hour);
                T(task_idx).ulei = max(T(task_idx).ulei, end_hour);
            end
            idx = idx + 1;
        end
    end
end

% ------------------------------
% 4. 为每个任务分配默认卫星
% ------------------------------
for i = 1:num_tasks
    if ~isempty(T(i).available_windows)
        T(i).si = T(i).available_windows(1).antenna;
    else
        T(i).si = randi(num_satellites);
        T(i).uesi = 0;
        T(i).ulei = 24;
    end
    if isempty(T(i).available_windows)
        win.qosi = 0;
        win.qoei = 24;
        win.antenna = T(i).si;
        T(i).available_windows = win;
    end
end

% ------------------------------
% 5. 生成天线调整时间（注意拼写为 antena 而非 antenna）
% ------------------------------
antena_adjustment = rand(num_antennas, 1) * 0.5;

% ------------------------------
% 6. 组装完整数据结构体
% ------------------------------
data = struct();
data.num_satellites = num_satellites;
data.num_antennas = num_antennas;
data.num_tasks = num_tasks;
data.time_horizon = time_horizon;
data.antenna_adjustment = antena_adjustment;   % 字段名必须与 MO_SRSP 中一致
data.T = T;

save('real_srsp_data.mat', 'data');
fprintf('数据转换完成，已保存为 real_srsp_data.mat\n');