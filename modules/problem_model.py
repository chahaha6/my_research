import pandas as pd
import numpy as np

class SatelliteSchedulingProblem:
    def __init__(self, task_info_file, attitude_files, timewindow_files):
        """
        初始化问题模型
        :param task_info_file: task_info_standard.csv 文件路径
        :param attitude_files: outputattitude_X_standard.csv 文件路径列表
        :param timewindow_files: outputtimewindow_X_standard.csv 文件路径列表
        """
        self.tasks = pd.read_csv(task_info_file)
        self.attitudes = [pd.read_csv(f) for f in attitude_files]
        self.timewindows = [pd.read_csv(f) for f in timewindow_files]

        # 将时间列统一转为 datetime 类型
        for df in self.attitudes:
            df['dt'] = pd.to_datetime(df['dt'])
        for df in self.timewindows:
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])

        # 预先构建合法分配集合：task_id -> [(sat_idx, window_idx), ...]
        self.valid_assignments = {}
        self.valid_set = {}
        for task_id in self.tasks['task_id']:
            opts = []
            for sat_idx, tw_df in enumerate(self.timewindows):
                matching_rows = tw_df[tw_df['task_id'] == task_id].index.tolist()
                for win_idx in matching_rows:
                    opts.append((sat_idx, win_idx))
            self.valid_assignments[task_id] = opts
            self.valid_set[task_id] = set(opts)

    # ---------- 约束检查 ----------
    def check_task_uniqueness(self, solution):
        """每个任务最多被执行一次"""
        seen = set()
        for task_id, _, _ in solution:
            if task_id in seen:
                return False
            seen.add(task_id)
        return True

    def check_window_validity(self, solution):
        """观测窗口必须在可见窗口内"""
        for task_id, win_idx, sat_idx in solution:
            if (sat_idx, win_idx) not in self.valid_set.get(task_id, set()):
                return False
        return True

    def check_satellite_capacity(self, solution):
        """未要求容量约束，默认返回 True"""
        return True

    # ---------- 目标函数 ----------
    def total_profit(self, solution):
        '''总收益最大'''
        total = 0.0
        for task_id, window_idx, sat_idx in solution:
            task_profit = self.tasks.loc[self.tasks['task_id'] == task_id, 'profit'].values[0]
            total += task_profit
        return total

    def load_balance(self, solution):
        '''负载均衡最小'''
        satellite_load = [0.0] * len(self.attitudes)
        for task_id, window_idx, sat_idx in solution:
            tw = self.timewindows[sat_idx].iloc[window_idx]
            duration = (tw['end_time'] - tw['start_time']).total_seconds()
            satellite_load[sat_idx] += duration
        if not satellite_load:
            return 0.0
        mean_load = np.mean(satellite_load)
        variance = np.mean([(l - mean_load) ** 2 for l in satellite_load])
        return variance

    def attitude_manoeuvre(self, solution):
        '''姿态机动最小'''
        total_cost = 0.0
        for sat_idx in range(len(self.attitudes)):
            sat_tasks = [s for s in solution if s[2] == sat_idx]
            if len(sat_tasks) < 2:
                continue
            sat_tasks.sort(key=lambda x: self.timewindows[sat_idx].iloc[x[1]]['start_time'])
            att_df = self.attitudes[sat_idx]
            for i in range(len(sat_tasks) - 1):
                t_curr = self.timewindows[sat_idx].iloc[sat_tasks[i][1]]['start_time']
                t_next = self.timewindows[sat_idx].iloc[sat_tasks[i+1][1]]['start_time']
                idx_curr = (att_df['dt'] - t_curr).abs().idxmin()
                idx_next = (att_df['dt'] - t_next).abs().idxmin()
                curr_angle = att_df.loc[idx_curr, ['angle_x','angle_y','angle_z']].values.astype(float)
                next_angle = att_df.loc[idx_next, ['angle_x','angle_y','angle_z']].values.astype(float)
                total_cost += np.linalg.norm(next_angle - curr_angle)
        return total_cost

    def image_quality(self, solution):
        '''图像质量最大'''
        total_quality = 0.0
        for task_id, window_idx, sat_idx in solution:
            tw = self.timewindows[sat_idx].iloc[window_idx]
            bt_s = tw['start_time']
            et_s = tw['end_time']
            wb_t_s = bt_s + (et_s - bt_s) / 2.0
            ut = wb_t_s
            denominator = wb_t_s - bt_s
            if denominator != pd.Timedelta(0):
                quality = 10.0 - 9.0 * abs((ut - wb_t_s).total_seconds()) / denominator.total_seconds()
            else:
                quality = 10.0 if abs((ut - wb_t_s).total_seconds()) < 1e-9 else 0.0
            quality = max(0.0, quality)
            total_quality += quality
        return total_quality

    # ---------- 综合评价 ----------
    def evaluate_solution(self, solution):
        feasible = (self.check_task_uniqueness(solution) and
                    self.check_window_validity(solution) and
                    self.check_satellite_capacity(solution))
        objectives = {
            'total_profit': self.total_profit(solution),
            'load_balance': self.load_balance(solution),
            'attitude_manoeuvre': self.attitude_manoeuvre(solution),
            'image_quality': self.image_quality(solution)
        }
        return objectives, feasible