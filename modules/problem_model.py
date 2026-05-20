import pandas as pd
import numpy as np


class SatelliteSchedulingProblem:
    def __init__(self, task_info_file, attitude_files, timewindow_files):
        self.tasks = pd.read_csv(task_info_file)
        self.attitudes = [pd.read_csv(f) for f in attitude_files]
        self.timewindows = [pd.read_csv(f) for f in timewindow_files]

        # ---------- task ----------
        self.tasks["task_id"] = self.tasks["task_id"].astype(int)
        self.tasks["profit"] = self.tasks["profit"].astype(float)
        self.tasks["duration"] = self.tasks["duration"].astype(float)

        # ---------- attitude ----------
        for df in self.attitudes:
            df["dt"] = pd.to_datetime(df["dt"])
            df["angle_x"] = df["angle_x"].astype(float)
            df["angle_y"] = df["angle_y"].astype(float)
            df["angle_z"] = df["angle_z"].astype(float)

        # ---------- time windows ----------
        for df in self.timewindows:
            df["task_id"] = df["task_id"].astype(int)
            df["start_time"] = pd.to_datetime(df["start_time"])
            df["end_time"] = pd.to_datetime(df["end_time"])

            if "best_time" in df.columns:
                df["best_time"] = pd.to_datetime(df["best_time"])
            else:
                df["best_time"] = df["start_time"] + (
                    df["end_time"] - df["start_time"]
                ) / 2.0

            if "window_length" not in df.columns:
                df["window_length"] = (
                    df["end_time"] - df["start_time"]
                ).dt.total_seconds()

        # ---------- task info dict ----------
        self.task_info = {}

        for _, row in self.tasks.iterrows():
            task_id = int(row["task_id"])

            self.task_info[task_id] = {
                "profit": float(row["profit"]),
                "duration": float(row["duration"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "extra_attr": float(row["extra_attr"]) if "extra_attr" in row else 0.0,
            }

        # ---------- valid assignment ----------
        self.valid_assignments = {}
        self.valid_set = {}

        for task_id in self.tasks["task_id"]:
            task_id = int(task_id)
            opts = []

            for sat_idx, tw_df in enumerate(self.timewindows):
                rows = tw_df[tw_df["task_id"] == task_id].index.tolist()

                for win_idx in rows:
                    opts.append((sat_idx, win_idx))

            self.valid_assignments[task_id] = opts
            self.valid_set[task_id] = set(opts)

    # =========================================================
    # basic getters
    # =========================================================

    def get_profit(self, task_id):
        return self.task_info[int(task_id)]["profit"]

    def get_duration(self, task_id):
        return self.task_info[int(task_id)]["duration"]

    def get_window(self, sat_idx, win_idx):
        return self.timewindows[int(sat_idx)].iloc[int(win_idx)]

    # =========================================================
    # image quality objective
    # =========================================================

    def single_image_quality(self, task_id, win_idx, sat_idx, actual_start):
        """
        图像质量目标：
        实际观测中心越接近 VTW 中点 best_time，质量越高。

        注意：
        这里只是目标函数，不是约束。
        没有 quality >= min_quality。
        """

        tw = self.get_window(sat_idx, win_idx)

        bt = tw["start_time"]
        best_time = tw["best_time"]

        duration = self.get_duration(task_id)
        actual_center = actual_start + pd.to_timedelta(duration / 2.0, unit="s")

        denom = (best_time - bt).total_seconds()

        if denom <= 0:
            return 0.0

        quality = 10.0 - 9.0 * abs(
            (actual_center - best_time).total_seconds()
        ) / denom

        return max(0.0, quality)

    def image_quality(self, solution):
        total_quality = 0.0

        for task_id, win_idx, sat_idx, actual_start, actual_end in solution:
            total_quality += self.single_image_quality(
                task_id,
                win_idx,
                sat_idx,
                actual_start
            )

        return total_quality

    # =========================================================
    # constraints
    # =========================================================

    def check_task_uniqueness(self, solution):
        seen = set()

        for item in solution:
            task_id = int(item[0])

            if task_id in seen:
                return False

            seen.add(task_id)

        return True

    def check_window_validity(self, solution):
        for task_id, win_idx, sat_idx, actual_start, actual_end in solution:
            task_id = int(task_id)
            win_idx = int(win_idx)
            sat_idx = int(sat_idx)

            if (sat_idx, win_idx) not in self.valid_set.get(task_id, set()):
                return False

        return True

    def check_observation_inside_window(self, solution):
        """
        OTW 必须位于 VTW 内。
        """

        for task_id, win_idx, sat_idx, actual_start, actual_end in solution:
            tw = self.get_window(sat_idx, win_idx)

            if actual_start < tw["start_time"]:
                return False

            if actual_end > tw["end_time"]:
                return False

        return True

    def check_time_window_overlap(self, solution):
        """
        同一颗卫星上的 OTW 不能重叠。
        """

        for sat_idx in range(len(self.timewindows)):
            tasks_on_sat = [
                s for s in solution
                if int(s[2]) == sat_idx
            ]

            if len(tasks_on_sat) < 2:
                continue

            intervals = []

            for task_id, win_idx, _, actual_start, actual_end in tasks_on_sat:
                intervals.append((actual_start, actual_end))

            intervals.sort(key=lambda x: x[0])

            for i in range(len(intervals) - 1):
                if intervals[i + 1][0] < intervals[i][1]:
                    return False

        return True

    # =========================================================
    # constraint violation
    # =========================================================

    def constraint_violation(self, solution):
        cv = 0.0

        # 1. 非法分配
        for task_id, win_idx, sat_idx, actual_start, actual_end in solution:
            if (int(sat_idx), int(win_idx)) not in self.valid_set.get(int(task_id), set()):
                cv += 1000.0

        # 2. OTW 超出 VTW
        for task_id, win_idx, sat_idx, actual_start, actual_end in solution:
            tw = self.get_window(sat_idx, win_idx)

            if actual_start < tw["start_time"]:
                cv += (tw["start_time"] - actual_start).total_seconds()

            if actual_end > tw["end_time"]:
                cv += (actual_end - tw["end_time"]).total_seconds()

        # 3. 同一卫星 OTW 重叠
        for sat_idx in range(len(self.timewindows)):
            intervals = []

            for item in solution:
                task_id, win_idx, s_idx, actual_start, actual_end = item

                if int(s_idx) == sat_idx:
                    intervals.append((actual_start, actual_end))

            intervals.sort(key=lambda x: x[0])

            for i in range(len(intervals) - 1):
                if intervals[i + 1][0] < intervals[i][1]:
                    overlap = (
                        intervals[i][1] - intervals[i + 1][0]
                    ).total_seconds()
                    cv += max(0.0, overlap)

        # 4. 重复任务
        seen = set()
        for item in solution:
            task_id = int(item[0])

            if task_id in seen:
                cv += 1000.0
            else:
                seen.add(task_id)

        return cv

    # =========================================================
    # objectives
    # =========================================================

    def total_profit(self, solution):
        """
        只统计实际执行任务。
        没进入 solution 的任务视为不执行。
        """

        total = 0.0

        for task_id, win_idx, sat_idx, actual_start, actual_end in solution:
            total += self.get_profit(task_id)

        return total

    def load_balance(self, solution):
        """
        用各卫星实际观测时长方差表示负载均衡。
        越小越好。
        """

        satellite_load = [0.0] * len(self.timewindows)

        for task_id, win_idx, sat_idx, actual_start, actual_end in solution:
            duration = (actual_end - actual_start).total_seconds()
            satellite_load[int(sat_idx)] += duration

        mean_load = np.mean(satellite_load)
        variance = np.mean([(l - mean_load) ** 2 for l in satellite_load])

        return variance

    def attitude_manoeuvre(self, solution):
        """
        暂时沿用近似姿态代价：
        同一卫星相邻任务 actual_start 对应姿态角差值。
        """

        total_cost = 0.0

        for sat_idx in range(len(self.attitudes)):
            sat_tasks = [
                s for s in solution
                if int(s[2]) == sat_idx
            ]

            if len(sat_tasks) < 2:
                continue

            sat_tasks.sort(key=lambda x: x[3])

            att_df = self.attitudes[sat_idx]

            for i in range(len(sat_tasks) - 1):
                t_curr = sat_tasks[i][3]
                t_next = sat_tasks[i + 1][3]

                idx_curr = (att_df["dt"] - t_curr).abs().idxmin()
                idx_next = (att_df["dt"] - t_next).abs().idxmin()

                curr_angle = att_df.loc[
                    idx_curr,
                    ["angle_x", "angle_y", "angle_z"]
                ].values.astype(float)

                next_angle = att_df.loc[
                    idx_next,
                    ["angle_x", "angle_y", "angle_z"]
                ].values.astype(float)

                total_cost += np.linalg.norm(next_angle - curr_angle)

        return total_cost

    # =========================================================
    # evaluate
    # =========================================================

    def evaluate_solution(self, solution):
        feasible = (
            self.check_task_uniqueness(solution)
            and self.check_window_validity(solution)
            and self.check_observation_inside_window(solution)
            and self.check_time_window_overlap(solution)
        )

        objectives = {
            "total_profit": self.total_profit(solution),
            "load_balance": self.load_balance(solution),
            "attitude_manoeuvre": self.attitude_manoeuvre(solution),
            "image_quality": self.image_quality(solution),
        }

        return objectives, feasible