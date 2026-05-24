import os
os.environ["OMP_NUM_THREADS"] = "1"

import random
from pathlib import Path
import numpy as np
import pandas as pd

from .utils_moea import (
    non_dominated_sort,
    crowding_distance,
    fast_cal_distance,
    precompute_neighbors,
    generate_uniform_weights,
)
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from .dpic_operator import dpic_repair


class MODE_SDAS_DPIC_DUAL_ARCHIVE:
    """
    MODE-SDAS + DPIC + 双档案机制。

    与原始 MODE-SDAS 的区别：
    1. 保留 MODE-SDAS 的子空间选择、DE 生成子代、可行性优先环境选择；
    2. 增加两个外部档案：
       - feasible_archive：保存可行非支配精英解；
       - infeasible_archive：保存低约束违反度的不可行解；
    3. 在 DE 变异时，以 archive_prob 的概率从档案中选 base 向量，引导搜索；
    4. 在 DE 子代生成后加入 DPIC 离散种群图像卷积修复。

    注意：本文件不包含子空间自适应增删，适合用作最终主算法的
    “MODE-SDAS + DPIC + 双档案”版本。
    """

    def __init__(self, problem, pop_size=100, generations=100,
                 Nsub=21, L=7, CR=0.85, F=0.40,
                 archive_prob=0.08,
                 feasible_archive_size=200,
                 infeasible_archive_size=200,
                 alpha=0.018, sigma=0.0015, gamma=0.90,
                 use_dpic=True,
                 ref_front_file=None,
                 execute_prob=0.15):
        self.problem = problem
        self.pop_size = int(pop_size)
        self.max_gen = int(generations)
        self.Nsub = int(Nsub)
        self.L = int(L)
        self.CR = float(CR)
        self.F = float(F)
        self.execute_prob = float(execute_prob)

        # 双档案参数
        self.archive_prob = float(archive_prob)
        self.feasible_archive_size = int(feasible_archive_size)
        self.infeasible_archive_size = int(infeasible_archive_size)

        # DPIC 参数：当前实验效果偏弱时，采用“弱扰动 + 大档案”的稳健设置。
        # alpha / sigma 较小，避免卷积修复过度破坏优质子代；
        # gamma 较大，使卷积强度更新更平滑。
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.use_dpic = bool(use_dpic)
        self.r1 = 0.25
        self.r2 = 0.70
        self.S = 1.0

        self.task_options = self._build_task_options()
        self.task_ids = list(self.task_options.keys())
        self.num_tasks = len(self.task_ids)
        self.option_counts = [len(self.task_options[t]) for t in self.task_ids]

        self.D_task = self.num_tasks
        self.D_ratio = self.num_tasks
        self.D = self.D_task + self.D_ratio
        self.max_options = self.option_counts

        self.pop_dec = None
        self.pop_obj = None
        self.pop_raw = None
        self.pop_feas = None
        self.pop_cv = None
        self.associate = None

        self.W = None
        self.neighbor_list = None
        self.zideal = None
        self.mem_theta = np.zeros(self.Nsub)

        # 双档案
        self.feasible_archive_dec = []
        self.feasible_archive_obj = []
        self.infeasible_archive_dec = []
        self.infeasible_archive_cv = []

        self.hv_history = []
        self.hv_indicator = None
        self.ideal_point = None
        self.nadir_point = None

        if ref_front_file is not None and Path(ref_front_file).exists():
            self.ref_front = np.loadtxt(ref_front_file, delimiter=",")
            self.igd_indicator = IGD(self.ref_front)
        else:
            self.ref_front = None
            self.igd_indicator = None

    # =========================================================
    # Build task options
    # =========================================================

    def _build_task_options(self):
        options = {}
        for task_id in self.problem.tasks["task_id"]:
            task_id = int(task_id)
            opts = []
            for sat_idx, tw_df in enumerate(self.problem.timewindows):
                rows = tw_df[tw_df["task_id"] == task_id].index.tolist()
                for win_idx in rows:
                    opts.append((sat_idx, win_idx))
            options[task_id] = opts
        return options

    # =========================================================
    # Decode / evaluate
    # =========================================================

    def decode(self, x):
        x = np.asarray(x, dtype=float)
        solution = []

        for i, task_id in enumerate(self.task_ids):
            opts = self.task_options[task_id]
            if len(opts) == 0:
                continue

            k = len(opts)
            choice_gene = int(round(x[i]))
            choice_gene = max(0, min(k, choice_gene))
            if choice_gene == 0:
                continue

            choice_idx = choice_gene - 1
            sat_idx, win_idx = opts[choice_idx]
            tw = self.problem.timewindows[sat_idx].iloc[win_idx]

            duration = float(
                self.problem.tasks.loc[
                    self.problem.tasks["task_id"] == task_id, "duration"
                ].values[0]
            )
            duration_td = pd.to_timedelta(duration, unit="s")

            earliest = tw["start_time"]
            latest = tw["end_time"] - duration_td

            ratio_gene = float(x[self.D_task + i]) if self.D_task + i < len(x) else 0.0
            ratio = max(0.0, min(1.0, ratio_gene))

            if latest < earliest:
                actual_start = earliest
            else:
                available_seconds = (latest - earliest).total_seconds()
                actual_start = earliest + pd.to_timedelta(ratio * available_seconds, unit="s")

            actual_end = actual_start + duration_td
            solution.append((task_id, win_idx, sat_idx, actual_start, actual_end))

        return solution

    def evaluate(self, x):
        sol = self.decode(x)
        raw, feasible = self.problem.evaluate_solution(sol)
        obj = {
            "profit": -raw["total_profit"],
            "load": raw["load_balance"],
            "attitude": raw["attitude_manoeuvre"],
            "quality": -raw["image_quality"],
        }
        return obj, raw, feasible

    def get_obj_matrix(self, objs):
        return np.array([
            [d["profit"], d["load"], d["attitude"], d["quality"]]
            for d in objs
        ], dtype=float)

    def constraint_violation(self, x):
        sol = self.decode(x)
        return float(self.problem.constraint_violation(sol))

    # =========================================================
    # Encoding helpers / operators
    # =========================================================

    def repair_individual(self, individual):
        individual = np.asarray(individual, dtype=float).copy()

        for d in range(self.D_task):
            k = self.option_counts[d]
            if k <= 0:
                individual[d] = 0
            else:
                individual[d] = int(round(individual[d]))
                individual[d] = max(0, min(k, individual[d]))

        for d in range(self.D_task, self.D):
            individual[d] = max(0.0, min(1.0, float(individual[d])))

        return individual

    def initialize(self):
        pop_dec = np.zeros((self.pop_size, self.D), dtype=float)

        for i in range(self.pop_size):
            for d in range(self.D_task):
                k = self.option_counts[d]
                if k <= 0:
                    pop_dec[i, d] = 0
                else:
                    if np.random.rand() < self.execute_prob:
                        pop_dec[i, d] = np.random.randint(1, k + 1)
                    else:
                        pop_dec[i, d] = 0

            for d in range(self.D_task, self.D):
                pop_dec[i, d] = np.random.rand()

        return pop_dec

    def _sample_archive_base(self, parent_dec):
        """从双档案中抽取 base 向量。优先使用可行精英档案。"""
        if random.random() >= self.archive_prob:
            return parent_dec.copy()

        if self.feasible_archive_dec:
            return np.array(random.choice(self.feasible_archive_dec), dtype=float).copy()

        if self.infeasible_archive_dec:
            return np.array(random.choice(self.infeasible_archive_dec), dtype=float).copy()

        return parent_dec.copy()

    def de_operator(self, parent_idx):
        parent_dec = self.pop_dec[parent_idx]
        sub = int(self.associate[parent_idx]) if self.associate is not None else -1

        if sub < 0 or sub >= self.Nsub:
            sub = random.randint(0, self.Nsub - 1)

        neighbors = (
            self.neighbor_list[sub]
            if self.neighbor_list is not None and sub < len(self.neighbor_list) and self.neighbor_list[sub]
            else list(range(self.Nsub))
        )

        # 优先从邻域可行解中选择差分向量
        avail = [
            j for j in range(self.pop_size)
            if self.associate is not None
            and self.associate[j] in neighbors
            and self.pop_feas[j]
        ]
        if len(avail) < 2:
            avail = [j for j in range(self.pop_size) if self.pop_feas[j]]
        if len(avail) < 2:
            avail = list(range(self.pop_size))

        if len(avail) >= 2:
            r1_idx, r2_idx = random.sample(avail, 2)
        else:
            r1_idx = r2_idx = 0

        x1 = self.pop_dec[r1_idx]
        x2 = self.pop_dec[r2_idx]
        base_dec = self._sample_archive_base(parent_dec)

        trial = np.zeros(self.D, dtype=float)
        for d in range(self.D):
            phi = base_dec[d] + self.F * (x1[d] - x2[d])

            if d < self.D_task:
                k = self.option_counts[d]
                if k <= 0:
                    trial[d] = 0
                else:
                    trial[d] = int(round(phi)) % (k + 1)
            else:
                trial[d] = max(0.0, min(1.0, phi))

        mask = np.random.rand(self.D) < self.CR
        if self.D > 0 and not np.any(mask):
            mask[random.randint(0, self.D - 1)] = True

        child_dec = parent_dec.copy()
        child_dec[mask] = trial[mask]
        return self.repair_individual(child_dec)

    # =========================================================
    # DPIC helper
    # =========================================================

    def apply_dpic_operator(self, offspring_dec, off_obj, offspring_sub, gen):
        """
        对 DE 生成的子代执行 DPIC 修复。

        编码结构：
        - 前 D_task 维：离散任务执行/窗口选择基因，合法范围 0..k；
        - 后 D_task 维：连续 OTW ratio，合法范围 [0, 1]。

        DPIC 只修复前半部分离散基因，ratio 部分保持不变。
        这样可以避免图像卷积破坏实际观测时间位置，同时保持任务/窗口选择合法。
        """
        if (not self.use_dpic) or len(offspring_dec) < 3:
            return offspring_dec

        offspring_dec = np.asarray(offspring_dec, dtype=float)
        task_part = offspring_dec[:, :self.D_task].copy()
        ratio_part = offspring_dec[:, self.D_task:].copy()

        old_max_options = self.max_options
        try:
            # 对于离散任务选择基因，max_options[j] 是真实上界 k，合法值为 0..k。
            self.max_options = self.option_counts.copy()
            repaired_task, self.r1, self.r2, self.S = dpic_repair(
                self,
                task_part,
                off_obj,
                offspring_sub,
                gen,
                self.max_gen,
                ratio_part=ratio_part,
            )
            repaired_task = np.asarray(repaired_task, dtype=float)
        finally:
            self.max_options = old_max_options

        repaired_dec = np.hstack([repaired_task, ratio_part])
        repaired_dec = np.array([self.repair_individual(dec) for dec in repaired_dec], dtype=float)
        return repaired_dec

    # =========================================================
    # 双档案更新
    # =========================================================

    def update_archives(self):
        """
        更新双档案：
        - feasible_archive：当前种群可行解 + 历史可行档案，按非支配排序和拥挤距离截断；
        - infeasible_archive：当前种群不可行解 + 历史不可行档案，按 CV 从小到大截断。
        """
        if self.pop_dec is None or self.pop_obj is None:
            return

        # ---------- 可行非支配档案 ----------
        feasible_idx = [i for i, f in enumerate(self.pop_feas) if f]
        cur_fea_dec = [self.pop_dec[i].copy() for i in feasible_idx]
        cur_fea_obj = [self.pop_obj[i] for i in feasible_idx]

        combined_fea_dec = self.feasible_archive_dec + cur_fea_dec
        combined_fea_obj = self.feasible_archive_obj + cur_fea_obj

        if combined_fea_obj:
            fronts = non_dominated_sort(combined_fea_obj)
            new_dec = []
            new_obj = []

            for front in fronts:
                if len(new_dec) >= self.feasible_archive_size:
                    break

                if len(new_dec) + len(front) <= self.feasible_archive_size:
                    chosen = front
                else:
                    need = self.feasible_archive_size - len(new_dec)
                    cd = crowding_distance(front, combined_fea_obj)
                    chosen = sorted(front, key=lambda idx: cd[idx], reverse=True)[:need]

                for idx in chosen:
                    new_dec.append(np.array(combined_fea_dec[idx], dtype=float).copy())
                    new_obj.append(combined_fea_obj[idx])

            self.feasible_archive_dec = new_dec
            self.feasible_archive_obj = new_obj

        # ---------- 不可行低 CV 档案 ----------
        infeasible_idx = [i for i, f in enumerate(self.pop_feas) if not f]
        cur_inf_dec = [self.pop_dec[i].copy() for i in infeasible_idx]
        cur_inf_cv = [float(self.pop_cv[i]) for i in infeasible_idx]

        combined_inf_dec = self.infeasible_archive_dec + cur_inf_dec
        combined_inf_cv = self.infeasible_archive_cv + cur_inf_cv

        if combined_inf_dec:
            order = np.argsort(combined_inf_cv)
            order = order[:self.infeasible_archive_size]
            self.infeasible_archive_dec = [np.array(combined_inf_dec[i], dtype=float).copy() for i in order]
            self.infeasible_archive_cv = [float(combined_inf_cv[i]) for i in order]

    # =========================================================
    # HV and environmental selection
    # =========================================================

    def compute_hv(self):
        feasible_obj = [self.pop_obj[i] for i, f in enumerate(self.pop_feas) if f]
        if not feasible_obj:
            return 0.0

        F = self.get_obj_matrix(feasible_obj)
        fronts = non_dominated_sort(feasible_obj)
        nd_F = F[fronts[0]]

        if self.ideal_point is None:
            self.ideal_point = np.min(nd_F, axis=0)
        else:
            self.ideal_point = np.minimum(self.ideal_point, np.min(nd_F, axis=0))

        if self.nadir_point is None:
            self.nadir_point = np.max(nd_F, axis=0)
        else:
            self.nadir_point = np.maximum(self.nadir_point, np.max(nd_F, axis=0))

        denom = self.nadir_point - self.ideal_point
        denom[denom < 1e-12] = 1.0
        norm_nd = (nd_F - self.ideal_point) / denom

        if self.hv_indicator is None:
            self.hv_indicator = Hypervolume(ref_point=np.ones(4))

        return float(self.hv_indicator(norm_nd))

    def select_next_generation(self, combined_dec, combined_obj, combined_raw,
                               combined_feas, combined_cv):
        feasible_idx = [i for i, f in enumerate(combined_feas) if f]
        infeasible_idx = [i for i, f in enumerate(combined_feas) if not f]
        selected = []

        if len(feasible_idx) >= self.pop_size:
            feasible_obj = [combined_obj[i] for i in feasible_idx]
            obj_mat = self.get_obj_matrix(feasible_obj)
            D_all = fast_cal_distance(obj_mat, self.W)
            all_assoc = np.argmin(D_all, axis=1)

            base = self.pop_size // self.Nsub
            remainder = self.pop_size - base * self.Nsub
            capacities = [base] * self.Nsub
            for i in range(remainder):
                capacities[i] += 1

            sub_inds = [[] for _ in range(self.Nsub)]
            for local_idx, sub in enumerate(all_assoc):
                sub_inds[sub].append(local_idx)

            selected_local = []
            for sub in range(self.Nsub):
                inds = sub_inds[sub]
                if not inds:
                    continue
                sub_obj = [feasible_obj[i] for i in inds]
                fronts = non_dominated_sort(sub_obj)
                filled = 0
                for front in fronts:
                    if filled >= capacities[sub]:
                        break
                    if filled + len(front) <= capacities[sub]:
                        selected_local.extend([inds[i] for i in front])
                        filled += len(front)
                    else:
                        need = capacities[sub] - filled
                        cd = crowding_distance(front, sub_obj)
                        sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                        selected_local.extend([inds[i] for i in sorted_front[:need]])
                        break

            selected = [feasible_idx[i] for i in selected_local]

            if len(selected) < self.pop_size:
                selected_set = set(selected)
                remaining_local = [
                    i for i in range(len(feasible_idx))
                    if feasible_idx[i] not in selected_set
                ]
                if remaining_local:
                    remaining_obj = [feasible_obj[i] for i in remaining_local]
                    fronts = non_dominated_sort(remaining_obj)
                    for front in fronts:
                        if len(selected) >= self.pop_size:
                            break
                        if len(selected) + len(front) <= self.pop_size:
                            selected.extend([feasible_idx[remaining_local[i]] for i in front])
                        else:
                            need = self.pop_size - len(selected)
                            cd = crowding_distance(front, remaining_obj)
                            sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                            selected.extend([
                                feasible_idx[remaining_local[i]]
                                for i in sorted_front[:need]
                            ])
                            break
        else:
            selected.extend(feasible_idx)
            remain = self.pop_size - len(selected)
            if remain > 0 and infeasible_idx:
                sorted_inf = sorted(infeasible_idx, key=lambda i: combined_cv[i])
                selected.extend(sorted_inf[:remain])

        if len(selected) < self.pop_size:
            need = self.pop_size - len(selected)
            if selected:
                selected.extend(np.random.choice(selected, need, replace=True).tolist())
            else:
                all_idx = list(range(len(combined_dec)))
                selected.extend(np.random.choice(all_idx, need, replace=True).tolist())

        selected = selected[:self.pop_size]
        self.pop_dec = np.array([combined_dec[i] for i in selected], dtype=float)
        self.pop_obj = [combined_obj[i] for i in selected]
        self.pop_raw = [combined_raw[i] for i in selected]
        self.pop_feas = [combined_feas[i] for i in selected]
        self.pop_cv = [combined_cv[i] for i in selected]

    def _evaluate_population(self, pop_dec):
        objs, raws, feas, cvs = [], [], [], []
        for dec in pop_dec:
            dec = self.repair_individual(dec)
            obj, raw, feasible = self.evaluate(dec)
            objs.append(obj)
            raws.append(raw)
            feas.append(feasible)
            cvs.append(0.0 if feasible else float(self.constraint_violation(dec)))
        return objs, raws, feas, cvs

    # =========================================================
    # Run
    # =========================================================

    def run(self):
        self.W = generate_uniform_weights(self.Nsub, 4)
        self.neighbor_list = precompute_neighbors(self.W, min(self.L, max(1, self.Nsub - 1)))

        self.pop_dec = self.initialize()
        self.pop_obj, self.pop_raw, self.pop_feas, self.pop_cv = self._evaluate_population(self.pop_dec)

        obj_mat = self.get_obj_matrix(self.pop_obj)
        self.zideal = np.min(obj_mat, axis=0)
        D_mat = fast_cal_distance(obj_mat, self.W)
        self.associate = np.argmin(D_mat, axis=1)

        self.update_archives()

        hv = self.compute_hv()
        self.hv_history.append(hv)
        print(f"Init: HV={hv:.4f}, feasible ratio={sum(self.pop_feas) / self.pop_size:.2f}")

        self.mem_theta = np.zeros(self.Nsub)

        for gen in range(self.max_gen):
            theta_sum = np.sum(self.mem_theta)
            if theta_sum <= 1e-12:
                prob_sub = np.ones(self.Nsub) / self.Nsub
            else:
                sigma_val = np.mean(self.mem_theta)
                denom = np.sum(self.mem_theta + sigma_val)
                prob_sub = (self.mem_theta + sigma_val) / denom if denom > 1e-12 else np.ones(self.Nsub) / self.Nsub
                prob_sub = np.clip(prob_sub, 0, None)
                prob_sub = prob_sub / prob_sub.sum()

            n_evo = np.zeros(self.Nsub, dtype=int)
            cum_prob = np.cumsum(prob_sub)
            for _ in range(self.pop_size):
                sub = np.searchsorted(cum_prob, random.random())
                if sub >= self.Nsub:
                    sub = random.randint(0, self.Nsub - 1)
                n_evo[sub] += 1

            offspring_dec = []
            offspring_sub = []
            for sub in range(self.Nsub):
                if n_evo[sub] == 0:
                    continue
                idx_sub = [i for i in range(self.pop_size) if self.associate[i] == sub]
                if not idx_sub:
                    idx_sub = list(range(self.pop_size))
                for _ in range(n_evo[sub]):
                    parent_idx = random.choice(idx_sub)
                    offspring_dec.append(self.de_operator(parent_idx))
                    offspring_sub.append(sub)

            offspring_dec = np.array(offspring_dec, dtype=float) if offspring_dec else np.empty((0, self.D), dtype=float)
            off_obj, off_raw, off_feas, off_cv = self._evaluate_population(offspring_dec)

            if len(offspring_dec) > 0:
                off_obj_mat = self.get_obj_matrix(off_obj)
                self.zideal = np.minimum(self.zideal, np.min(off_obj_mat, axis=0))

                # -------------------------------------------------
                # DPIC insertion point:
                # DE 子代生成并初评后、父子代合并前。
                # -------------------------------------------------
                if self.use_dpic and len(offspring_dec) >= 3:
                    offspring_dec = self.apply_dpic_operator(offspring_dec, off_obj, offspring_sub, gen)
                    off_obj, off_raw, off_feas, off_cv = self._evaluate_population(offspring_dec)
                    off_obj_mat = self.get_obj_matrix(off_obj)
                    self.zideal = np.minimum(self.zideal, np.min(off_obj_mat, axis=0))

                combined_dec = np.vstack([self.pop_dec, offspring_dec])
                combined_obj = self.pop_obj + off_obj
                combined_raw = self.pop_raw + off_raw
                combined_feas = self.pop_feas + off_feas
                combined_cv = self.pop_cv + off_cv
            else:
                combined_dec = self.pop_dec.copy()
                combined_obj = self.pop_obj.copy()
                combined_raw = self.pop_raw.copy()
                combined_feas = self.pop_feas.copy()
                combined_cv = self.pop_cv.copy()

            self.select_next_generation(combined_dec, combined_obj, combined_raw,
                                        combined_feas, combined_cv)

            obj_mat_new = self.get_obj_matrix(self.pop_obj)
            D_new = fast_cal_distance(obj_mat_new, self.W)
            self.associate = np.argmin(D_new, axis=1)

            self.update_archives()

            # Update push-distance memory using feasible offspring only.
            self.mem_theta = np.zeros(self.Nsub)
            for idx, sub in enumerate(offspring_sub):
                if idx >= len(off_obj) or idx >= len(off_feas) or not off_feas[idx]:
                    continue
                sub_pop_idx = [
                    i for i in range(len(self.pop_obj))
                    if self.associate[i] == sub and self.pop_feas[i]
                ]
                if sub_pop_idx:
                    sub_objs = self.get_obj_matrix([self.pop_obj[i] for i in sub_pop_idx])
                    dist_to_ideal = np.sqrt(np.sum((sub_objs - self.zideal) ** 2, axis=1))
                    child_arr = self.get_obj_matrix([off_obj[idx]])[0]
                    child_dist = np.sqrt(np.sum((child_arr - self.zideal) ** 2))
                    self.mem_theta[sub] += max(0.0, np.min(dist_to_ideal) - child_dist)

            hv = self.compute_hv()
            self.hv_history.append(hv)

            profits = [raw["total_profit"] for raw in self.pop_raw]
            loads = [raw["load_balance"] for raw in self.pop_raw]
            attitudes = [raw["attitude_manoeuvre"] for raw in self.pop_raw]
            qualities = [raw["image_quality"] for raw in self.pop_raw]
            task_counts = [len(self.decode(dec)) for dec in self.pop_dec]
            fea_ratio = sum(self.pop_feas) / self.pop_size

            print(
                f"Gen {gen + 1:3d}: "
                f"profit_max={max(profits):.2f}, profit_avg={np.mean(profits):.2f}, "
                f"load_min={min(loads):.2f}, att_min={min(attitudes):.2f}, "
                f"quality_max={max(qualities):.2f}, quality_avg={np.mean(qualities):.2f}, "
                f"tasks_avg={np.mean(task_counts):.2f}, HV={hv:.4f}, "
                f"feasible ratio={fea_ratio:.2f}, "
                f"fea_archive={len(self.feasible_archive_dec)}, inf_archive={len(self.infeasible_archive_dec)}"
            )

        return self.pop_dec, self.pop_raw, self.pop_obj, self.hv_history


# 兼容性别名，可按需导入
MODE_SDAS_DPIC_ARCHIVE = MODE_SDAS_DPIC_DUAL_ARCHIVE
MODE_SDAS_DPIC_DA = MODE_SDAS_DPIC_DUAL_ARCHIVE
