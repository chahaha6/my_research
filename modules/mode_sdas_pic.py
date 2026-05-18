import numpy as np
import random
import math
import os
from pathlib import Path

from .problem_model import SatelliteSchedulingProblem
from .utils_moea import (non_dominated_sort, crowding_distance,
                        fast_cal_distance, precompute_neighbors, generate_uniform_weights)
from .pic_operator import pic_repair
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD

os.environ['OMP_NUM_THREADS'] = '1'

class MODE_SDAS_PIC:
    def __init__(self, problem, pop_size=100, generations=100,
                 Nsub=9, L=10, Rmem=5, CR=0.5, F=0.5, probVar=0.8,
                 alpha=0.15, sigma=0.01, gamma=0.7,          # 调整 PICEO 参数
                 ref_front_file=None, elite_prob=0.6, archive_size=50):
        """alpha=0.15, sigma=0.01"""
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = generations
        self.Nsub = Nsub
        self.L = L
        self.Rmem = Rmem
        self.CR = CR
        self.F = F
        self.probVar = probVar
        self.alpha = alpha
        self.sigma = sigma
        self.gamma = gamma
        self.elite_prob = elite_prob
        self.archive_size = archive_size

        self.task_options = self._build_task_options()
        self.num_tasks = len(self.task_options)
        self.max_options = [len(opts) for opts in self.task_options.values()]
        self.D = self.num_tasks

        self.W = None
        self.neighbor_list = None
        self.zideal = None
        self.r1 = 0.6 
        self.r2 = 0.85
        self.S = 1.0
        self.gen_count = 0
        self.mem_theta = np.zeros((Rmem, Nsub))

        # 自适应权重增删参数
        self.check_interval = 12
        self.weight_deletion_threshold = 2
        self.max_subspace_num = int(Nsub * 1.5)
        self.min_subspace_num = max(2, Nsub // 2)
        self.last_assoc_gen = np.full(Nsub, -1)
        self.crowd_factor = 1.5

        self.pop_dec = None
        self.pop_obj = None
        self.pop_raw = None
        self.associate = None

        # 双档案精英引导相关
        self.external_archive_dec = []
        self.external_archive_obj = []

        self.hv_history = []
        self.hv_indicator = None
        self.ideal_point = None
        self.nadir_point = None

        self.ref_front = None
        if ref_front_file is not None and os.path.exists(ref_front_file):
            self.ref_front = np.loadtxt(ref_front_file, delimiter=',')
            self.igd_indicator = IGD(self.ref_front)
        else:
            self.igd_indicator = None

    def _build_task_options(self):
        options = {}
        for task_id in self.problem.tasks['task_id']:
            opts = []
            for sat_idx, tw_df in enumerate(self.problem.timewindows):
                matching_rows = tw_df[tw_df['task_id'] == task_id].index.tolist()
                for win_idx in matching_rows:
                    opts.append((sat_idx, win_idx))
            options[task_id] = opts
        return options

    def decode(self, x):
        solution = []
        task_ids = list(self.task_options.keys())
        for i, task_id in enumerate(task_ids):
            choice_idx = int(round(x[i]))
            choice_idx = max(0, min(len(self.task_options[task_id])-1, choice_idx))
            sat_idx, win_idx = self.task_options[task_id][choice_idx]
            solution.append((task_id, win_idx, sat_idx))
        return solution

    def evaluate(self, x):
        sol = self.decode(x)
        raw, feasible = self.problem.evaluate_solution(sol)
        obj = {
            'profit': -raw['total_profit'],
            'load': raw['load_balance'],
            'attitude': raw['attitude_manoeuvre'],
            'quality': -raw['image_quality']
        }
        return obj, raw, feasible

    def get_obj_matrix(self, objs):
        return np.array([[d['profit'], d['load'], d['attitude'], d['quality']] for d in objs])

    def initialize(self):
        pop_dec = np.zeros((self.pop_size, self.D), dtype=float)
        for i in range(self.pop_size):
            for d in range(self.D):
                pop_dec[i, d] = random.randint(0, self.max_options[d]-1)
        return pop_dec

    # ---------- 双档案精英引导的差分变异 ----------
    def de_operator(self, parent_idx):
        D = self.D
        parent_dec = self.pop_dec[parent_idx]
        sub = self.associate[parent_idx]
        if sub >= self.Nsub or sub < 0:
            sub = random.randint(0, self.Nsub - 1)

        if (random.random() < self.elite_prob) and len(self.external_archive_dec) >= 2:
            base_idx = random.randrange(len(self.external_archive_dec))
            base_dec = self.external_archive_dec[base_idx]
            neighbors = self.neighbor_list[sub] if sub < len(self.neighbor_list) and self.neighbor_list[sub] else list(range(self.Nsub))
            avail = [j for j in range(self.pop_size) if self.associate[j] in neighbors]
            if len(avail) < 2:
                avail = list(range(self.pop_size))
            r1, r2 = random.sample(avail, 2)
            x1 = self.pop_dec[r1]
            x2 = self.pop_dec[r2]
        else:
            base_dec = parent_dec
            neighbors = self.neighbor_list[sub] if sub < len(self.neighbor_list) and self.neighbor_list[sub] else list(range(self.Nsub))
            avail = [j for j in range(self.pop_size) if self.associate[j] in neighbors]
            if len(avail) < 2:
                avail = list(range(self.pop_size))
            r1, r2 = random.sample(avail, 2)
            x1 = self.pop_dec[r1]
            x2 = self.pop_dec[r2]

        trial = np.zeros(D)
        for d in range(D):
            ki = self.max_options[d]
            phi = base_dec[d] + self.F * (x1[d] - x2[d])
            phi = np.round(phi)
            phi = phi % ki
            trial[d] = phi

        mask = np.random.rand(D) < self.CR
        if not np.any(mask):
            mask[random.randint(0, D-1)] = True
        child_dec = parent_dec.copy()
        child_dec[mask] = trial[mask]
        child_dec = np.round(child_dec).astype(int)
        child_dec = np.clip(child_dec, 0, np.array(self.max_options)-1)
        return child_dec

    # ---------- 自适应权重增删 ----------
    def adapt_weights(self, gen):
        if gen % self.check_interval != 0 or gen == 0:
            return

        counts = np.zeros(self.Nsub, dtype=int)
        for sub in self.associate:
            if sub < self.Nsub:
                counts[sub] += 1

        for sub in range(self.Nsub):
            if counts[sub] > 0:
                self.last_assoc_gen[sub] = gen

        to_delete = []
        for sub in range(self.Nsub):
            if gen - self.last_assoc_gen[sub] >= self.weight_deletion_threshold:
                if self.Nsub - len(to_delete) > self.min_subspace_num:
                    to_delete.append(sub)

        if to_delete:
            for sub in reversed(to_delete):
                self.W = np.delete(self.W, sub, axis=0)
                self.last_assoc_gen = np.delete(self.last_assoc_gen, sub)
                self.mem_theta = np.delete(self.mem_theta, sub, axis=1)
            self.Nsub = self.W.shape[0]
            self.neighbor_list = precompute_neighbors(self.W, self.L)

        counts = np.zeros(self.Nsub, dtype=int)
        for sub in self.associate:
            if sub < self.Nsub:
                counts[sub] += 1

        if self.Nsub == 0:
            return

        avg_count = np.mean(counts) if np.sum(counts) > 0 else 0
        new_weights = []
        for sub in range(self.Nsub):
            if counts[sub] > self.crowd_factor * avg_count and counts[sub] > 1:
                neighbors = self.neighbor_list[sub] if self.neighbor_list else []
                if not neighbors:
                    continue
                nb = random.choice(neighbors)
                w_new = 0.5 * self.W[sub] + 0.5 * self.W[nb]
                w_new = w_new / np.linalg.norm(w_new)
                new_weights.append(w_new)

        num_to_add = min(len(new_weights), self.max_subspace_num - self.Nsub)
        if num_to_add > 0:
            added = np.array(new_weights[:num_to_add])
            self.W = np.vstack([self.W, added])
            self.last_assoc_gen = np.append(self.last_assoc_gen, np.full(num_to_add, gen))
            extra_mem = np.zeros((self.Rmem, num_to_add))
            self.mem_theta = np.hstack([self.mem_theta, extra_mem])

        self.Nsub = self.W.shape[0]
        self.neighbor_list = precompute_neighbors(self.W, self.L)

    # ---------- 环境选择 ----------
    def select_next_generation(self, combined_dec, combined_obj, combined_raw):
        merged_norm = combined_obj
        obj_mat = self.get_obj_matrix(merged_norm)
        D_all = fast_cal_distance(obj_mat, self.W)
        all_assoc = np.argmin(D_all, axis=1)

        sub_inds = [[] for _ in range(self.Nsub)]
        for idx, sub in enumerate(all_assoc):
            sub_inds[sub].append(idx)

        base = self.pop_size // self.Nsub
        remainder = self.pop_size - base * self.Nsub
        capacities = [base] * self.Nsub
        for i in range(remainder):
            capacities[i] += 1

        selected_indices = []
        for sub in range(self.Nsub):
            inds = sub_inds[sub]
            if not inds:
                continue
            sub_obj_list = [merged_norm[i] for i in inds]
            fronts = non_dominated_sort(sub_obj_list)
            filled = 0
            for f in fronts:
                if filled >= capacities[sub]:
                    break
                if filled + len(f) <= capacities[sub]:
                    selected_indices.extend([inds[i] for i in f])
                    filled += len(f)
                else:
                    need = capacities[sub] - filled
                    cd = crowding_distance(f, sub_obj_list)
                    sorted_f = sorted(f, key=lambda i: cd[i], reverse=True)
                    selected_indices.extend([inds[i] for i in sorted_f[:need]])
                    filled = capacities[sub]
                    break

        if len(selected_indices) < self.pop_size:
            all_indices = set(range(len(merged_norm)))
            remaining = list(all_indices - set(selected_indices))
            if remaining:
                need = self.pop_size - len(selected_indices)
                add = np.random.choice(remaining, min(need, len(remaining)), replace=False)
                selected_indices.extend(add)

        selected_indices = selected_indices[:self.pop_size]
        self.pop_dec = combined_dec[selected_indices]
        self.pop_obj = [merged_norm[i] for i in selected_indices]
        self.pop_raw = [combined_raw[i] for i in selected_indices]

    # ---------- 更新全局精英档案 ----------
    def update_archive(self):
        if len(self.pop_obj) == 0:
            return
        combined_obj = self.external_archive_obj + self.pop_obj
        combined_dec = self.external_archive_dec + [d for d in self.pop_dec]
        if len(combined_obj) == 0:
            return
        fronts = non_dominated_sort(combined_obj)
        new_archive_obj = []
        new_archive_dec = []
        count = 0
        for f in fronts:
            if count >= self.archive_size:
                break
            if count + len(f) <= self.archive_size:
                for i in f:
                    new_archive_obj.append(combined_obj[i])
                    new_archive_dec.append(combined_dec[i])
                count += len(f)
            else:
                need = self.archive_size - count
                cd = crowding_distance(f, combined_obj)
                sorted_f = sorted(f, key=lambda i: cd[i], reverse=True)
                for i in sorted_f[:need]:
                    new_archive_obj.append(combined_obj[i])
                    new_archive_dec.append(combined_dec[i])
                break
        self.external_archive_obj = new_archive_obj
        self.external_archive_dec = new_archive_dec

    def compute_hv(self, objs):
        F = self.get_obj_matrix(objs)
        fronts = non_dominated_sort(objs)
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
        denom[denom < 1e-12] = 1
        norm_nd = (nd_F - self.ideal_point) / denom
        if self.hv_indicator is None:
            self.hv_indicator = Hypervolume(ref_point=np.ones(4))
        hv = self.hv_indicator(norm_nd)
        return hv

    def run(self):
        self.W = generate_uniform_weights(self.Nsub, 4)
        self.neighbor_list = precompute_neighbors(self.W, self.L)

        self.pop_dec = self.initialize()
        self.pop_obj = []
        self.pop_raw = []
        for i in range(self.pop_size):
            obj, raw, _ = self.evaluate(self.pop_dec[i])
            self.pop_obj.append(obj)
            self.pop_raw.append(raw)

        obj_mat = self.get_obj_matrix(self.pop_obj)
        self.zideal = np.min(obj_mat, axis=0)
        D_mat = fast_cal_distance(obj_mat, self.W)
        self.associate = np.argmin(D_mat, axis=1)

        self.update_archive()

        hv = self.compute_hv(self.pop_obj)
        self.hv_history.append(hv)
        print(f"Init: HV={hv:.4f}")

        for gen in range(self.max_gen):
            # 自适应选择概率
            if self.gen_count < self.Rmem:
                prob_sub = np.ones(self.Nsub) / self.Nsub
            else:
                theta_sum = np.sum(self.mem_theta, axis=0)
                if np.all(theta_sum == 0):
                    prob_sub = np.ones(self.Nsub) / self.Nsub
                else:
                    sigma_val = np.mean(theta_sum) / self.Nsub
                    denom = np.sum(theta_sum) + self.Nsub * sigma_val
                    prob_sub = (theta_sum + sigma_val) / denom
                    prob_sub = np.clip(prob_sub, 0, None)
                    prob_sub = prob_sub / prob_sub.sum()

            n_evo = np.zeros(self.Nsub, dtype=int)
            cum_prob = np.cumsum(prob_sub)
            for _ in range(self.pop_size):
                r = random.random()
                sub = np.searchsorted(cum_prob, r)
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
                for k in range(n_evo[sub]):
                    parent_idx = random.choice(idx_sub)
                    child_dec = self.de_operator(parent_idx)
                    offspring_dec.append(child_dec)
                    offspring_sub.append(sub)

            offspring_dec = np.array(offspring_dec) if offspring_dec else np.empty((0, self.D))
            off_obj = []
            off_raw = []
            for dec in offspring_dec:
                obj, raw, _ = self.evaluate(dec)
                off_obj.append(obj)
                off_raw.append(raw)

            if len(offspring_dec) > 0:
                off_obj_mat = self.get_obj_matrix(off_obj)
                self.zideal = np.minimum(self.zideal, np.min(off_obj_mat, axis=0))

            # ---- PICEO 修复（参数已调小）----
            if len(offspring_dec) >= 3:
                offspring_dec, self.r1, self.r2, self.S = pic_repair(
                    self, offspring_dec, off_obj, offspring_sub, gen, self.max_gen)
                # 重新评估修复后的子代
                off_obj = []
                off_raw = []
                for dec in offspring_dec:
                    obj, raw, _ = self.evaluate(dec)
                    off_obj.append(obj)
                    off_raw.append(raw)

            # 合并父子代
            if len(offspring_dec) > 0:
                combined_obj = self.pop_obj + off_obj
                combined_raw = self.pop_raw + off_raw
                combined_dec = np.vstack([self.pop_dec, offspring_dec])
            else:
                combined_obj = self.pop_obj
                combined_raw = self.pop_raw
                combined_dec = self.pop_dec.copy()

            # 环境选择
            self.select_next_generation(combined_dec, combined_obj, combined_raw)

            # 重新关联
            obj_mat_new = self.get_obj_matrix(self.pop_obj)
            D_new = fast_cal_distance(obj_mat_new, self.W)
            self.associate = np.argmin(D_new, axis=1)

            # 更新精英档案
            self.update_archive()

            # 更新推动距离记忆
            theta_sub = np.zeros(self.Nsub)
            for idx, child_dec in enumerate(offspring_dec):
                sub = offspring_sub[idx] if idx < len(offspring_sub) else 0
                if sub >= self.Nsub:
                    sub = random.randint(0, self.Nsub - 1)
                child_obj = off_obj[idx]
                sub_pop_idx = [i for i in range(len(self.pop_obj)) if self.associate[i] == sub]
                if sub_pop_idx:
                    sub_objs = self.get_obj_matrix([self.pop_obj[i] for i in sub_pop_idx])
                    dist_to_ideal = np.sqrt(np.sum((sub_objs - self.zideal)**2, axis=1))
                    min_dist = np.min(dist_to_ideal)
                    child_arr = self.get_obj_matrix([child_obj])[0]
                    child_dist = np.sqrt(np.sum((child_arr - self.zideal)**2))
                    push = min_dist - child_dist
                    theta_sub[sub] += max(0, push)

            if self.gen_count < self.Rmem:
                self.mem_theta[self.gen_count, :] = theta_sub
            else:
                self.mem_theta[:-1, :] = self.mem_theta[1:, :]
                self.mem_theta[-1, :] = theta_sub

            self.gen_count += 1

            # 自适应权重增删
            self.adapt_weights(gen)

            hv = self.compute_hv(self.pop_obj)
            self.hv_history.append(hv)

            profits = [raw['total_profit'] for raw in self.pop_raw]
            loads = [raw['load_balance'] for raw in self.pop_raw]
            attitudes = [raw['attitude_manoeuvre'] for raw in self.pop_raw]
            print(f"Gen {gen+1:3d}: profit_max={max(profits):6.1f}, "
                  f"load_min={min(loads):8.2f}, "
                  f"att_min={min(attitudes):8.2f}, "
                  f"HV={hv:.4f}, Nsub={self.Nsub}")

        return self.pop_dec, self.pop_raw, self.pop_obj, self.hv_history
