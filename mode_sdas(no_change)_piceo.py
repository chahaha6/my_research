import numpy as np
import random
import math
import os
from pathlib import Path

from .problem_model import SatelliteSchedulingProblem
from .utils_moea import (dominates, non_dominated_sort, crowding_distance,
                        fast_cal_distance, precompute_neighbors, generate_uniform_weights)
from .pic_operator import pic_repair

from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD

os.environ['OMP_NUM_THREADS'] = '1'


class MODE_SDAS_PIC:
    def __init__(self, problem, pop_size=100, generations=100,
                 Nsub=10, L=10, Rmem=5, CR=0.5, F=0.5, probVar=0.8,
                 alpha=0.15, sigma=0.01, gamma=0.7,
                 ref_front_file=None):
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

        self.task_options = self._build_task_options()
        self.num_tasks = len(self.task_options)
        self.max_options = [len(opts) for opts in self.task_options.values()]
        self.D = self.num_tasks

        self.W = None
        self.neighbor_list = None
        self.zideal = None
        self.r1 = 0.6 + 0.4 * np.random.rand()
        self.r2 = self.r1 + (1 - self.r1) * np.random.rand()
        self.S = 1.0
        self.gen_count = 0
        self.mem_theta = np.zeros((Rmem, Nsub))

        self.pop_dec = None
        self.pop_obj = None
        self.pop_raw = None
        self.associate = None

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

    def de_operator(self, parent_idx):
        D = self.D
        parent_dec = self.pop_dec[parent_idx]
        sub = self.associate[parent_idx]
        neighbors = self.neighbor_list[sub] if self.neighbor_list[sub] else list(range(self.Nsub))
        avail = [j for j in range(self.pop_size) if self.associate[j] in neighbors]
        if len(avail) < 2:
            avail = list(range(self.pop_size))
        r1, r2 = random.sample(avail, 2)
        trial = np.zeros(D)
        for d in range(D):
            ki = self.max_options[d]
            phi = parent_dec[d] + self.F * (self.pop_dec[r1, d] - self.pop_dec[r2, d])
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
                    sub = random.randint(0, self.Nsub-1)
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

            # ---------- PICEO 修复 ----------
            if len(offspring_dec) >= 3:
                offspring_dec, self.r1, self.r2, self.S = pic_repair(
                    self, offspring_dec, off_obj, offspring_sub, gen, self.max_gen)
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

            obj_mat_comb = self.get_obj_matrix(combined_obj)
            D_all = fast_cal_distance(obj_mat_comb, self.W)
            all_assoc = np.argmin(D_all, axis=1)

            # 子空间环境选择
            sub_inds = [[] for _ in range(self.Nsub)]
            for idx in range(len(combined_obj)):
                sub = all_assoc[idx]
                sub_inds[sub].append(idx)

            kept = []
            for sub in range(self.Nsub):
                inds = sub_inds[sub]
                if not inds:
                    continue
                sub_objs = [combined_obj[i] for i in inds]
                fronts = non_dominated_sort(sub_objs)
                for f in fronts[:1]:
                    for i in f:
                        kept.append(inds[i])

            kept = list(set(kept))
            if len(kept) < self.pop_size:
                remaining = [i for i in range(len(combined_obj)) if i not in kept]
                if remaining:
                    add_n = min(self.pop_size - len(kept), len(remaining))
                    add_idx = np.random.choice(remaining, add_n, replace=False)
                    kept.extend(add_idx)
            kept = kept[:self.pop_size]

            self.pop_dec = combined_dec[kept]
            self.pop_obj = [combined_obj[i] for i in kept]
            self.pop_raw = [combined_raw[i] for i in kept]

            obj_mat_new = self.get_obj_matrix(self.pop_obj)
            D_new = fast_cal_distance(obj_mat_new, self.W)
            self.associate = np.argmin(D_new, axis=1)

            # 更新推动距离记忆
            theta_sub = np.zeros(self.Nsub)
            for idx, child_dec in enumerate(offspring_dec):
                sub = offspring_sub[idx] if idx < len(offspring_sub) else 0
                child_obj = off_obj[idx]
                sub_pop_idx = [i for i in range(len(self.pop_obj)) if self.associate[i] == sub]
                if sub_pop_idx:
                    sub_objs = self.get_obj_matrix([self.pop_obj[i] for i in sub_pop_idx])
                    dist_to_ideal = np.sqrt(np.sum((sub_objs - self.zideal) ** 2, axis=1))
                    min_dist = np.min(dist_to_ideal)
                    child_arr = self.get_obj_matrix([child_obj])[0]
                    child_dist = np.sqrt(np.sum((child_arr - self.zideal) ** 2))
                    push = min_dist - child_dist
                    theta_sub[sub] += max(0, push)

            if self.gen_count < self.Rmem:
                self.mem_theta[self.gen_count, :] = theta_sub
            else:
                self.mem_theta[:-1, :] = self.mem_theta[1:, :]
                self.mem_theta[-1, :] = theta_sub

            self.gen_count += 1

            hv = self.compute_hv(self.pop_obj)
            self.hv_history.append(hv)

            profits = [raw['total_profit'] for raw in self.pop_raw]
            loads = [raw['load_balance'] for raw in self.pop_raw]
            print(f"Gen {gen+1:3d}: profit_max={max(profits):6.1f}, "
                  f"load_min={min(loads):8.2f}, HV={hv:.4f}")

        return self.pop_dec, self.pop_raw, self.pop_obj, self.hv_history