import numpy as np
import random
import os
from .problem_model import SatelliteSchedulingProblem
from .utils_moea import non_dominated_sort, crowding_distance
from pymoo.indicators.hv import Hypervolume

os.environ['OMP_NUM_THREADS'] = '1'

class NSGA2:
    def __init__(self, problem, pop_size=100, generations=100,
                 crossover_prob=0.9, mutation_prob=0.1,
                 ref_front_file=None):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.task_options = self._build_task_options()
        self.num_tasks = len(self.task_options)
        self.max_options = [len(opts) for opts in self.task_options.values()]
        self.D = self.num_tasks

        self.pop_dec = None
        self.pop_obj = None
        self.pop_raw = None

        self.hv_history = []
        self.hv_indicator = None
        self.ideal_point = None
        self.nadir_point = None

        if ref_front_file is not None and os.path.exists(ref_front_file):
            from pymoo.indicators.igd import IGD
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
            choice_idx = int(x[i])
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
        pop_dec = np.zeros((self.pop_size, self.D), dtype=int)
        for i in range(self.pop_size):
            for d in range(self.D):
                pop_dec[i, d] = random.randint(0, self.max_options[d]-1)
        return pop_dec

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_prob:
            point = random.randint(1, self.D-1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        for d in range(self.D):
            if random.random() < self.mutation_prob:
                individual[d] = random.randint(0, self.max_options[d]-1)
        return individual

    def tournament_select(self, fronts, cd):
        pool = random.sample(range(self.pop_size), 2)
        rank0 = next(fi for fi, f in enumerate(fronts) if pool[0] in f)
        rank1 = next(fi for fi, f in enumerate(fronts) if pool[1] in f)
        if rank0 < rank1:
            return pool[0]
        elif rank1 < rank0:
            return pool[1]
        else:
            if cd[pool[0]] > cd[pool[1]]:
                return pool[0]
            else:
                return pool[1]

    def select_next_generation(self, combined_dec, combined_obj, combined_raw):
        N = len(combined_obj)
        fronts = non_dominated_sort(combined_obj)
        selected = []
        rank = 0
        while len(selected) + len(fronts[rank]) <= self.pop_size:
            selected.extend(fronts[rank])
            rank += 1
            if rank >= len(fronts):
                break
        if len(selected) < self.pop_size:
            last_front = fronts[rank]
            # 修正：直接传入 combined_obj 作为完整目标列表
            cd_dict = crowding_distance(last_front, combined_obj)
            sorted_last = sorted(last_front, key=lambda i: cd_dict[i], reverse=True)
            remain = self.pop_size - len(selected)
            selected.extend(sorted_last[:remain])

        self.pop_dec = np.array([combined_dec[i] for i in selected])
        self.pop_obj = [combined_obj[i] for i in selected]
        self.pop_raw = [combined_raw[i] for i in selected]

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
        self.pop_dec = self.initialize()
        self.pop_obj = []
        self.pop_raw = []
        for i in range(self.pop_size):
            obj, raw, _ = self.evaluate(self.pop_dec[i])
            self.pop_obj.append(obj)
            self.pop_raw.append(raw)

        hv = self.compute_hv(self.pop_obj)
        self.hv_history.append(hv)
        print(f"Init: HV={hv:.4f}")

        for gen in range(self.max_gen):
            fronts = non_dominated_sort(self.pop_obj)
            cd = {}
            for f in fronts:
                # 修正：传入 self.pop_obj 作为完整目标列表
                cd.update(crowding_distance(f, self.pop_obj))

            offspring_dec = []
            for _ in range(self.pop_size // 2):
                p1 = self.tournament_select(fronts, cd)
                p2 = self.tournament_select(fronts, cd)
                c1, c2 = self.crossover(self.pop_dec[p1], self.pop_dec[p2])
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                offspring_dec.append(c1)
                offspring_dec.append(c2)
            offspring_dec = np.array(offspring_dec)

            off_obj = []
            off_raw = []
            for dec in offspring_dec:
                obj, raw, _ = self.evaluate(dec)
                off_obj.append(obj)
                off_raw.append(raw)

            combined_dec = np.vstack([self.pop_dec, offspring_dec])
            combined_obj = self.pop_obj + off_obj
            combined_raw = self.pop_raw + off_raw
            self.select_next_generation(combined_dec, combined_obj, combined_raw)

            hv = self.compute_hv(self.pop_obj)
            self.hv_history.append(hv)

            profits = [raw['total_profit'] for raw in self.pop_raw]
            loads = [raw['load_balance'] for raw in self.pop_raw]
            attitudes = [raw['attitude_manoeuvre'] for raw in self.pop_raw]
            print(f"Gen {gen+1:3d}: profit_max={max(profits):6.1f}, "
                  f"load_min={min(loads):8.2f}, "
                  f"att_min={min(attitudes):8.2f}, "
                  f"HV={hv:.4f}")

        return self.pop_dec, self.pop_raw, self.pop_obj, self.hv_history