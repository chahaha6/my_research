import numpy as np
import random
import math
import os
from pathlib import Path
from sklearn.cluster import KMeans
from problem_model import SatelliteSchedulingProblem

os.environ['OMP_NUM_THREADS'] = '1'

from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD

# ------------------------------
# 支配关系与非支配排序（全部目标越小越好）
# ------------------------------
def dominates(obj1, obj2):
    return all(obj1[k] <= obj2[k] for k in obj1) and any(obj1[k] < obj2[k] for k in obj1)

def non_dominated_sort(objectives):
    N = len(objectives)
    S = [[] for _ in range(N)]
    n = [0] * N
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if dominates(objectives[p], objectives[q]):
                S[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts

def crowding_distance(front, objectives):
    if len(front) <= 2:
        return {idx: float('inf') for idx in front}
    dist = {idx: 0.0 for idx in front}
    M = len(objectives[0])
    for m in range(M):
        key = list(objectives[0].keys())[m]
        sorted_front = sorted(front, key=lambda idx: objectives[idx][key])
        dist[sorted_front[0]] = float('inf')
        dist[sorted_front[-1]] = float('inf')
        fmax = objectives[sorted_front[-1]][key]
        fmin = objectives[sorted_front[0]][key]
        if fmax == fmin:
            continue
        for i in range(1, len(front)-1):
            dist[sorted_front[i]] += (objectives[sorted_front[i+1]][key] - objectives[sorted_front[i-1]][key]) / (fmax - fmin)
    return dist

def fast_cal_distance(pop_obj, W):
    norm_pop = np.linalg.norm(pop_obj, axis=1, keepdims=True)
    norm_pop[norm_pop < 1e-12] = 1
    pop_norm = pop_obj / norm_pop
    norm_w = np.linalg.norm(W, axis=1, keepdims=True)
    norm_w[norm_w < 1e-12] = 1
    w_norm = W / norm_w
    cos_sim = np.dot(pop_norm, w_norm.T)
    cos_sim = np.clip(cos_sim, -1, 1)
    return np.arccos(cos_sim)

def precompute_neighbors(W, L):
    Nsub = W.shape[0]
    neighbors = [[] for _ in range(Nsub)]
    for i in range(Nsub):
        angles = np.zeros(Nsub)
        for j in range(Nsub):
            if i == j:
                angles[j] = np.inf
            else:
                cos_sim = np.dot(W[i], W[j])
                angles[j] = np.arccos(np.clip(cos_sim, -1, 1))
        idx = np.argsort(angles)[:L]
        neighbors[i] = idx.tolist()
    return neighbors

def generate_uniform_weights(Nsub, M):
    if Nsub == 1:
        return np.ones((1, M)) / M
    rand_vecs = np.random.dirichlet(np.ones(M), size=100 * Nsub)
    kmeans = KMeans(n_clusters=Nsub, n_init=10, random_state=0).fit(rand_vecs)
    W = kmeans.cluster_centers_
    W = W / np.sqrt(np.sum(W**2, axis=1, keepdims=True))
    return W

# ------------------------------
# MODE-SDAS-PIC 主类（全部目标最小化）
# ------------------------------
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

    def pic_repair(self, V_dec, Offspring_obj, offspring_sub, gen, max_gen):
        N_off = V_dec.shape[0]
        D = V_dec.shape[1]
        if N_off < 3:
            return V_dec, self.r1, self.r2, self.S

        CV = np.zeros(N_off)
        obj_mat = self.get_obj_matrix(Offspring_obj)
        idx_cv = np.argsort(CV)
        tau1 = np.zeros(N_off, dtype=int)
        tau1[idx_cv] = np.arange(1, N_off+1)
        fronts = non_dominated_sort(Offspring_obj)
        front_no = np.zeros(N_off, dtype=int)
        for f_idx, front in enumerate(fronts):
            for i in front:
                front_no[i] = f_idx + 1
        tau2 = front_no
        tau = tau1 + tau2

        sort_idx = np.argsort(tau)
        sorted_V = V_dec[sort_idx].copy()
        V_work = sorted_V.copy()
        s_i = self.S + self.sigma * np.random.randn(N_off)
        U = sorted_V.copy()

        N1 = int(np.floor(self.r1 * (N_off - 2)) + 1)
        N2 = int(np.floor(self.r2 * (N_off - 1)) + 1)
        N1 = max(1, min(N_off, N1))
        N2 = max(N1, min(N_off, N2))

        for i in range(N_off):
            if i == 0:
                continue
            if i < N1:
                op = 1
            elif i < N2:
                op = 2
            else:
                op = 3
            k1 = i-1
            k2 = i+1 if i+1 < N_off else i
            for j in range(D):
                if np.random.rand() < self.alpha:
                    if op == 1:
                        delta = s_i[i] * (V_work[k1, j] - V_work[k2, j])
                        U[i, j] = V_work[i, j] + delta
                    elif op == 2:
                        mV = (V_work[i, j] + V_work[k1, j] + V_work[k2, j]) / 3.0
                        delta = s_i[i] * (mV - V_work[i, j])
                        U[i, j] = V_work[i, j] + delta
                    else:
                        minV = min(V_work[i, j], V_work[k1, j], V_work[k2, j])
                        tmp = V_work[i, j] + s_i[i] * (minV - V_work[i, j])
                        maxV = max(V_work[i, j], V_work[k1, j], V_work[k2, j])
                        U[i, j] = tmp + s_i[i] * (maxV - tmp)
                    V_work[i, j] = U[i, j]
                    U[i, j] = int(np.clip(np.round(U[i, j]), 0, self.max_options[j]-1))
                    V_work[i, j] = U[i, j]

        inv_sort = np.argsort(sort_idx)
        U = U[inv_sort]

        push_all = np.zeros(N_off)
        for i in range(N_off):
            orig_idx = sort_idx[i]
            sub = offspring_sub[orig_idx] if orig_idx < len(offspring_sub) else 0
            child_obj = Offspring_obj[orig_idx]
            sub_pop_idx = [k for k in range(len(self.pop_obj)) if self.associate[k] == sub]
            if sub_pop_idx:
                sub_objs = self.get_obj_matrix([self.pop_obj[k] for k in sub_pop_idx])
                dist_to_ideal = np.sqrt(np.sum((sub_objs - self.zideal)**2, axis=1))
                min_dist = np.min(dist_to_ideal)
                child_arr = self.get_obj_matrix([child_obj])[0]
                child_dist = np.sqrt(np.sum((child_arr - self.zideal)**2))
                push = min_dist - child_dist
                push_all[i] = max(0, push)

        c1 = np.mean(push_all[:N1]) if N1 >= 1 else 0
        c2 = np.mean(push_all[N1:N2]) if N2 > N1 else 0
        c3 = np.mean(push_all[N2:]) if N_off > N2 else 0
        total_push = c1 + c2 + c3
        if total_push > 0:
            c1 /= total_push
            c2 /= total_push
            c3 /= total_push

        T = self.max_gen
        if c1 >= c2 and c1 >= c3:
            self.r1 = min(1, self.r1 + 1.0/T)
        elif c2 >= c1 and c2 >= c3:
            self.r2 = min(1, self.r2 + 1.0/T)
        elif c3 >= c1 and c3 >= c2:
            self.r2 = max(self.r1, self.r2 - 1.0/T)
        self.r1 = np.clip(self.r1, 0, 1)
        self.r2 = np.clip(self.r2, self.r1, 1)

        beta = gen / max_gen if max_gen > 0 else 0
        S1 = 1 - 1 / (1 + math.exp(10 - 20 * beta))
        if total_push > 0 and np.sum(push_all) > 0:
            S2 = np.sum(push_all * s_i) / np.sum(push_all)
            self.S = self.gamma * S1 + (1 - self.gamma) * S2
        else:
            self.S = S1

        return U, self.r1, self.r2, self.S

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

            if len(offspring_dec) >= 3:
                offspring_dec, self.r1, self.r2, self.S = self.pic_repair(
                    offspring_dec, off_obj, offspring_sub, gen, self.max_gen)
                off_obj = []
                off_raw = []
                for dec in offspring_dec:
                    obj, raw, _ = self.evaluate(dec)
                    off_obj.append(obj)
                    off_raw.append(raw)

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

            theta_sub = np.zeros(self.Nsub)
            for idx, child_dec in enumerate(offspring_dec):
                sub = offspring_sub[idx] if idx < len(offspring_sub) else 0
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

            hv = self.compute_hv(self.pop_obj)
            self.hv_history.append(hv)

            profits = [raw['total_profit'] for raw in self.pop_raw]
            loads = [raw['load_balance'] for raw in self.pop_raw]
            print(f"Gen {gen+1:3d}: profit_max={max(profits):6.1f}, "
                  f"load_min={min(loads):8.2f}, HV={hv:.4f}")

        return self.pop_dec, self.pop_raw, self.pop_obj, self.hv_history

# -------------------------------
# 主程序
# -------------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "CSV_Data"

    task_file = DATA_DIR / "task_info_standard.csv"
    attitude_files = [DATA_DIR / f"outputattitude_{i}.csv" for i in range(1, 11)]
    timewindow_files = [DATA_DIR / f"outputtimewindow_{i}.csv" for i in range(1, 11)]

    problem = SatelliteSchedulingProblem(str(task_file),
                                         [str(f) for f in attitude_files],
                                         [str(f) for f in timewindow_files])

    ref_front_file = BASE_DIR / "results" / "reference_front.csv"
    if not ref_front_file.exists():
        ref_front_file = None

    alg = MODE_SDAS_PIC(problem, pop_size=20, generations=30,
                        Nsub=5, L=3, Rmem=5, CR=0.5, F=0.5,
                        alpha=0.15, sigma=0.01, gamma=0.7,
                        ref_front_file=ref_front_file)

    final_pop_dec, final_raw, final_obj, hv_history = alg.run()

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    np.savetxt(results_dir / "hv_history.csv", np.array(hv_history), delimiter=',')

    import pandas as pd
    best_idx = np.argmax([raw['total_profit'] for raw in final_raw])
    best_dec = final_pop_dec[best_idx]
    best_sol = alg.decode(best_dec)
    df = pd.DataFrame(best_sol, columns=["task_id", "window_idx", "sat_idx"])
    df.to_csv(results_dir / "final_solution_pic.csv", index=False)

    if alg.igd_indicator is not None:
        min_objs = [alg.evaluate(dec)[0] for dec in final_pop_dec]
        fronts = non_dominated_sort(min_objs)
        nd_F = alg.get_obj_matrix([min_objs[i] for i in fronts[0]])
        igd_val = alg.igd_indicator(nd_F)
        print(f"IGD = {igd_val:.4f}")
        with open(results_dir / "igd.txt", 'w') as f:
            f.write(f"{igd_val}\n")

    print(f"HV history saved to {results_dir / 'hv_history.csv'}")
    print(f"Final solution saved to {results_dir / 'final_solution_pic.csv'}")