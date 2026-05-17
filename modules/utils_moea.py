import numpy as np
import math
from sklearn.cluster import KMeans

# ------------------------------
# 支配关系（所有目标越小越好）
# ------------------------------
def dominates(obj1, obj2):
    return all(obj1[k] <= obj2[k] for k in obj1) and any(obj1[k] < obj2[k] for k in obj1)

# ------------------------------
# 非支配排序
# ------------------------------
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

# ------------------------------
# 拥挤距离
# ------------------------------
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

# ------------------------------
# 角度距离
# ------------------------------
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

# ------------------------------
# 邻域预计算
# ------------------------------
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

# ------------------------------
# 均匀权重向量生成
# ------------------------------
def generate_uniform_weights(Nsub, M):
    if Nsub == 1:
        return np.ones((1, M)) / M
    rand_vecs = np.random.dirichlet(np.ones(M), size=100 * Nsub)
    kmeans = KMeans(n_clusters=Nsub, n_init=10, random_state=0).fit(rand_vecs)
    W = kmeans.cluster_centers_
    W = W / np.sqrt(np.sum(W**2, axis=1, keepdims=True))
    return W