import numpy as np
import math
from sklearn.cluster import KMeans
from pymoo.indicators.hv import Hypervolume

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


# =========================================================
# Unified HV calculation
# 所有算法使用统一 ideal / nadir / reference point 重新计算 HV
# =========================================================

OBJ_KEYS = ["profit", "load", "attitude", "quality"]


def obj_list_to_matrix(obj_list, obj_keys=OBJ_KEYS):
    """
    将目标字典列表转为 numpy 矩阵。
    默认目标顺序：
        profit   = -total_profit
        load     = load_balance
        attitude = attitude_manoeuvre
        quality  = -image_quality

    所有目标默认都是最小化形式。
    """
    if obj_list is None or len(obj_list) == 0:
        return np.empty((0, len(obj_keys)))

    return np.array(
        [[obj[k] for k in obj_keys] for obj in obj_list],
        dtype=float
    )


def compute_global_bounds(all_obj_lists, obj_keys=OBJ_KEYS):
    """
    根据所有算法的最终目标值统一计算 ideal / nadir。

    Parameters
    ----------
    all_obj_lists : list
        形式如：
        [
            final_obj_mode_sdas,
            final_obj_mode_sdas_pic,
            final_obj_mode_sdas_dpic,
            final_obj_nsga2
        ]

    Returns
    -------
    ideal : np.ndarray
    nadir : np.ndarray
    denom : np.ndarray
    """
    all_F = []

    for obj_list in all_obj_lists:
        if obj_list is None or len(obj_list) == 0:
            continue

        F = obj_list_to_matrix(obj_list, obj_keys)
        if F.size > 0:
            all_F.append(F)

    if not all_F:
        raise ValueError("No objective values found for unified HV calculation.")

    all_F = np.vstack(all_F)

    ideal = np.min(all_F, axis=0)
    nadir = np.max(all_F, axis=0)

    denom = nadir - ideal
    denom[denom < 1e-12] = 1.0

    return ideal, nadir, denom


def compute_unified_hv(obj_list, ideal, denom, obj_keys=OBJ_KEYS, ref_point=None):
    """
    使用统一 ideal / nadir 归一化后计算 HV。

    Parameters
    ----------
    obj_list : list[dict]
        某一个算法的最终目标值列表。
    ideal : np.ndarray
        所有算法统一 ideal point。
    denom : np.ndarray
        nadir - ideal。
    ref_point : np.ndarray or None
        HV reference point。默认使用 1.1。

    Returns
    -------
    hv_value : float
    """
    if obj_list is None or len(obj_list) == 0:
        return 0.0

    fronts = non_dominated_sort(obj_list)
    if not fronts or len(fronts[0]) == 0:
        return 0.0

    nd_obj = [obj_list[i] for i in fronts[0]]
    F = obj_list_to_matrix(nd_obj, obj_keys)

    F_norm = (F - ideal) / denom

    # 防止极少数点因为统一边界外推导致异常
    F_norm = np.clip(F_norm, 0.0, 1.2)

    if ref_point is None:
        ref_point = np.ones(F_norm.shape[1]) * 1.1

    hv = Hypervolume(ref_point=ref_point)
    return float(hv(F_norm))