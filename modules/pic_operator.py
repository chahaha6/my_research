import numpy as np
import math
from .utils_moea import non_dominated_sort

def pic_repair(state, V_dec, Offspring_obj, offspring_sub, gen, max_gen):
    """
    PICEO 卷积修复算子（推动距离驱动）
    
    参数
    ----
    state : object
        算法状态对象，需包含以下属性：
        - r1, r2, S : 当前操作区参数
        - sigma, alpha, gamma : 算子参数
        - max_gen : 总代数
        - max_options : 每个决策维度的最大选项数 (list)
        - zideal : 当前理想点 (numpy array)
        - pop_obj : 当前种群目标值列表 (list of dict)
        - associate : 种群个体所属子空间索引 (numpy array)
    V_dec : np.ndarray (N_off, D)
        子代决策变量
    Offspring_obj : list of dict
        子代的目标值（最小化版本）
    offspring_sub : list of int
        每个子代所在的子空间索引
    gen : int
        当前代数（0-based）
    max_gen : int
        最大代数
        
    返回
    ----
    U : np.ndarray
        修复后的决策变量
    r1, r2, S : float
        更新后的操作区参数
    """
    N_off = V_dec.shape[0]
    D = V_dec.shape[1]
    if N_off < 3:
        return V_dec, state.r1, state.r2, state.S

    # 约束违反（暂为0，无约束）
    CV = np.zeros(N_off)
    # τ排序：约束违反排名 + 非支配前沿排名
    idx_cv = np.argsort(CV)
    tau1 = np.zeros(N_off, dtype=int)
    tau1[idx_cv] = np.arange(1, N_off + 1)
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
    s_i = state.S + state.sigma * np.random.randn(N_off)
    U = sorted_V.copy()

    N1 = int(np.floor(state.r1 * (N_off - 2)) + 1)
    N2 = int(np.floor(state.r2 * (N_off - 1)) + 1)
    N1 = max(1, min(N_off, N1))
    N2 = max(N1, min(N_off, N2))

    # 卷积操作
    for i in range(N_off):
        if i == 0:
            continue
        if i < N1:
            op = 1   # 锐化
        elif i < N2:
            op = 2   # 钝化
        else:
            op = 3   # 打开

        k1 = i - 1
        k2 = i + 1 if i + 1 < N_off else i
        for j in range(D):
            if np.random.rand() < state.alpha:
                if op == 1:      # 锐化
                    delta = s_i[i] * (V_work[k1, j] - V_work[k2, j])
                    U[i, j] = V_work[i, j] + delta
                elif op == 2:    # 钝化
                    mV = (V_work[i, j] + V_work[k1, j] + V_work[k2, j]) / 3.0
                    delta = s_i[i] * (mV - V_work[i, j])
                    U[i, j] = V_work[i, j] + delta
                else:            # 打开
                    minV = min(V_work[i, j], V_work[k1, j], V_work[k2, j])
                    tmp = V_work[i, j] + s_i[i] * (minV - V_work[i, j])
                    maxV = max(V_work[i, j], V_work[k1, j], V_work[k2, j])
                    U[i, j] = tmp + s_i[i] * (maxV - tmp)
                V_work[i, j] = U[i, j]
                # 离散化到合法范围
                U[i, j] = int(np.clip(np.round(U[i, j]), 0, state.max_options[j] - 1))
                V_work[i, j] = U[i, j]

    # 恢复原始顺序
    inv_sort = np.argsort(sort_idx)
    U = U[inv_sort]

    # ----- 计算各区平均推动距离，用于更新 r1,r2,S -----
    push_all = np.zeros(N_off)
    for i in range(N_off):
        orig_idx = sort_idx[i]
        sub = offspring_sub[orig_idx] if orig_idx < len(offspring_sub) else 0
        child_obj = Offspring_obj[orig_idx]
        sub_pop_idx = [k for k in range(len(state.pop_obj)) if state.associate[k] == sub]
        if sub_pop_idx:
            sub_objs = _get_obj_matrix([state.pop_obj[k] for k in sub_pop_idx])
            dist_to_ideal = np.sqrt(np.sum((sub_objs - state.zideal) ** 2, axis=1))
            min_dist = np.min(dist_to_ideal)
            child_arr = _get_obj_matrix([child_obj])[0]
            child_dist = np.sqrt(np.sum((child_arr - state.zideal) ** 2))
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

    T = max_gen
    new_r1 = state.r1
    new_r2 = state.r2
    if c1 >= c2 and c1 >= c3:
        new_r1 = min(1, state.r1 + 1.0 / T)
    elif c2 >= c1 and c2 >= c3:
        new_r2 = min(1, state.r2 + 1.0 / T)
    elif c3 >= c1 and c3 >= c2:
        new_r2 = max(state.r1, state.r2 - 1.0 / T)
    new_r1 = np.clip(new_r1, 0, 1)
    new_r2 = np.clip(new_r2, new_r1, 1)

    beta = gen / max_gen if max_gen > 0 else 0
    S1 = 1 - 1 / (1 + math.exp(10 - 20 * beta))
    new_S = state.S
    if total_push > 0 and np.sum(push_all) > 0:
        S2 = np.sum(push_all * s_i) / np.sum(push_all)
        new_S = state.gamma * S1 + (1 - state.gamma) * S2
    else:
        new_S = S1

    return U, new_r1, new_r2, new_S


def _get_obj_matrix(objs):
    """将目标字典列表转为 numpy 矩阵（小工具，避免重复定义）"""
    return np.array([[d['profit'], d['load'], d['attitude'], d['quality']] for d in objs])