import numpy as np
import math
from .utils_moea import non_dominated_sort


def pic_repair(state, V_dec, Offspring_obj, offspring_sub, gen, max_gen):
    """
    PICEO 卷积修复算子，适配 2D 编码。

    新编码结构：
        前 state.D_task 维：
            任务选择基因
            0 = 不执行
            1~k = 选择第 k 个 VTW

        后 state.D_task 维：
            OTW 位置 ratio
            ratio ∈ [0,1]

    修复策略：
        1. 前半部分按离散任务选择修复到 [0, k]
        2. 后半部分按连续 ratio 修复到 [0, 1]
        3. 排序、卷积、推动距离更新逻辑保持原 PICEO 思路
    """

    N_off = V_dec.shape[0]
    D = V_dec.shape[1]

    if N_off < 3:
        return V_dec, state.r1, state.r2, state.S

    # ---------- 约束违反度 ----------
    # 如果算法类有 constraint_violation，可以使用真实约束违反度
    CV = np.zeros(N_off)

    if hasattr(state, "constraint_violation"):
        for i in range(N_off):
            try:
                CV[i] = state.constraint_violation(V_dec[i])
            except Exception:
                CV[i] = 0.0

    # ---------- tau 排序：约束违反排名 + 非支配前沿排名 ----------
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
    U = sorted_V.copy()

    s_i = state.S + state.sigma * np.random.randn(N_off)

    N1 = int(np.floor(state.r1 * (N_off - 2)) + 1)
    N2 = int(np.floor(state.r2 * (N_off - 1)) + 1)

    N1 = max(1, min(N_off, N1))
    N2 = max(N1, min(N_off, N2))

    D_task = getattr(state, "D_task", D)

    # =========================================================
    # 卷积操作
    # =========================================================
    for i in range(N_off):
        if i == 0:
            continue

        if i < N1:
            op = 1      # 锐化
        elif i < N2:
            op = 2      # 钝化
        else:
            op = 3      # 打开

        k1 = i - 1
        k2 = i + 1 if i + 1 < N_off else i

        for j in range(D):
            if np.random.rand() >= state.alpha:
                continue

            # ---------- 原 PICEO 三类卷积 ----------
            if op == 1:
                delta = s_i[i] * (V_work[k1, j] - V_work[k2, j])
                U[i, j] = V_work[i, j] + delta

            elif op == 2:
                mV = (V_work[i, j] + V_work[k1, j] + V_work[k2, j]) / 3.0
                delta = s_i[i] * (mV - V_work[i, j])
                U[i, j] = V_work[i, j] + delta

            else:
                minV = min(V_work[i, j], V_work[k1, j], V_work[k2, j])
                maxV = max(V_work[i, j], V_work[k1, j], V_work[k2, j])

                tmp = V_work[i, j] + s_i[i] * (minV - V_work[i, j])
                U[i, j] = tmp + s_i[i] * (maxV - tmp)

            # =================================================
            # 2D 编码边界修复
            # =================================================

            if j < D_task:
                # 前半部分：任务选择基因，离散整数
                # 取值范围：0~k
                # 0 表示不执行，1~k 表示选择窗口
                k = state.max_options[j]

                if k <= 0:
                    U[i, j] = 0
                else:
                    U[i, j] = int(np.clip(np.round(U[i, j]), 0, k))

            else:
                # 后半部分：ratio 基因，连续变量
                # 取值范围：[0,1]
                U[i, j] = float(np.clip(U[i, j], 0.0, 1.0))

            V_work[i, j] = U[i, j]

    # ---------- 恢复原始顺序 ----------
    inv_sort = np.argsort(sort_idx)
    U = U[inv_sort]

    # =========================================================
    # 计算各区平均推动距离，用于更新 r1, r2, S
    # =========================================================

    push_all = np.zeros(N_off)

    for i in range(N_off):
        orig_idx = sort_idx[i]

        sub = offspring_sub[orig_idx] if orig_idx < len(offspring_sub) else 0

        if sub < 0 or sub >= len(state.W):
            continue

        child_obj = Offspring_obj[orig_idx]

        sub_pop_idx = [
            k for k in range(len(state.pop_obj))
            if state.associate[k] == sub
        ]

        if sub_pop_idx:
            sub_objs = _get_obj_matrix([
                state.pop_obj[k] for k in sub_pop_idx
            ])

            dist_to_ideal = np.sqrt(
                np.sum((sub_objs - state.zideal) ** 2, axis=1)
            )

            min_dist = np.min(dist_to_ideal)

            child_arr = _get_obj_matrix([child_obj])[0]
            child_dist = np.sqrt(
                np.sum((child_arr - state.zideal) ** 2)
            )

            push = min_dist - child_dist
            push_all[i] = max(0.0, push)

    c1 = np.mean(push_all[:N1]) if N1 >= 1 else 0.0
    c2 = np.mean(push_all[N1:N2]) if N2 > N1 else 0.0
    c3 = np.mean(push_all[N2:]) if N_off > N2 else 0.0

    total_push = c1 + c2 + c3

    if total_push > 0:
        c1 /= total_push
        c2 /= total_push
        c3 /= total_push

    T = max_gen if max_gen > 0 else 1

    new_r1 = state.r1
    new_r2 = state.r2

    if c1 >= c2 and c1 >= c3:
        new_r1 = min(1.0, state.r1 + 1.0 / T)

    elif c2 >= c1 and c2 >= c3:
        new_r2 = min(1.0, state.r2 + 1.0 / T)

    elif c3 >= c1 and c3 >= c2:
        new_r2 = max(state.r1, state.r2 - 1.0 / T)

    new_r1 = float(np.clip(new_r1, 0.0, 1.0))
    new_r2 = float(np.clip(new_r2, new_r1, 1.0))

    beta = gen / max_gen if max_gen > 0 else 0.0

    S1 = 1.0 - 1.0 / (1.0 + math.exp(10.0 - 20.0 * beta))

    if total_push > 0 and np.sum(push_all) > 0:
        S2 = np.sum(push_all * s_i) / np.sum(push_all)
        new_S = state.gamma * S1 + (1.0 - state.gamma) * S2
    else:
        new_S = S1

    return U, new_r1, new_r2, new_S


def _get_obj_matrix(objs):
    """
    将目标字典列表转为 numpy 矩阵。
    目标均按最小化形式：
        profit  = -total_profit
        load    = load_balance
        attitude= attitude_manoeuvre
        quality = -image_quality
    """

    return np.array([
        [d["profit"], d["load"], d["attitude"], d["quality"]]
        for d in objs
    ])