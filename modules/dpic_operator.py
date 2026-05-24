"""
DPIC operator for MODE-SDAS-DPIC.

This module implements a Discrete Population Image Convolution (DPIC)
operator.  It is adapted from the PIC idea in PICEO, but is used only on the
integer task-selection / visible-time-window-selection part of a mixed
scheduling chromosome.

Expected usage from mode_sdas_dpic.py:
    task_part = offspring_dec[:, :D_task]
    ratio_part = offspring_dec[:, D_task:]
    repaired_task, r1, r2, S = dpic_repair(
        state, task_part, off_obj, offspring_sub, gen, max_gen,
        ratio_part=ratio_part,
    )

The DPIC operator itself does not touch the continuous ratio genes.  If
ratio_part is provided, it is used only for computing the true constraint
violation of the corresponding full mixed-encoding individual.
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .utils_moea import non_dominated_sort


def dpic_repair(
    state,
    V_dec: np.ndarray,
    Offspring_obj: Sequence[dict],
    offspring_sub: Sequence[int],
    gen: int,
    max_gen: int,
    ratio_part: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Discrete Population Image Convolution repair operator.

    Parameters
    ----------
    state:
        The MODE_SDAS_DPIC instance.  The operator uses these attributes:
        r1, r2, S, sigma, alpha, gamma, max_options, W, pop_obj, associate,
        zideal, and optionally constraint_violation.
    V_dec:
        Offspring decision matrix containing only the discrete task-selection
        genes.  Shape: (N_off, D_task).  Gene value 0 means not scheduled;
        gene value 1..k means selecting one candidate visible time window.
    Offspring_obj:
        Objective dictionaries of the same offspring before DPIC.  These are
        used to sort individuals and to update the internal DPIC parameters.
    offspring_sub:
        Subspace index from which each offspring was generated.
    gen, max_gen:
        Current generation and maximum generation.
    ratio_part:
        Optional continuous ratio genes with shape (N_off, D_task).  DPIC does
        not change these genes; they are used only to compute real constraint
        violation for the full mixed individual.

    Returns
    -------
    U:
        Repaired discrete task-selection genes.
    new_r1, new_r2, new_S:
        Updated DPIC/PIC control parameters.
    """

    V_dec = np.asarray(V_dec, dtype=float)
    N_off, D = V_dec.shape

    if N_off < 3 or D == 0:
        return V_dec.copy(), float(state.r1), float(state.r2), float(state.S)

    # ---------------------------------------------------------
    # 1) Constraint-violation ranking.
    #    If ratio_part is available, compute CV using the full mixed
    #    chromosome.  Otherwise, safely fall back to zeros.
    # ---------------------------------------------------------
    CV = np.zeros(N_off, dtype=float)
    if ratio_part is not None and hasattr(state, "constraint_violation"):
        ratio_part = np.asarray(ratio_part, dtype=float)
        for i in range(N_off):
            try:
                full_x = np.hstack([V_dec[i], ratio_part[i]])
                full_x = state.repair_individual(full_x)
                CV[i] = float(state.constraint_violation(full_x))
            except Exception:
                CV[i] = 0.0

    idx_cv = np.argsort(CV)
    tau_cv = np.zeros(N_off, dtype=int)
    tau_cv[idx_cv] = np.arange(1, N_off + 1)

    # ---------------------------------------------------------
    # 2) Non-dominated front ranking.
    # ---------------------------------------------------------
    front_no = np.ones(N_off, dtype=int) * (N_off + 1)
    try:
        fronts = non_dominated_sort(list(Offspring_obj))
        for f_idx, front in enumerate(fronts):
            for idx in front:
                if 0 <= idx < N_off:
                    front_no[idx] = f_idx + 1
    except Exception:
        front_no[:] = 1

    # Smaller tau means better individual and should appear earlier in the
    # population image.
    tau = tau_cv + front_no
    sort_idx = np.argsort(tau)
    inv_sort = np.argsort(sort_idx)

    V_work = V_dec[sort_idx].copy()
    U = V_work.copy()

    # ---------------------------------------------------------
    # 3) DPIC image-convolution zones.
    #    Good rows: sharpening; middle rows: smoothing/passivation;
    #    inferior rows: opening-like operation.
    # ---------------------------------------------------------
    r1 = float(np.clip(getattr(state, "r1", 0.6), 0.0, 1.0))
    r2 = float(np.clip(getattr(state, "r2", 0.85), r1, 1.0))
    alpha = float(np.clip(getattr(state, "alpha", 0.15), 0.0, 1.0))
    sigma = float(getattr(state, "sigma", 0.01))
    S = float(getattr(state, "S", 1.0))

    s_i = S + sigma * np.random.randn(N_off)

    N1 = int(np.floor(r1 * (N_off - 2)) + 1)
    N2 = int(np.floor(r2 * (N_off - 1)) + 1)
    N1 = max(1, min(N_off, N1))
    N2 = max(N1, min(N_off, N2))

    max_options = getattr(state, "max_options", None)
    if max_options is None or len(max_options) < D:
        max_options = [1] * D

    for i in range(N_off):
        if i == 0:
            # Preserve the best row in the population image.
            continue

        if i < N1:
            op = 1      # sharpening
        elif i < N2:
            op = 2      # passivation / smoothing
        else:
            op = 3      # opening-like operation

        k1 = i - 1
        k2 = i + 1 if i + 1 < N_off else i

        for j in range(D):
            if np.random.rand() >= alpha:
                continue

            if op == 1:
                # Image sharpening: strengthen the difference between
                # neighboring rows.
                delta = s_i[i] * (V_work[k1, j] - V_work[k2, j])
                U[i, j] = V_work[i, j] + delta

            elif op == 2:
                # Image passivation/smoothing: move the current row toward the
                # local average.
                mean_val = (V_work[i, j] + V_work[k1, j] + V_work[k2, j]) / 3.0
                delta = s_i[i] * (mean_val - V_work[i, j])
                U[i, j] = V_work[i, j] + delta

            else:
                # Opening-like operation: first move toward the local minimum,
                # then toward the local maximum.  This provides a larger but
                # still image-structured perturbation for inferior rows.
                min_val = min(V_work[i, j], V_work[k1, j], V_work[k2, j])
                max_val = max(V_work[i, j], V_work[k1, j], V_work[k2, j])
                tmp = V_work[i, j] + s_i[i] * (min_val - V_work[i, j])
                U[i, j] = tmp + s_i[i] * (max_val - tmp)

            # Discrete boundary repair.  Here max_options[j] is the true upper
            # bound k, not k+1.  Valid values are integers in [0, k].
            upper = int(max(0, max_options[j]))
            U[i, j] = int(np.clip(np.round(U[i, j]), 0, upper))
            V_work[i, j] = U[i, j]

    # Restore original offspring order.
    U = U[inv_sort]

    # ---------------------------------------------------------
    # 4) Update r1, r2, and S using push-distance feedback.
    #    The feedback follows the original PICEO/PIC spirit while keeping the
    #    MODE-SDAS subspace information.
    # ---------------------------------------------------------
    push_all_sorted = np.zeros(N_off, dtype=float)

    for sorted_pos in range(N_off):
        orig_idx = int(sort_idx[sorted_pos])
        if orig_idx >= len(Offspring_obj):
            continue

        sub = int(offspring_sub[orig_idx]) if orig_idx < len(offspring_sub) else 0
        if not hasattr(state, "W") or state.W is None or sub < 0 or sub >= len(state.W):
            continue
        if not hasattr(state, "pop_obj") or state.pop_obj is None:
            continue
        if not hasattr(state, "associate") or state.associate is None:
            continue
        if not hasattr(state, "zideal") or state.zideal is None:
            continue

        sub_pop_idx = [
            k for k in range(len(state.pop_obj))
            if 0 <= k < len(state.associate) and state.associate[k] == sub
        ]
        if not sub_pop_idx:
            continue

        try:
            sub_objs = _get_obj_matrix([state.pop_obj[k] for k in sub_pop_idx])
            dist_to_ideal = np.sqrt(np.sum((sub_objs - state.zideal) ** 2, axis=1))
            min_dist = float(np.min(dist_to_ideal))

            child_arr = _get_obj_matrix([Offspring_obj[orig_idx]])[0]
            child_dist = float(np.sqrt(np.sum((child_arr - state.zideal) ** 2)))
            push_all_sorted[sorted_pos] = max(0.0, min_dist - child_dist)
        except Exception:
            push_all_sorted[sorted_pos] = 0.0

    c1 = float(np.mean(push_all_sorted[:N1])) if N1 >= 1 else 0.0
    c2 = float(np.mean(push_all_sorted[N1:N2])) if N2 > N1 else 0.0
    c3 = float(np.mean(push_all_sorted[N2:])) if N_off > N2 else 0.0
    total_push = c1 + c2 + c3

    if total_push > 1e-12:
        c1 /= total_push
        c2 /= total_push
        c3 /= total_push

    T = max(1, int(max_gen))
    new_r1 = r1
    new_r2 = r2

    if total_push > 1e-12:
        if c1 >= c2 and c1 >= c3:
            new_r1 = min(1.0, r1 + 1.0 / T)
        elif c2 >= c1 and c2 >= c3:
            new_r2 = min(1.0, r2 + 1.0 / T)
        else:
            new_r2 = max(new_r1, r2 - 1.0 / T)

    new_r1 = float(np.clip(new_r1, 0.0, 1.0))
    new_r2 = float(np.clip(new_r2, new_r1, 1.0))

    beta = float(gen) / float(max_gen) if max_gen > 0 else 0.0
    # Logistic time-varying component.  It gradually changes the perturbation
    # scale along the evolution.
    S1 = 1.0 - 1.0 / (1.0 + math.exp(10.0 - 20.0 * beta))

    if total_push > 1e-12 and np.sum(push_all_sorted) > 1e-12:
        # Align the successful local scale with the push-distance contribution.
        S2 = float(np.sum(push_all_sorted * s_i) / np.sum(push_all_sorted))
        gamma = float(np.clip(getattr(state, "gamma", 0.7), 0.0, 1.0))
        new_S = gamma * S1 + (1.0 - gamma) * S2
    else:
        new_S = S1

    return U.astype(float), new_r1, new_r2, float(new_S)


# Backward-compatible alias.  This lets older imports continue to work if
# needed, but new code should import dpic_repair directly.
pic_repair = dpic_repair


def _get_obj_matrix(objs: Sequence[dict]) -> np.ndarray:
    """
    Convert objective dictionaries to a matrix.

    All objectives are assumed to be in minimization form:
        profit   = -total_profit
        load     = load_balance
        attitude = attitude_manoeuvre
        quality  = -image_quality
    """

    return np.array(
        [[d["profit"], d["load"], d["attitude"], d["quality"]] for d in objs],
        dtype=float,
    )
