import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from modules.utils_moea import (
    OBJ_KEYS,
    compute_global_bounds,
    compute_unified_hv,
    non_dominated_sort,
    obj_list_to_matrix,
)


def load_final_objs(case_name):
    """读取某个数据集下所有 final_objs_*.csv。"""
    result_dir = Path("results") / case_name
    files = sorted(result_dir.glob("final_objs_*.csv"))
    if not files:
        raise FileNotFoundError(f"No final_objs_*.csv found in {result_dir}")

    data = {}
    for path in files:
        algorithm = path.stem.replace("final_objs_", "", 1)
        df = pd.read_csv(path)
        missing = [key for key in OBJ_KEYS if key not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        obj_list = df[OBJ_KEYS].astype(float).to_dict("records")
        data[algorithm] = obj_list
    return data


def get_non_dominated_objs(obj_list):
    """返回目标值列表中的第一非支配层。所有目标均按越小越好处理。"""
    if not obj_list:
        return []
    fronts = non_dominated_sort(obj_list)
    if not fronts:
        return []
    return [obj_list[i] for i in fronts[0]]


def normalize_matrix(obj_list, ideal, denom):
    """使用统一 ideal/nadir 归一化目标矩阵。"""
    if not obj_list:
        return np.empty((0, len(OBJ_KEYS)))
    return (obj_list_to_matrix(obj_list, OBJ_KEYS) - ideal) / denom


def nearest_distances(source, target):
    """计算 source 中每个点到 target 点集的最近欧氏距离。"""
    if len(source) == 0 or len(target) == 0:
        return np.array([], dtype=float)
    diff = source[:, None, :] - target[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return np.min(dist, axis=1)


def compute_igd(algorithm_front, reference_front):
    """IGD：参考前沿到算法前沿的平均最近距离，越小越好。"""
    distances = nearest_distances(reference_front, algorithm_front)
    if len(distances) == 0:
        return float("inf")
    return float(np.mean(distances))


def compute_gd(algorithm_front, reference_front):
    """GD：算法前沿到参考前沿的均方根最近距离，越小越好。"""
    distances = nearest_distances(algorithm_front, reference_front)
    if len(distances) == 0:
        return float("inf")
    return float(np.sqrt(np.mean(distances * distances)))


def compute_spread_delta_p(front):
    """
    多目标 Spread / Delta_p 的近似版本。

    这里用归一化非支配前沿的最近邻距离变异系数衡量分布均匀性：
    数值越小，说明点间距越均匀。
    """
    if len(front) < 2:
        return 0.0

    diff = front[:, None, :] - front[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    mean_nearest = np.mean(nearest)
    if mean_nearest < 1e-12:
        return 0.0
    return float(np.std(nearest, ddof=1) / mean_nearest)


def compute_metrics(case_name):
    all_objs_by_algorithm = load_final_objs(case_name)
    all_obj_lists = list(all_objs_by_algorithm.values())
    ideal, _, denom = compute_global_bounds(all_obj_lists, OBJ_KEYS)

    combined_objs = []
    for obj_list in all_obj_lists:
        combined_objs.extend(obj_list)
    reference_objs = get_non_dominated_objs(combined_objs)
    reference_front = normalize_matrix(reference_objs, ideal, denom)

    rows = []
    for algorithm, obj_list in sorted(all_objs_by_algorithm.items()):
        nd_objs = get_non_dominated_objs(obj_list)
        algorithm_front = normalize_matrix(nd_objs, ideal, denom)

        rows.append({
            "Algorithm": algorithm,
            "Point_Count": len(obj_list),
            "ND_Size": len(nd_objs),
            "Reference_Size": len(reference_objs),
            "HV": compute_unified_hv(obj_list, ideal, denom, OBJ_KEYS),
            "IGD": compute_igd(algorithm_front, reference_front),
            "GD": compute_gd(algorithm_front, reference_front),
            "Spread_Delta_p": compute_spread_delta_p(algorithm_front),
        })

    df = pd.DataFrame(rows)
    return df.sort_values(["HV", "IGD"], ascending=[False, True])


def main():
    parser = argparse.ArgumentParser(
        description="根据 saved final_objs 生成 HV、IGD、GD、Spread/Delta_p 指标。"
    )
    parser.add_argument("--case", default="s5_t100", help="例如 s5_t100、s5_t300")
    parser.add_argument(
        "--output",
        default=None,
        help="输出 CSV 路径；默认写到 results/<case>/multi_metric_from_saved_final_objs.csv",
    )
    args = parser.parse_args()

    df = compute_metrics(args.case)
    output = Path(args.output) if args.output else (
        Path("results") / args.case / "multi_metric_from_saved_final_objs.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Saved metrics to: {output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
