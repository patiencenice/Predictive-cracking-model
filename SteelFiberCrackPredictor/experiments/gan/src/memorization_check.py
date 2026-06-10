from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _numeric_feature_frame(
    real_df: pd.DataFrame, syn_df: pd.DataFrame, feature_cols: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if feature_cols is None:
        common = [c for c in real_df.columns if c in syn_df.columns]
    else:
        common = [c for c in feature_cols if c in real_df.columns and c in syn_df.columns]
    cols: list[str] = []
    for c in common:
        rr = pd.to_numeric(real_df[c], errors="coerce")
        ss = pd.to_numeric(syn_df[c], errors="coerce")
        if rr.notna().any() and ss.notna().any():
            cols.append(c)
    if not cols:
        raise ValueError("No shared numeric columns available for memorization check.")

    r = real_df[cols].apply(pd.to_numeric, errors="coerce")
    s = syn_df[cols].apply(pd.to_numeric, errors="coerce")
    r = r.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").reset_index(drop=True)
    s = s.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").reset_index(drop=True)
    if r.empty or s.empty:
        raise ValueError("No valid numeric rows after dropping NaN/Inf for memorization check.")
    return r, s, cols


def _nearest_distances(train_x: np.ndarray, query_x: np.ndarray, k: int = 1) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(train_x)
    d, _ = nn.kneighbors(query_x)
    return d[:, k - 1]


def run_memorization_check(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    必做项：
    - synthetic vs real 最近邻距离
    - synthetic 内部重复/近重复
    """
    r, s, used_cols = _numeric_feature_frame(real_df, synthetic_df, feature_cols)
    r_x = r.to_numpy(dtype=np.float64)
    s_x = s.to_numpy(dtype=np.float64)

    # real-real 最近邻（排除自身）作为数据尺度参考
    rr_2 = _nearest_distances(r_x, r_x, k=2)
    rr_ref = {
        "q01": float(np.quantile(rr_2, 0.01)),
        "q05": float(np.quantile(rr_2, 0.05)),
        "q10": float(np.quantile(rr_2, 0.10)),
        "median": float(np.quantile(rr_2, 0.50)),
    }

    # synthetic 到 real 的最近邻距离
    sr_1 = _nearest_distances(r_x, s_x, k=1)
    near_copy_thr = rr_ref["q05"]
    suspicious_mask = sr_1 <= near_copy_thr
    n_susp = int(suspicious_mask.sum())
    frac_susp = float(n_susp / len(sr_1))

    # synthetic 内部重复/近重复（排除自身）
    ss_2 = _nearest_distances(s_x, s_x, k=2)
    near_dup_thr = rr_ref["q05"]
    near_dup_mask = ss_2 <= near_dup_thr
    n_near_dup = int(near_dup_mask.sum())
    frac_near_dup = float(n_near_dup / len(ss_2))

    # 精确重复（按数值列 round 后判重）
    s_round = s.round(8)
    n_exact_dup = int(s_round.duplicated(keep=False).sum())
    frac_exact_dup = float(n_exact_dup / len(s_round))

    # 风险分级（保守）
    if frac_susp >= 0.15 or frac_near_dup >= 0.25 or frac_exact_dup >= 0.10:
        risk = "high"
    elif frac_susp >= 0.05 or frac_near_dup >= 0.10 or frac_exact_dup >= 0.03:
        risk = "medium"
    else:
        risk = "low"

    if risk == "high":
        verdict = "存在明显记忆风险：synthetic 可能包含近复制真实样本或内部模式塌缩。"
    elif risk == "medium":
        verdict = "存在一定记忆风险：需在进入对照实验前进一步收紧生成约束。"
    else:
        verdict = "未见显著记忆风险，但仍需结合分布偏移与标签耦合检查综合判断。"

    return {
        "used_numeric_columns": used_cols,
        "n_real_checked": int(len(r)),
        "n_synthetic_checked": int(len(s)),
        "real_real_nn_distance_reference": rr_ref,
        "synthetic_to_real_nn_distance": {
            "min": float(sr_1.min()),
            "q01": float(np.quantile(sr_1, 0.01)),
            "q05": float(np.quantile(sr_1, 0.05)),
            "q50": float(np.quantile(sr_1, 0.50)),
            "mean": float(sr_1.mean()),
            "max": float(sr_1.max()),
        },
        "suspicious_near_copy": {
            "threshold": float(near_copy_thr),
            "count": n_susp,
            "ratio": frac_susp,
        },
        "synthetic_internal_repeat": {
            "near_duplicate_threshold": float(near_dup_thr),
            "near_duplicate_count": n_near_dup,
            "near_duplicate_ratio": frac_near_dup,
            "exact_duplicate_count": n_exact_dup,
            "exact_duplicate_ratio": frac_exact_dup,
        },
        "memorization_risk": risk,
        "honest_conclusion_zh": verdict,
    }
