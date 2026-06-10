"""
只读：减水剂类型 / 减水率 与强度标签的共现与可学习性摘要（不改 CSV、不改标签）。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import pandas as pd

from src.lab_strength_residual.lab_mix_features import (
    LAB_MIX_EXTRA_FEATURE_NAMES,
    ensure_lab_mix_extra_columns_in_dataframe,
    lab_mix_extra_row_vector,
)


def _finite_y(row: pd.Series) -> tuple[float | None, float | None]:
    yc = yf = None
    if "compressive_true" in row.index and pd.notna(row["compressive_true"]):
        try:
            v = float(row["compressive_true"])
            if math.isfinite(v):
                yc = v
        except (TypeError, ValueError):
            pass
    if "flexural_true" in row.index and pd.notna(row["flexural_true"]):
        try:
            v = float(row["flexural_true"])
            if math.isfinite(v):
                yf = v
        except (TypeError, ValueError):
            pass
    return yc, yf


def _summarize_values(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    n = len(vals)
    mean = sum(vals) / n
    if n == 1:
        return {"n": n, "mean": mean, "std": None, "min": vals[0], "max": vals[0]}
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    std = math.sqrt(var) if var >= 0 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": min(vals),
        "max": max(vals),
    }


def _type_bucket(te: float, tm: float) -> str:
    if tm >= 0.5:
        return "unknown"
    if abs(te - 4.0) < 1e-9:
        return "none"
    if te in (0.0, 1.0, 2.0, 3.0):
        return "known"
    return "unknown"


def _rate_bucket(r: float, rm: float) -> str:
    if rm >= 0.5:
        return "unknown"
    if abs(r) < 1e-12:
        return "zero"
    return "known_nonzero"


def build_water_reducer_learnability_report(df: pd.DataFrame) -> dict[str, Any]:
    """
    基于与训练一致的 lab_mix_extra_row_vector 语义做统计；不修改 df 磁盘内容。
    """
    df_work = df
    if any(c not in df.columns for c in LAB_MIX_EXTRA_FEATURE_NAMES):
        df_work = ensure_lab_mix_extra_columns_in_dataframe(df)

    n = len(df_work)
    n_type_none = n_type_unknown = n_type_known = 0
    n_rate_known_total = n_rate_unknown = n_rate_zero = 0
    n_rate_known_nonzero = 0

    type_encs_known: list[float] = []
    rates_known_nonzero: list[float] = []

    by_sg: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n_rows_water_reducer_type_none": 0,
            "n_rows_water_reducer_type_unknown": 0,
            "n_rows_water_reducer_type_known": 0,
            "n_rows_water_reduction_rate_known": 0,
            "n_rows_water_reduction_rate_unknown": 0,
            "n_rows_water_reduction_rate_zero": 0,
            "n_rows_water_reduction_rate_known_nonzero": 0,
        }
    )

    yc_by_type: dict[str, list[float]] = defaultdict(list)
    yf_by_type: dict[str, list[float]] = defaultdict(list)
    yc_by_rate: dict[str, list[float]] = defaultdict(list)
    yf_by_rate: dict[str, list[float]] = defaultdict(list)

    for i in range(n):
        row = df_work.iloc[i]
        vec = lab_mix_extra_row_vector(row)
        te, tm, r, rm, _a, _am = vec
        tb = _type_bucket(te, tm)
        rb = _rate_bucket(r, rm)

        if tb == "none":
            n_type_none += 1
        elif tb == "known":
            n_type_known += 1
            type_encs_known.append(float(te))
        else:
            n_type_unknown += 1

        if rm >= 0.5:
            n_rate_unknown += 1
        else:
            n_rate_known_total += 1
            if abs(r) < 1e-12:
                n_rate_zero += 1
            else:
                n_rate_known_nonzero += 1
                rates_known_nonzero.append(float(r))

        sg = ""
        if "source_group" in row.index and pd.notna(row["source_group"]):
            sg = str(row["source_group"]).strip()
        b = by_sg[sg]
        if tb == "none":
            b["n_rows_water_reducer_type_none"] += 1
        elif tb == "known":
            b["n_rows_water_reducer_type_known"] += 1
        else:
            b["n_rows_water_reducer_type_unknown"] += 1
        if rm >= 0.5:
            b["n_rows_water_reduction_rate_unknown"] += 1
        else:
            b["n_rows_water_reduction_rate_known"] += 1
            if abs(r) < 1e-12:
                b["n_rows_water_reduction_rate_zero"] += 1
            else:
                b["n_rows_water_reduction_rate_known_nonzero"] += 1

        yc, yf = _finite_y(row)
        if yc is not None:
            yc_by_type[tb].append(yc)
            yc_by_rate[rb].append(yc)
        if yf is not None:
            yf_by_type[tb].append(yf)
            yf_by_rate[rb].append(yf)

    n_distinct_type_enc = len({round(x, 6) for x in type_encs_known})
    n_distinct_rate_positive = len({round(x, 6) for x in rates_known_nonzero})

    type_learnable = n_type_known >= 3 and n_distinct_type_enc >= 2
    rate_learnable_nonzero = (
        n_rate_known_nonzero >= 3 and n_distinct_rate_positive >= 2
    )
    rate_learnable_any_variation = n_rate_known_total >= 3 and (
        n_rate_known_nonzero >= 2
        or (n_rate_zero > 0 and n_rate_known_nonzero > 0)
    )

    frac_type_unknown = n_type_unknown / n if n else 0.0
    frac_rate_unknown = n_rate_unknown / n if n else 0.0

    if frac_type_unknown >= 0.6 and frac_rate_unknown >= 0.6:
        dominant = "missing_pattern"
    elif frac_type_unknown < 0.35 and frac_rate_unknown < 0.35:
        dominant = "mostly_observed_features"
    else:
        dominant = "mixed_missing_and_observed"

    more_missing_than_signal = (
        not type_learnable
        and not rate_learnable_nonzero
        and (frac_type_unknown >= 0.5 or frac_rate_unknown >= 0.5)
    )

    if n_rate_known_nonzero == 0 and n_rate_zero > 0:
        rate_note_extra = (
            " 已知减水率行均为 0%（多为无外加剂闭合），无正减水率取值跨度。"
        )
    else:
        rate_note_extra = ""

    if more_missing_than_signal:
        evm = (
            "当前更像在学习 missing pattern（占位/标志）与分组常数，"
            "而非减水剂类型或可辨识减水率水平的真实物理效应。"
        )
    elif rate_learnable_nonzero:
        evm = (
            "减水率存在多档非零观测，具备有限条件下学习「率—强度」梯度的可能，"
            "但仍受样本量、分组泄漏与混杂制约，不宜过度解读。"
        )
    elif type_learnable:
        evm = "类型维度存在一定变异，可尝试学习品种间差异，但需独立验证泛化。"
    else:
        evm = (
            "类型或（非零）减水率信息不足，信号以缺失与协变量为主；"
            "与 missing pattern 的区分度有限。"
        )

    conclusions: dict[str, Any] = {
        "type_dimension_learnable": bool(type_learnable),
        "type_dimension_note_zh": (
            f"类型已知行数={n_type_known}，已知类内不同 enc 数={n_distinct_type_enc}。"
            "可学习性判据：至少 3 行类型已知且至少 2 种 enc。"
            if n_type_known > 0
            else "当前无「类型已知」行，无法在类型维度拟合减水剂品种效应。"
        ),
        "rate_dimension_learnable_nonzero_variation": bool(rate_learnable_nonzero),
        "rate_dimension_note_zh": (
            (
                f"减水率「已知且>0」行数={n_rate_known_nonzero}，不同取值数={n_distinct_rate_positive}。"
                "可学习性判据（非零率效应）：至少 3 行且至少 2 个不同正减水率。"
            )
            + rate_note_extra
            if n_rate_known_nonzero > 0
            else (
                "无「已知且非零」减水率样本，无法从率的分档学习真实减水率—强度梯度。"
                + rate_note_extra
            )
        ),
        "rate_dimension_any_cross_bucket_signal": bool(rate_learnable_any_variation),
        "dominant_pattern": dominant,
        "more_missing_than_mechanistic_signal": bool(more_missing_than_signal),
        "effect_vs_missing_pattern_zh": evm,
        "honest_summary_zh": (
            "类型与（非零）减水率均缺乏交叉变异：模型更易利用 missing flag / 常数段，"
            "而非可解释的减水剂化学—率效应。"
            if more_missing_than_signal
            else (
                "部分标签与减水剂字段可同时观测，但仍需结合样本量与分组方差谨慎解读。"
                if dominant != "missing_pattern"
                else "缺失占位在样本中占主导，对效应学习不友好。"
            )
        ),
    }

    y_overview = {
        "by_water_reducer_type_bucket": {
            k: {
                "compressive_true": _summarize_values(yc_by_type[k]),
                "flexural_true": _summarize_values(yf_by_type[k]),
            }
            for k in ("none", "unknown", "known")
        },
        "by_water_reduction_rate_bucket": {
            k: {
                "compressive_true": _summarize_values(yc_by_rate[k]),
                "flexural_true": _summarize_values(yf_by_rate[k]),
            }
            for k in ("unknown", "zero", "known_nonzero")
        },
    }

    return {
        "n_rows_analyzed": n,
        "n_rows_water_reducer_type_none": n_type_none,
        "n_rows_water_reducer_type_unknown": n_type_unknown,
        "n_rows_water_reducer_type_known": n_type_known,
        "n_rows_water_reduction_rate_known": n_rate_known_total,
        "n_rows_water_reduction_rate_unknown": n_rate_unknown,
        "n_rows_water_reduction_rate_zero": n_rate_zero,
        "n_rows_water_reduction_rate_known_nonzero": n_rate_known_nonzero,
        "n_distinct_water_reducer_type_enc_among_known": n_distinct_type_enc,
        "n_distinct_water_reduction_rate_pct_among_known_nonzero": n_distinct_rate_positive,
        "water_reducer_learnability_by_source_group": {
            k: dict(v) for k, v in sorted(by_sg.items())
        },
        "y_true_distribution_overview": y_overview,
        "conclusions": conclusions,
    }
