"""
lab_strength_residual 训练用 CSV 闸门：列完整性、数值、分组、重复、正值。

减水剂扩展六列（water_reducer_type_enc 等）在 LAB_STRENGTH_FEATURE_COLUMNS 中为硬必需；
若原始文献/合并表缺列，应由 merge_lab_strength_training_csv 或 scripts/prepare_lab_strength_missing_columns
在闸门前写入与 lab_mix_extra_row_vector 一致的占位与 missing_flag（不得臆造真实减水率）。

可选列 needs_manual_review（0/1）：为 1 时该行保留在 CSV，但 build_xy_matrices 不进入公式基线训练；
典型情形见仓库约定（fcu 语义未闭合、试件/加载协议列缺失或与真值口径不一致、原文歧义等）。
可选列 manual_review_note：非空时写入报告 manual_review_samples 的 note。

基准组（fiber_content、aspect_ratio、tensile_strength 同时为 0）：
- 不参与 fiber_content / aspect_ratio 的「参考范围」warning 统计（仍检查可解析性）。
- 不放宽 compressive_true / flexural_true 正值与硬闸门。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.lab_strength_residual.lab_mix_features import (
    LAB_STRENGTH_FEATURE_COLUMNS,
    summarize_water_reducer_features,
)
from src.lab_strength_residual.provenance_columns import summarize_tracing_columns

TRAINING_LABEL_AND_GROUP = ("compressive_true", "flexural_true", "source_group")

# 与 check_lab_strength_training_csv 一致的参考范围（仅 warning）
REFERENCE_RANGES: dict[str, tuple[float, float]] = {
    "fiber_content": (0.5, 3.0),
    "aspect_ratio": (30.0, 100.0),
    "w_b_ratio": (0.3, 0.5),
}


def baseline_zero_mask(df: pd.DataFrame) -> pd.Series:
    """基准组：三字段同时为 0（可解析为数值时）。"""
    need = ("fiber_content", "aspect_ratio", "tensile_strength")
    if not all(c in df.columns for c in need):
        return pd.Series(False, index=df.index)
    fc = pd.to_numeric(df["fiber_content"], errors="coerce")
    ar = pd.to_numeric(df["aspect_ratio"], errors="coerce")
    ts = pd.to_numeric(df["tensile_strength"], errors="coerce")
    return (fc.fillna(-1) == 0) & (ar.fillna(-1) == 0) & (ts.fillna(-1) == 0)


def _missing_columns(df: pd.DataFrame) -> list[str]:
    need = list(LAB_STRENGTH_FEATURE_COLUMNS) + list(TRAINING_LABEL_AND_GROUP)
    return [c for c in need if c not in df.columns]


def _coerce_issues(df: pd.DataFrame, cols: list[str]) -> list[str]:
    issues: list[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        if c == "source_group":
            continue
        s = df[c]
        out = pd.to_numeric(s, errors="coerce")
        bad = s.notna() & out.isna()
        if bad.any():
            idx = bad.index[bad].tolist()[:15]
            issues.append(f"列 {c}: 存在非空但不可解析为数值的行 index={list(map(int, idx))}")
    return issues


def _positive_strength_issues(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    for c in ("compressive_true", "flexural_true"):
        if c not in df.columns:
            issues.append(f"缺少标签列 {c}")
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        ok = v.dropna()
        if (ok <= 0).any():
            issues.append(
                f"{c}: 存在非正数值（>0 条数）={(ok <= 0).sum()}，示例 index={ok.index[ok <= 0].tolist()[:10]}"
            )
    return issues


def _duplicate_issues(df: pd.DataFrame) -> list[str]:
    sub = [
        c
        for c in list(LAB_STRENGTH_FEATURE_COLUMNS) + list(TRAINING_LABEL_AND_GROUP)
        if c in df.columns
    ]
    if len(sub) < 3:
        return ["判重列过少，无法可靠检查重复"]
    dup = df.duplicated(subset=sub, keep=False)
    if dup.any():
        return [
            f"在 {len(sub)} 列组合上存在重复样本: {int(dup.sum())} 行，示例 index={df.index[dup].tolist()[:20]}"
        ]
    return []


def _source_group_issues(df: pd.DataFrame) -> list[str]:
    if "source_group" not in df.columns:
        return ["缺少 source_group"]
    sg = df["source_group"].astype(str).replace("nan", "NA")
    if sg.nunique(dropna=False) < 2:
        return [f"source_group 唯一值不足 2（当前 {int(sg.nunique(dropna=False))}）"]
    return []


def _reference_range_warnings(df: pd.DataFrame) -> dict[str, Any]:
    """非 hard fail；基准组不参与 fiber_content/aspect_ratio 越界计数。"""
    base = baseline_zero_mask(df)
    out: dict[str, Any] = {}
    for col, (lo, hi) in REFERENCE_RANGES.items():
        if col not in df.columns:
            out[col] = {"skipped": True}
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        valid = v.notna()
        if col in ("fiber_content", "aspect_ratio"):
            check_rows = valid & ~base
        else:
            check_rows = valid
        sub = v[check_rows]
        below = sub < lo
        above = sub > hi
        outside = below | above
        out[col] = {
            "reference_min": lo,
            "reference_max": hi,
            "n_outside_excluding_baseline_fiber_aspect": int(outside.sum()),
            "row_indices_outside_sample": sub.index[outside].tolist()[:25],
        }
    return {"type": "reference_ranges_warning", "columns": out}


def _suspicious_flex_ge_comp(df: pd.DataFrame) -> dict[str, Any]:
    if "flexural_true" not in df.columns or "compressive_true" not in df.columns:
        return {"skipped": True}
    fc = pd.to_numeric(df["compressive_true"], errors="coerce")
    ff = pd.to_numeric(df["flexural_true"], errors="coerce")
    m = fc.notna() & ff.notna() & (ff >= fc)
    return {
        "n_suspicious": int(m.sum()),
        "row_indices_sample": df.index[m].tolist()[:25],
    }


def _max_group_share_warning(df: pd.DataFrame) -> dict[str, Any]:
    if "source_group" not in df.columns or len(df) == 0:
        return {"skipped": True}
    vc = df["source_group"].astype(str).value_counts()
    top = int(vc.iloc[0])
    return {
        "largest_group": str(vc.index[0]),
        "largest_group_count": top,
        "largest_group_share": round(top / len(df), 6),
    }


def validate_for_lab_strength_training(df: pd.DataFrame) -> dict[str, Any]:
    """
    返回 dict:
      ok: bool
      problems: list[str]（任一即 ok=False）
      warnings: dict（不导致 ok=False）
      n_rows, n_columns
    """
    problems: list[str] = []
    miss = _missing_columns(df)
    if miss:
        problems.append(f"缺少训练必需列（共 {len(miss)}）: {miss}")

    num_cols = [
        c for c in LAB_STRENGTH_FEATURE_COLUMNS if c != "source_group"
    ] + list(TRAINING_LABEL_AND_GROUP)
    problems.extend(_coerce_issues(df, num_cols))
    problems.extend(_positive_strength_issues(df))
    problems.extend(_duplicate_issues(df))
    problems.extend(_source_group_issues(df))

    n_mr = 0
    if "needs_manual_review" in df.columns:
        mr = pd.to_numeric(df["needs_manual_review"], errors="coerce").fillna(0)
        n_mr = int((mr >= 0.5).sum())

    try:
        wr_summary = summarize_water_reducer_features(df)
    except Exception as ex:  # noqa: BLE001
        wr_summary = {"error": type(ex).__name__, "detail": str(ex)[:200]}

    warnings = {
        "reference_ranges": _reference_range_warnings(df),
        "suspicious_flexural_ge_compressive": _suspicious_flex_ge_comp(df),
        "source_group_max_share": _max_group_share_warning(df),
        "needs_manual_review_rows": {
            "count": n_mr,
            "note": "为 1 的行不入公式基线矩阵，详见训练报告 manual_review_*",
        },
        "optional_tracing_summary": summarize_tracing_columns(df),
        "water_reducer_feature_summary": wr_summary,
    }

    return {
        "ok": len(problems) == 0,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "problems": problems,
        "warnings": warnings,
    }
