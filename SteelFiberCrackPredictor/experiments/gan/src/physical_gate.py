from __future__ import annotations

from typing import Any

import pandas as pd


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _range_mask(df: pd.DataFrame, col: str, lo: float, hi: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    v = _to_num(df[col])
    return v.notna() & (v >= lo) & (v <= hi)


def _positive_mask(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    v = _to_num(df[col])
    return v.notna() & (v > 0.0)


def _non_negative_mask(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    v = _to_num(df[col])
    return v.notna() & (v >= 0.0)


def _is_x_only_mode(config: dict[str, Any]) -> bool:
    gen = config.get("generation") or {}
    return str(gen.get("mode") or "").strip().lower() == "x_only"


def run_physical_gate(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """
    物理闸门仅筛选，不改值。
    """
    gate_cfg = config.get("physical_gate") or {}
    wb_lo = float(gate_cfg.get("w_b_ratio_min", 0.2))
    wb_hi = float(gate_cfg.get("w_b_ratio_max", 0.8))
    fc_lo = float(gate_cfg.get("fiber_content_min", 0.0))
    fc_hi = float(gate_cfg.get("fiber_content_max", 5.0))
    ar_lo = float(gate_cfg.get("aspect_ratio_min", 20.0))
    ar_hi = float(gate_cfg.get("aspect_ratio_max", 150.0))
    non_negative_cols = list(gate_cfg.get("non_negative_columns") or [])
    x_only = _is_x_only_mode(config)

    checks: dict[str, pd.Series] = {
        "w_b_ratio_range": _range_mask(df, "w_b_ratio", wb_lo, wb_hi),
        "fiber_content_range": _range_mask(df, "fiber_content", fc_lo, fc_hi),
        "aspect_ratio_range": _range_mask(df, "aspect_ratio", ar_lo, ar_hi),
    }
    if not x_only:
        checks["compressive_true_positive"] = _positive_mask(df, "compressive_true")
        checks["flexural_true_positive"] = _positive_mask(df, "flexural_true")

    if not x_only and {"flexural_true", "compressive_true"} <= set(df.columns):
        f = _to_num(df["flexural_true"])
        c = _to_num(df["compressive_true"])
        checks["flexural_lt_compressive"] = (
            f.notna() & c.notna() & (f > 0.0) & (c > 0.0) & (f < c)
        )
    elif not x_only:
        checks["flexural_lt_compressive"] = pd.Series(False, index=df.index)

    for col in non_negative_cols:
        checks[f"non_negative:{col}"] = _non_negative_mask(df, col)

    final_mask = pd.Series(True, index=df.index)
    for m in checks.values():
        final_mask = final_mask & m

    failed_idx = df.index[~final_mask]
    failed_reason_counts: dict[str, int] = {}
    for name, mask in checks.items():
        n_bad = int((~mask).sum())
        if n_bad > 0:
            failed_reason_counts[name] = n_bad

    return {
        "ok": bool(final_mask.all()),
        "n_rows_in": int(len(df)),
        "n_rows_pass": int(final_mask.sum()),
        "n_rows_fail": int((~final_mask).sum()),
        "pass_mask": final_mask,
        "failed_row_indices_sample": [int(i) for i in failed_idx[:50].tolist()],
        "failed_reason_counts": failed_reason_counts,
        "check_order": list(checks.keys()),
    }
