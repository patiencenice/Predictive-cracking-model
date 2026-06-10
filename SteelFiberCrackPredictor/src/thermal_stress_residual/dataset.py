from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.thermal_stress_residual.formula import (
    C30_DEFAULTS,
    augment_formula_column,
    row_sigma_T_formula_mpa,
)
from src.thermal_stress_residual.restraint_map import restraint_factor_from_percent

THERMAL_RESIDUAL_FEATURE_NAMES: tuple[str, ...] = (
    "w_b_ratio",
    "restraint_factor_R",
    "time_h_norm",
    "specimen_temperature_c",
    "delta_T_point",
    "deformation_um",
    "deformation_missing_flag",
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yt - yp
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(mask.sum())}


def _finite_float(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _row_feature_vector(row: pd.Series) -> np.ndarray | None:
    vals: list[float] = []
    for name in THERMAL_RESIDUAL_FEATURE_NAMES:
        if name == "deformation_missing_flag":
            d = _finite_float(row.get("deformation_um"))
            vals.append(0.0 if d is not None else 1.0)
            continue
        x = _finite_float(row.get(name))
        if x is None:
            return None
        vals.append(x)
    return np.asarray(vals, dtype=np.float64)


def build_xy_matrices(
    df: pd.DataFrame,
    *,
    target_col: str = "axial_stress_mpa",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray | None,
    np.ndarray,
    dict[str, Any],
]:
    """
    构建残差学习矩阵。
    y_resid = y_true - sigma_formula_pred
    groups = source_file（物理试验文件，避免同 xls 多 R 标签泄漏）
    """
    X_rows: list[np.ndarray] = []
    y_resid: list[float] = []
    y_true: list[float] = []
    formula: list[float] = []
    groups: list[str] = []
    row_indices: list[int] = []

    dropped = 0
    for i, row in df.iterrows():
        y = _finite_float(row.get(target_col))
        f, det = row_sigma_T_formula_mpa(row)
        if y is None or f is None or det.get("missing"):
            dropped += 1
            continue
        xv = _row_feature_vector(row)
        if xv is None:
            dropped += 1
            continue
        X_rows.append(xv)
        y_resid.append(y - f)
        y_true.append(y)
        formula.append(f)
        sf = str(row.get("source_file", row.get("sample_id", i)))
        groups.append(sf)
        row_indices.append(int(i))

    if not X_rows:
        raise ValueError("无有效训练行：检查 axial_stress_mpa、温度列与公式输入。")

    stats = {
        "n_rows_input": int(len(df)),
        "n_rows_used": len(X_rows),
        "n_rows_dropped": dropped,
        "feature_names": list(THERMAL_RESIDUAL_FEATURE_NAMES),
        "target_col": target_col,
        "C30_defaults": dict(C30_DEFAULTS),
    }
    return (
        np.vstack(X_rows),
        np.asarray(y_resid, dtype=np.float64),
        np.asarray(y_true, dtype=np.float64),
        np.asarray(formula, dtype=np.float64),
        list(THERMAL_RESIDUAL_FEATURE_NAMES),
        np.asarray(groups, dtype=object),
        np.asarray(row_indices, dtype=np.int64),
        stats,
    )


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """补全 C30 缺省材料参数与 restraint_factor_R。"""
    out = df.copy()
    if "cube_strength_mpa" not in out.columns:
        out["cube_strength_mpa"] = C30_DEFAULTS["cube_strength_mpa"]
    out["cube_strength_mpa"] = out["cube_strength_mpa"].fillna(C30_DEFAULTS["cube_strength_mpa"])
    if "thermal_expansion_alpha" not in out.columns:
        out["thermal_expansion_alpha"] = C30_DEFAULTS["thermal_expansion_alpha"]
    out["thermal_expansion_alpha"] = out["thermal_expansion_alpha"].fillna(
        C30_DEFAULTS["thermal_expansion_alpha"]
    )
    if "restraint_factor_R" not in out.columns:
        out["restraint_factor_R"] = out["restraint_percent"].map(
            lambda x: restraint_factor_from_percent(x) if pd.notna(x) else None
        )
    return augment_formula_column(out)
