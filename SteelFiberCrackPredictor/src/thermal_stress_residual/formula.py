"""
C30 温度应力公式基线：σ_T* = R · E · α · |ΔT|（与 engineering_chain 同量级定义）。
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from experiments.thermal_stress.derive import _E_mpa_from_row, _series_float
from src.thermal_stress_residual.restraint_map import restraint_factor_from_percent

C30_DEFAULTS: dict[str, float] = {
    "cube_strength_mpa": 30.0,
    "thermal_expansion_alpha": 1.0e-5,
}


def _row_float(row: pd.Series, key: str, default: float | None = None) -> float | None:
    v = _series_float(row, key)
    if v is not None:
        return v
    if default is not None and key in C30_DEFAULTS:
        return float(C30_DEFAULTS[key])
    if default is not None:
        return default
    return C30_DEFAULTS.get(key)


def row_delta_T_point(row: pd.Series) -> float | None:
    """点级有效温差：优先 delta_T_point，否则 T - T_reference。"""
    d = _series_float(row, "delta_T_point")
    if d is not None:
        return d
    t = _series_float(row, "specimen_temperature_c")
    t_ref = _series_float(row, "T_reference")
    if t is not None and t_ref is not None:
        return t - t_ref
    return None


def row_restraint_factor_R(row: pd.Series) -> float | None:
    r = _series_float(row, "restraint_factor_R")
    if r is not None and 0.0 <= r <= 1.0:
        return r
    pct = _series_float(row, "restraint_percent")
    return restraint_factor_from_percent(pct) if pct is not None else None


def row_sigma_T_formula_mpa(row: pd.Series) -> tuple[float | None, dict[str, Any]]:
    """
    单行公式基线预测（MPa，拉应力侧与仪器符号一致时 y_true 可为负）。
    返回 (sigma_formula, detail)。
    """
    detail: dict[str, Any] = {}
    alpha = _row_float(row, "thermal_expansion_alpha")
    delta_t = row_delta_T_point(row)
    E_mpa, e_src = _E_mpa_from_row(row)
    if E_mpa is None and _row_float(row, "cube_strength_mpa") is None:
        row_with_fcu = row.copy()
        row_with_fcu["cube_strength_mpa"] = C30_DEFAULTS["cube_strength_mpa"]
        E_mpa, e_src = _E_mpa_from_row(row_with_fcu)
    R = row_restraint_factor_R(row)

    detail.update(
        {
            "alpha": alpha,
            "delta_T_point": delta_t,
            "E_MPa": E_mpa,
            "E_source": e_src,
            "restraint_factor_R": R,
        }
    )

    if alpha is None or delta_t is None or E_mpa is None or R is None:
        detail["missing"] = True
        return None, detail

    eps_T = float(alpha * float(delta_t))
    sigma_signed = float(R) * float(E_mpa) * eps_T
    detail["epsilon_T"] = eps_T
    detail["missing"] = False
    return sigma_signed, detail


def augment_formula_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    preds: list[float | None] = []
    for _, row in out.iterrows():
        s, _ = row_sigma_T_formula_mpa(row)
        preds.append(s)
    out["sigma_formula_pred"] = preds
    if "axial_stress_mpa" in out.columns:
        out["residual_stress"] = out["axial_stress_mpa"] - out["sigma_formula_pred"]
    return out
