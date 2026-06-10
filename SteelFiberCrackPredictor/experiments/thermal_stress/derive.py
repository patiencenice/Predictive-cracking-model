"""
温度应力解释链（Phase 1）纯派生函数。

无量纲工程指数，不输出 MPa，不替代主模型 crack_* 输出。
禁止将未知输入静默改写为 0。
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

EPS_W_MM = 1e-3
EPS_SEGMENT_M = 1e-2

_RESTRAINT_BASE = {"low": 0.25, "medium": 0.50, "high": 0.75}

_E_REF_MIN = 25_000.0
_E_REF_MAX = 45_000.0
_FCU_REF_MIN = 25.0
_FCU_REF_MAX = 55.0
_E_FROM_FCU_COEF = 4700.0

_ALPHA_MIN = 8.0e-6
_ALPHA_MAX = 13.0e-6


def _to_float(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    if isinstance(v, str) and not v.strip():
        return None
    try:
        x = float(pd.to_numeric(v, errors="coerce"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _series_float(row: pd.Series, key: str) -> float | None:
    if key not in row.index:
        return None
    return _to_float(row[key])


def _series_str_lower(row: pd.Series, key: str) -> str | None:
    if key not in row.index:
        return None
    v = row[key]
    try:
        if pd.isna(v):
            return None
    except TypeError:
        pass
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s or s == "nan":
        return None
    return s


def _truthy_slip(v: Any) -> bool | None:
    """返回是否设置滑移层；无法解析则 None。"""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except TypeError:
        pass
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v) != 0.0
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "是", "有"):
        return True
    if s in ("0", "false", "no", "n", "否", "无"):
        return False
    return None


def g_delta_t_eff(delta_t_eff: float) -> float:
    """
    温差驱动项 g(|ΔT|)：分段线性上升至上限 1，减弱极端温差对指数的支配。
    单位：输入 ΔT 为 °C 与 K 等价差值。
    """
    a = abs(float(delta_t_eff))
    if a <= 8.0:
        return (a / 8.0) * 0.30
    if a <= 20.0:
        return 0.30 + (a - 8.0) / 12.0 * 0.45
    if a <= 40.0:
        return 0.75 + (a - 20.0) / 20.0 * 0.20
    return 1.0


def _norm_E_from_user_mpa(E_mpa: float) -> float:
    t = (E_mpa - _E_REF_MIN) / max(_E_REF_MAX - _E_REF_MIN, 1.0)
    return float(min(max(t, 0.0), 1.0))


def _E_mpa_from_row(row: pd.Series) -> tuple[float | None, str]:
    """返回 (E_MPa, source_tag)。"""
    Eu = _series_float(row, "elastic_modulus_E_user")
    if Eu is not None:
        if Eu <= 120.0:
            Eu = Eu * 1000.0
        if Eu < 5000.0:
            return None, "elastic_modulus_E_user_too_small"
        return Eu, "elastic_modulus_E_user"
    fcu = _series_float(row, "cube_strength_mpa")
    if fcu is None:
        return None, "missing"
    e = _E_FROM_FCU_COEF * math.sqrt(max(fcu, 1.0))
    return e, "from_cube_strength_mpa"


def _norm_E_proxy(E_mpa: float, source: str) -> float:
    if source == "from_cube_strength_mpa":
        e0 = _E_FROM_FCU_COEF * math.sqrt(_FCU_REF_MIN)
        e1 = _E_FROM_FCU_COEF * math.sqrt(_FCU_REF_MAX)
        t = (E_mpa - e0) / max(e1 - e0, 1.0)
        return float(min(max(t, 0.0), 1.0))
    return _norm_E_from_user_mpa(E_mpa)


def _norm_alpha(alpha: float) -> float:
    t = (alpha - _ALPHA_MIN) / max(_ALPHA_MAX - _ALPHA_MIN, 1e-12)
    return float(min(max(t, 0.0), 1.0))


def _enum_mult_hydration(row: pd.Series) -> tuple[float, int]:
    s = _series_str_lower(row, "hydration_heat_proxy_level")
    if s is None:
        return 1.0, 0
    if s == "low":
        return 0.95, 0
    if s == "medium":
        return 1.0, 0
    if s == "high":
        return 1.08, 0
    return 1.0, 1


def _enum_mult_insulation(row: pd.Series) -> tuple[float, int]:
    """保温好 → 风险系数略降。"""
    s = _series_str_lower(row, "surface_insulation_level")
    if s is None:
        return 1.0, 0
    if s == "low":
        return 1.05, 0
    if s == "medium":
        return 1.0, 0
    if s == "high":
        return 0.93, 0
    return 1.0, 1


def _wall_thickness_multiplier(w_mm: float) -> float:
    if w_mm >= 1200.0:
        return 1.10
    if w_mm >= 800.0:
        return 1.08
    if w_mm >= 400.0:
        return 1.05
    return 1.0


def derive_thermal_stress_features(row: pd.Series) -> dict[str, Any]:
    """
    由单行输入派生温度应力解释量与 missing_flag。

    不修改 row；未知不填 0；指数不可算时为 -1 且对应 missing_flag=1。
    """
    out: dict[str, Any] = {}

    # ----- delta_T_eff -----
    d_inner = _series_float(row, "delta_T_inner_outer")
    d_user = _series_float(row, "delta_T_user")
    t_peak = _series_float(row, "T_peak_observed")
    t_ref = _series_float(row, "T_reference")

    delta_t_eff: float | None = None
    delta_t_source: str | None = None
    if d_inner is not None:
        delta_t_eff = d_inner
        delta_t_source = "delta_T_inner_outer"
    elif d_user is not None:
        delta_t_eff = d_user
        delta_t_source = "delta_T_user"
    elif t_peak is not None and t_ref is not None:
        delta_t_eff = t_peak - t_ref
        delta_t_source = "T_peak_minus_T_reference"
    else:
        delta_t_source = "missing"

    out["delta_T_eff"] = delta_t_eff
    out["delta_T_eff_source"] = delta_t_source
    out["delta_T_eff_missing_flag"] = 1 if delta_t_eff is None else 0

    # ----- cooling_rate -----
    seg = _series_float(row, "segment_length_m")
    cooling_rate: float | None = None
    if delta_t_eff is not None and seg is not None and seg > 0.0:
        cooling_rate = abs(delta_t_eff) / max(seg, EPS_SEGMENT_M)
    out["cooling_rate"] = cooling_rate
    out["cooling_rate_missing_flag"] = 1 if cooling_rate is None else 0

    # ----- thermal_gradient_index -----
    w_mm = _series_float(row, "wall_thickness_mm")
    tgi: float | None = None
    if delta_t_eff is not None and w_mm is not None and w_mm > 0.0:
        tgi = delta_t_eff / max(w_mm, EPS_W_MM)
    out["thermal_gradient_index"] = tgi
    out["thermal_gradient_index_missing_flag"] = 1 if tgi is None else 0

    # ----- E_norm -----
    E_mpa, e_src = _E_mpa_from_row(row)
    E_norm: float | None = None
    if E_mpa is not None:
        E_norm = _norm_E_proxy(E_mpa, e_src)
    out["E_norm"] = E_norm
    out["E_norm_missing_flag"] = 1 if E_norm is None else 0

    # ----- alpha_norm -----
    alpha = _series_float(row, "thermal_expansion_alpha")
    alpha_norm: float | None = None
    if alpha is not None:
        alpha_norm = _norm_alpha(alpha)
    out["alpha_norm"] = alpha_norm
    out["alpha_norm_missing_flag"] = 1 if alpha_norm is None else 0

    # ----- restraint_factor_R -----
    rl = _series_str_lower(row, "restraint_level")
    R: float
    R_mf = 0
    illegal_rl = 0
    slip_raw = row["slip_layer_present"] if "slip_layer_present" in row.index else None
    if rl is None:
        R = -1.0
        R_mf = 1
    elif rl not in _RESTRAINT_BASE:
        R = -1.0
        R_mf = 1
        illegal_rl = 1
    else:
        R = float(_RESTRAINT_BASE[rl])
        if w_mm is not None and w_mm > 0.0:
            R *= _wall_thickness_multiplier(w_mm)
        rock = _series_str_lower(row, "rock_stiffness_class")
        if rock == "hard":
            R *= 1.05
        rebar = _series_str_lower(row, "rebar_constraint_class")
        if rebar == "heavy":
            R *= 1.05
        slip = _truthy_slip(slip_raw)
        if slip is True:
            R *= 0.775
        elif slip is False:
            pass
        R = float(min(max(R, 0.0), 1.0))

    out["restraint_factor_R"] = R
    out["restraint_factor_R_missing_flag"] = R_mf
    out["restraint_level_illegal_flag"] = illegal_rl

    # ----- thermal_stress_index -----
    tsi_mf = 0
    thermal_stress_index = -1.0
    if (
        delta_t_eff is not None
        and E_norm is not None
        and alpha_norm is not None
        and R_mf == 0
    ):
        g = g_delta_t_eff(delta_t_eff)
        thermal_stress_index = float(R * alpha_norm * E_norm * g)
        tsi_mf = 0
    else:
        thermal_stress_index = -1.0
        tsi_mf = 1

    out["thermal_stress_index"] = thermal_stress_index
    out["thermal_stress_index_missing_flag"] = tsi_mf

    # ----- thermal_crack_risk_index -----
    h_m, h_bad = _enum_mult_hydration(row)
    s_m, s_bad = _enum_mult_insulation(row)
    tcri_mf = 1
    thermal_crack_risk_index = -1.0
    if tsi_mf == 0 and thermal_stress_index >= 0.0 and math.isfinite(thermal_stress_index):
        thermal_crack_risk_index = float(min(thermal_stress_index * h_m * s_m, 1.5))
        tcri_mf = 0
        if h_bad or s_bad:
            out["thermal_crack_risk_enum_warning"] = 1
        else:
            out["thermal_crack_risk_enum_warning"] = 0
    else:
        thermal_crack_risk_index = -1.0
        tcri_mf = 1
        out["thermal_crack_risk_enum_warning"] = 0

    out["thermal_crack_risk_index"] = thermal_crack_risk_index
    out["thermal_crack_risk_index_missing_flag"] = tcri_mf

    return out


def thermal_stress_explain_sentence_zh(feats: dict[str, Any]) -> str:
    """基于派生结果的一句中文解释（启发式文案）。"""
    if feats.get("thermal_stress_index_missing_flag") == 1:
        return "当前温度应力指数无法闭合：请补全温差路径、弹性/膨胀参数与约束等级等关键输入。"
    tsi = feats.get("thermal_stress_index")
    try:
        tsi_f = float(tsi)
    except (TypeError, ValueError):
        tsi_f = -1.0
    if not math.isfinite(tsi_f) or tsi_f < 0.0:
        return "当前温度应力指数无法闭合：请检查输入枚举与数值是否合理。"
    R = float(feats.get("restraint_factor_R") or 0.0)
    g = g_delta_t_eff(float(feats.get("delta_T_eff") or 0.0))
    hi = tsi_f >= 0.35 or (R >= 0.65 and g >= 0.75)
    if hi:
        return "当前温度应力指数偏高，主要由较大温差与较高约束共同驱动。"
    if tsi_f >= 0.18:
        return "当前温度应力指数处于中等水平，温差与材料刚度/膨胀的组合对开裂解释有一定贡献。"
    return "当前温度应力指数相对较低；在已给约束与温差路径下，温度应力侧驱动偏弱（仍应结合主模型与现场）。"
