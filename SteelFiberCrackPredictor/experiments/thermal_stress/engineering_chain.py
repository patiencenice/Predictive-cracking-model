"""
温度应力 Phase 1 — 工程公式展示层（解释用）。

输出 ε_T、σ_T*、η 等展示量；不进入 FEATURE_COLUMNS，不改变 thermal_stress_index 算法。
σ_T* 按 σ = R·E·ε 量级计算，标注为工程解释标量，非有限元/真实 MPa 求解结果。
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from experiments.thermal_stress.derive import (
    _E_mpa_from_row,
    _series_float,
    derive_thermal_stress_features,
)
from experiments.thermal_stress.optional_fields import (
    derive_thermal_optional_context,
    f_t_source_label_zh,
    resolve_f_t_for_eta,
)

_ETA_LOW = 0.6
_ETA_HIGH = 1.0


def _eta_risk_band(eta: float) -> str:
    if eta < _ETA_LOW:
        return "低"
    if eta <= _ETA_HIGH:
        return "中"
    return "高"


def derive_thermal_engineering_display(row: pd.Series) -> dict[str, Any]:
    """
    由输入行派生工程公式链展示量（与 derive_thermal_stress_features 并行，不修改其输出）。
    """
    feats = derive_thermal_stress_features(row)
    out: dict[str, Any] = {
        "feats": feats,
        "engineering_chain_missing_flag": 1,
    }

    alpha = _series_float(row, "thermal_expansion_alpha")
    delta_t = feats.get("delta_T_eff")
    E_mpa, e_src = _E_mpa_from_row(row)
    _rmf = feats.get("restraint_factor_R_missing_flag")
    R_mf = 1 if _rmf is None else int(_rmf)
    R = feats.get("restraint_factor_R")

    dt_ok = feats.get("delta_T_eff_missing_flag") == 0
    a_ok = alpha is not None
    e_ok = E_mpa is not None
    r_ok = R_mf == 0 and R is not None and float(R) >= 0.0

    out["delta_T_eff"] = delta_t
    out["alpha_per_C"] = alpha
    out["E_MPa"] = E_mpa
    out["E_source"] = e_src
    out["restraint_factor_R"] = R
    out["step1_ok"] = bool(dt_ok and a_ok)
    out["step2_ok"] = bool(out["step1_ok"] and e_ok and r_ok)

    epsilon_T: float | None = None
    sigma_T_explain: float | None = None
    if out["step1_ok"] and delta_t is not None and alpha is not None:
        epsilon_T = float(alpha * abs(float(delta_t)))
        out["epsilon_T"] = epsilon_T

    if out["step2_ok"] and epsilon_T is not None and E_mpa is not None and R is not None:
        sigma_T_explain = float(R) * float(E_mpa) * epsilon_T
        out["sigma_T_explain"] = sigma_T_explain

    f_t_proxy, f_t_source, f_t_detail = resolve_f_t_for_eta(row)
    out["f_t_proxy_MPa"] = f_t_proxy
    out["f_t_proxy_source"] = f_t_source
    out["f_t_proxy_source_zh"] = f_t_source_label_zh(f_t_source)
    out["f_t_proxy_detail"] = f_t_detail
    out["fcu_k_MPa"] = _series_float(row, "cube_strength_mpa")
    out["splitting_tensile_strength_mpa"] = _series_float(row, "splitting_tensile_strength_mpa")
    out["flexural_strength_mpa"] = _series_float(row, "flexural_strength_mpa")

    out["optional_context"] = derive_thermal_optional_context(row)

    eta: float | None = None
    eta_band: str | None = None
    if (
        sigma_T_explain is not None
        and f_t_proxy is not None
        and f_t_proxy > 1e-6
        and math.isfinite(sigma_T_explain)
    ):
        eta = float(sigma_T_explain / f_t_proxy)
        eta_band = _eta_risk_band(eta)
        out["eta"] = eta
        out["eta_risk_band"] = eta_band
        out["step3_ok"] = True
        out["engineering_chain_missing_flag"] = 0
    else:
        out["step3_ok"] = False
        if out["step2_ok"]:
            if f_t_source == "missing":
                if _series_float(row, "splitting_tensile_strength_mpa") is None and _series_float(
                    row, "flexural_strength_mpa"
                ) is None and out["fcu_k_MPa"] is None:
                    out["eta_block_reason"] = (
                        "缺少 f_t：请提供实测劈裂抗拉、抗折或强度等级以估算"
                    )
                else:
                    out["eta_block_reason"] = "抗拉能力 f_t 不可用"
            else:
                out["eta_block_reason"] = "抗拉代理量不可用"
        else:
            out["eta_block_reason"] = "温差、α、E 或约束未闭合，无法计算完整公式链"

    return out


def thermal_engineering_conclusion_zh(display: dict[str, Any]) -> str:
    """基于 η 的工程解释句（报告口吻，非主模型输出）。"""
    if display.get("engineering_chain_missing_flag") == 1:
        reason = display.get("eta_block_reason") or "关键输入未闭合"
        return f"当前无法完成温度—应力—开裂判据闭合：{reason}。请补全侧栏温度应力可选输入与强度等级。"

    band = display.get("eta_risk_band")
    try:
        eta_f = float(display.get("eta"))
    except (TypeError, ValueError):
        eta_f = -1.0

    if band == "高":
        return (
            "当前温差与约束条件下，结构可能形成较明显的温度拉应力；"
            "若早龄期抗拉能力不足，则裂缝风险增加。"
        )
    if band == "中":
        return (
            "当前温度应力与抗拉能力之比处于中等水平，温度因素对开裂有一定贡献；"
            "宜结合养护、保温与约束释放措施复核。"
        )
    if band == "低":
        return "当前温度应力水平较低，温度变化尚不足以主导开裂行为。"
    return f"当前应力—抗拉比 η≈{eta_f:.2f}，请结合主模型裂缝指标与现场监测综合判定。"
