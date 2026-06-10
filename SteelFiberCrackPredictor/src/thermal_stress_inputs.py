"""
温度应力 Phase 1：侧栏可选输入与 → derive_thermal_stress_features 的输入行构造。

不参与 FEATURE_COLUMNS / 主训练；仅为解释面板提供 derive 所需列名与单位换算。
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src.features import STRENGTH_GRADE_TO_MPA


def parse_optional_float_text(raw: str) -> float | None:
    """空串或非有限值 → None；禁止静默成 0。"""
    s = (raw or "").strip()
    if not s:
        return None
    try:
        x = float(s.replace(",", "."))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


_RESTRAINT_LABELS = {
    "unknown": "unknown（未知，R 按缺失）",
    "low": "low（低约束）",
    "medium": "medium（中约束）",
    "high": "high（高约束）",
}

_SLIP_LABELS = {
    "unknown": "未知（不按「否」处理）",
    "yes": "是",
    "no": "否",
}


def render_thermal_stress_sidebar_inputs(data: dict[str, Any]) -> None:
    """
    在侧栏当前 expander 内渲染 Phase 1 可选字段，写入 data（与主表单同级键名，供 normalize 合并）。
    """
    st.markdown("###### 温度应力解释（Phase 1，可选）")
    st.caption(
        "仅用于下方「温度应力解释」卡片闭合 thermal_stress_index；**不进入**主模型训练特征。"
    )

    data["delta_T_inner_outer"] = parse_optional_float_text(
        st.text_input(
            "内外/芯表温差 ΔT（℃）",
            value="",
            key="sfc_ts_delta_T_inner_outer",
            help="可选。填则优先作为 ΔT_eff 来源；留空表示缺失（不填 0）。",
        )
    )

    r_opts = ["unknown", "low", "medium", "high"]
    data["restraint_level"] = st.selectbox(
        "约束等级",
        options=r_opts,
        index=0,
        format_func=lambda x: _RESTRAINT_LABELS.get(x, x),
        key="sfc_ts_restraint_level",
        help="unknown 时约束因子 R 按缺失处理，**不会**当作 low。",
    )

    data["wall_thickness_mm"] = parse_optional_float_text(
        st.text_input(
            "构件厚度 / 井壁厚度（mm）",
            value="",
            key="sfc_ts_wall_thickness_mm",
            help="可选。用于 thermal_gradient_index 与 R 厚度修正；留空=缺失。",
        )
    )

    alpha_raw = st.text_input(
        "线膨胀系数 α（×10⁻⁶/℃）",
        value="",
        key="sfc_ts_alpha_x1e6",
        help="可选。留空则 alpha 缺失；**不会**默认填 10。填入 10 表示 10×10⁻⁶/℃。",
    )
    a = parse_optional_float_text(alpha_raw)
    data["thermal_expansion_alpha"] = (a * 1e-6) if a is not None else None

    data["elastic_modulus_E_user"] = parse_optional_float_text(
        st.text_input(
            "弹性模量 E（GPa，可选）",
            value="",
            key="sfc_ts_elastic_modulus_E_user_gpa",
            help="可选。留空时 derive 可继续用强度等级对应的 cube_strength_mpa 估计 E。",
        )
    )

    s_opts = ["unknown", "yes", "no"]
    slip_ui = st.selectbox(
        "是否设置滑移/隔离层",
        options=s_opts,
        index=0,
        format_func=lambda x: _SLIP_LABELS.get(x, x),
        key="sfc_ts_slip_layer_present",
        help="未知不按「否」；仅「是」时 R 乘以 0.775。",
    )
    if slip_ui == "yes":
        data["slip_layer_present"] = True
    elif slip_ui == "no":
        data["slip_layer_present"] = False
    else:
        data["slip_layer_present"] = "unknown"

    with st.expander("实测抗拉/抗折（可选，η 优先实测 ft）", expanded=False):
        st.caption("若填写劈裂抗拉强度，η 分母优先用实测值；否则可填抗折或沿用强度等级估算。")
        data["splitting_tensile_strength_mpa"] = parse_optional_float_text(
            st.text_input(
                "劈裂抗拉强度 ft（MPa）",
                value="",
                key="sfc_ts_splitting_tensile_mpa",
                help="可选。留空=缺失，不填 0。",
            )
        )
        data["flexural_strength_mpa"] = parse_optional_float_text(
            st.text_input(
                "抗折强度（MPa，备选）",
                value="",
                key="sfc_ts_flexural_mpa",
                help="无劈裂抗拉时的备选代理；留空=缺失。",
            )
        )

    with st.expander("温度路径（可选，解释用）", expanded=False):
        st.caption("用于描述温度应力形成过程；不强制参与 Step1~2 基础公式。")
        data["core_peak_temperature_c"] = parse_optional_float_text(
            st.text_input("芯部峰值温度（℃）", value="", key="sfc_ts_core_peak_temp")
        )
        data["surface_temperature_c"] = parse_optional_float_text(
            st.text_input("表面温度（℃）", value="", key="sfc_ts_surface_temp")
        )
        data["time_to_peak_temperature_h"] = parse_optional_float_text(
            st.text_input("达峰时间（h）", value="", key="sfc_ts_time_to_peak")
        )
        data["cooling_rate_c_per_h"] = parse_optional_float_text(
            st.text_input("冷却速率（℃/h）", value="", key="sfc_ts_cooling_rate")
        )

    with st.expander("约束试验条件（可选，≠ 解释链 R）", expanded=False):
        st.caption(
            "C30 仪器试验原始约束标签；**不直接等同于**上方「约束等级」派生的 restraint_factor_R。"
        )
        rc_opts = ["unknown", "R0", "R50", "R100"]
        rc_ui = st.selectbox(
            "试验约束档 restraint_code",
            options=rc_opts,
            index=0,
            format_func=lambda x: "未知" if x == "unknown" else x,
            key="sfc_ts_restraint_code",
        )
        data["restraint_code"] = rc_ui if rc_ui != "unknown" else "unknown"
        data["restraint_percent"] = parse_optional_float_text(
            st.text_input(
                "试验约束率 restraint_percent（%）",
                value="",
                key="sfc_ts_restraint_percent",
                help="可选 0/50/100；留空=缺失（选择 R 档时不自动填数）。",
            )
        )

    with st.expander("裂缝观测（可选，验证 η 用）", expanded=False):
        st.caption("仅用于后续 η 与开裂关系验证；不接入主模型预测。")
        crack_opts = ["unknown", "yes", "no"]
        crack_ui = st.selectbox(
            "是否观测到温度裂缝",
            options=crack_opts,
            index=0,
            format_func=lambda x: {"unknown": "未知", "yes": "是", "no": "否"}.get(x, x),
            key="sfc_ts_thermal_crack_observed",
        )
        if crack_ui == "yes":
            data["thermal_crack_observed"] = True
        elif crack_ui == "no":
            data["thermal_crack_observed"] = False
        else:
            data["thermal_crack_observed"] = "unknown"
        data["thermal_crack_time_h"] = parse_optional_float_text(
            st.text_input("裂缝出现时间（h）", value="", key="sfc_ts_thermal_crack_time_h")
        )
        data["thermal_crack_width_mm"] = parse_optional_float_text(
            st.text_input("裂缝宽度（mm）", value="", key="sfc_ts_thermal_crack_width_mm")
        )
        filt_ui = st.selectbox(
            "表观裂缝已过滤",
            options=crack_opts,
            index=0,
            format_func=lambda x: {"unknown": "未知", "yes": "是", "no": "否"}.get(x, x),
            key="sfc_ts_apparent_crack_filtered",
        )
        if filt_ui == "yes":
            data["apparent_crack_filtered"] = True
        elif filt_ui == "no":
            data["apparent_crack_filtered"] = False
        else:
            data["apparent_crack_filtered"] = "unknown"


def cube_strength_mpa_from_user_inputs(user_inputs: dict[str, Any]) -> float | None:
    """由强度等级得到 fcu,k（MPa），供 derive 的 E 代理；等级非法则 None。"""
    g = user_inputs.get("strength_grade")
    if not isinstance(g, str):
        return None
    g = g.strip().upper()
    return float(STRENGTH_GRADE_TO_MPA[g]) if g in STRENGTH_GRADE_TO_MPA else None


def series_for_thermal_derive(user_inputs: dict[str, Any]) -> pd.Series:
    """
    合并主表单与温度应力可选列，并注入 cube_strength_mpa（仅当行中尚无有效数值时），
    再交给 derive_thermal_stress_features。
    """
    row_dict: dict[str, Any] = dict(user_inputs)
    fcu_existing = row_dict.get("cube_strength_mpa")
    need_fcu = True
    if fcu_existing is not None:
        try:
            fcf = float(fcu_existing)
            need_fcu = not math.isfinite(fcf)
        except (TypeError, ValueError):
            need_fcu = True
    if need_fcu:
        fcu = cube_strength_mpa_from_user_inputs(user_inputs)
        if fcu is not None:
            row_dict["cube_strength_mpa"] = fcu
    return pd.Series(row_dict)
