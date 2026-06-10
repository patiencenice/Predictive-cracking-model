"""
工程抗裂纤维体系 — 侧栏 UI（不进 FEATURE_COLUMNS / 主训练）。
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from experiments.fiber_engineering.derive import MATERIAL_FIBER_DEFAULTS
from src.features import FIBER_MATERIAL_MAP, FIBER_TYPE_MAP

_DISPERSION_OPTIONS = ["good", "normal", "poor"]
_DISPERSION_LABELS = {
    "good": "良好",
    "normal": "一般",
    "poor": "团聚风险",
}

_BOND_OPTIONS = ["weak", "medium", "strong"]
_BOND_LABELS = {
    "weak": "偏弱",
    "medium": "中等",
    "strong": "较强",
}

_ROLE_OPTIONS = [
    "plastic_crack_control",
    "thermal_crack_control",
    "toughness_enhancement",
    "multi_function",
]
_ROLE_LABELS = {
    "plastic_crack_control": "塑性抗裂",
    "thermal_crack_control": "温度抗裂",
    "toughness_enhancement": "韧性增强",
    "multi_function": "复合功能",
}


def _apply_material_defaults(data: dict[str, Any], mat: str) -> None:
    """材质切换时刷新 E_f / l_f 默认（仅当用户未锁定自定义值）。"""
    defs = MATERIAL_FIBER_DEFAULTS.get(mat, MATERIAL_FIBER_DEFAULTS["钢纤维"])
    prev = st.session_state.get("_sfc_fiber_material_prev")
    if prev != mat:
        st.session_state["sfc_fiber_elastic_modulus_gpa"] = float(defs["fiber_elastic_modulus_gpa"])
        st.session_state["sfc_fiber_length_mm"] = float(defs["fiber_length_mm"])
        st.session_state["_sfc_fiber_material_prev"] = mat


def render_fiber_sidebar_inputs(data: dict[str, Any]) -> None:
    """
    纤维参数侧栏：基础参数 / 工程抗裂参数 / 高级（折叠）。
    写入 data 字典，供 normalize 与 derive 使用。
    """
    st.caption(
        "工程抗裂纤维体系输入：用于机理解释、温度应力联动与诊断展示；"
        "**不进入**主模型训练特征。"
    )

    # —— 基础参数 ——
    st.markdown("**基础参数**")
    mat_opts = list(FIBER_MATERIAL_MAP.keys())
    cur_mat = str(data.get("fiber_material", mat_opts[0]))
    idx_mat = mat_opts.index(cur_mat) if cur_mat in mat_opts else 0
    mat = st.selectbox(
        "纤维材质",
        mat_opts,
        index=idx_mat,
        key="sfc_fiber_material",
    )
    data["fiber_material"] = mat
    _apply_material_defaults(data, mat)
    defs = MATERIAL_FIBER_DEFAULTS.get(mat, MATERIAL_FIBER_DEFAULTS["钢纤维"])

    data["fiber_content"] = st.number_input(
        "体积掺量 (%)",
        min_value=0.5,
        max_value=3.0,
        value=float(data.get("fiber_content", 1.0)),
        step=0.05,
        key="sfc_fiber_content",
    )
    data["fiber_length_mm"] = st.number_input(
        "纤维长度 l_f (mm)",
        min_value=3.0,
        max_value=80.0,
        value=float(
            st.session_state.get(
                "sfc_fiber_length_mm", data.get("fiber_length_mm", defs["fiber_length_mm"])
            )
        ),
        step=0.5,
        key="sfc_fiber_length_mm",
        help="长径比不能完全反映桥联尺度；长度影响裂缝跨越能力。",
    )
    data["aspect_ratio"] = st.number_input(
        "长径比",
        min_value=30.0,
        max_value=100.0,
        value=float(data.get("aspect_ratio", 50.0)),
        step=1.0,
        key="sfc_aspect_ratio",
    )
    data["tensile_strength"] = st.number_input(
        "单丝抗拉强度 (MPa)",
        min_value=300.0,
        max_value=3000.0,
        value=float(data.get("tensile_strength", 1200.0)),
        step=50.0,
        key="sfc_tensile_strength",
    )
    data["fiber_elastic_modulus_gpa"] = st.number_input(
        "纤维弹性模量 E_f (GPa)",
        min_value=1.0,
        max_value=250.0,
        value=float(
            st.session_state.get(
                "sfc_fiber_elastic_modulus_gpa",
                data.get("fiber_elastic_modulus_gpa", defs["fiber_elastic_modulus_gpa"]),
            )
        ),
        step=1.0,
        key="sfc_fiber_elastic_modulus_gpa",
        help="高模量比高强度更影响温度收缩约束与变形协调能力。",
    )
    ft_opts = list(FIBER_TYPE_MAP.keys())
    cur_ft = str(data.get("fiber_type", ft_opts[0]))
    data["fiber_type"] = st.selectbox(
        "外形/排布类型",
        ft_opts,
        index=ft_opts.index(cur_ft) if cur_ft in ft_opts else 0,
        key="sfc_fiber_type",
    )

    st.markdown("---")
    st.markdown("**工程抗裂参数**")
    cur_disp = str(data.get("fiber_dispersion_quality", "normal"))
    data["fiber_dispersion_quality"] = st.selectbox(
        "纤维分散等级",
        _DISPERSION_OPTIONS,
        index=_DISPERSION_OPTIONS.index(cur_disp) if cur_disp in _DISPERSION_OPTIONS else 1,
        format_func=lambda x: _DISPERSION_LABELS.get(x, x),
        key="sfc_fiber_dispersion_quality",
        help="解释局部弱区、团聚与施工离析风险。",
    )
    cur_bond = str(data.get("fiber_interface_bond", "medium"))
    data["fiber_interface_bond"] = st.selectbox(
        "纤维界面粘结等级",
        _BOND_OPTIONS,
        index=_BOND_OPTIONS.index(cur_bond) if cur_bond in _BOND_OPTIONS else 1,
        format_func=lambda x: _BOND_LABELS.get(x, x),
        key="sfc_fiber_interface_bond",
        help="解释 ITZ、拉拔行为与桥联能力。",
    )
    cur_role = str(data.get("fiber_function_role", "multi_function"))
    data["fiber_function_role"] = st.selectbox(
        "纤维功能定位",
        _ROLE_OPTIONS,
        index=_ROLE_OPTIONS.index(cur_role) if cur_role in _ROLE_OPTIONS else 3,
        format_func=lambda x: _ROLE_LABELS.get(x, x),
        key="sfc_fiber_function_role",
    )

    with st.expander("Fiber advanced settings（高级，占位）", expanded=False):
        st.caption("以下字段为研发占位，当前不参与任何计算。")
        d_mm = (
            float(data["fiber_length_mm"]) / float(data["aspect_ratio"])
            if float(data["aspect_ratio"]) > 1e-6
            else None
        )
        if d_mm is not None:
            st.text_input(
                "纤维直径（自动推导, mm）",
                value=f"{d_mm:.4f}",
                disabled=True,
                key="sfc_fiber_diameter_display",
            )
        data["fiber_orientation_factor"] = st.text_input(
            "取向系数（占位）",
            value=str(data.get("fiber_orientation_factor") or ""),
            key="sfc_fiber_orientation_factor",
            help="预留：纤维取向对桥联效率的影响。",
        )
        data["fiber_pullout_behavior"] = st.text_input(
            "pullout_behavior（占位）",
            value=str(data.get("fiber_pullout_behavior") or ""),
            key="sfc_fiber_pullout_behavior",
        )
        data["fiber_bridge_factor"] = st.text_input(
            "bridge_factor（占位）",
            value=str(data.get("fiber_bridge_factor") or ""),
            key="sfc_fiber_bridge_factor",
        )
