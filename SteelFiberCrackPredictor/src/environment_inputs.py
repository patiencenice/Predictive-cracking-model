"""
工程开裂环境场 + 施工工艺 — 侧栏 UI（不进 FEATURE_COLUMNS / 主训练）。
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.features import CASTING_METHOD_MAP

_CURING_OPTIONS = ["natural", "water", "membrane", "steam", "sealed"]
_CURING_LABELS = {
    "natural": "自然养护",
    "water": "洒水养护",
    "membrane": "覆膜养护",
    "steam": "蒸汽养护",
    "sealed": "密封养护",
}

_SOLAR_OPTIONS = ["low", "medium", "high"]
_SOLAR_LABELS = {
    "low": "阴凉",
    "medium": "普通日晒",
    "high": "强暴晒",
}

_VIBRATION_OPTIONS = ["poor", "normal", "good"]
_VIBRATION_LABELS = {
    "poor": "振捣不足",
    "normal": "正常",
    "good": "振捣良好",
}

_BLEEDING_OPTIONS = ["low", "medium", "high"]
_BLEEDING_LABELS = {
    "low": "低",
    "medium": "中",
    "high": "高",
}

_SEASON_OPTIONS = ["spring", "summer", "autumn", "winter"]
_SEASON_LABELS = {
    "spring": "春季",
    "summer": "夏季",
    "autumn": "秋季",
    "winter": "冬季",
}

_MEMBER_OPTIONS = ["slab", "wall", "beam", "column", "mass_concrete"]
_MEMBER_LABELS = {
    "slab": "板",
    "wall": "墙",
    "beam": "梁",
    "column": "柱",
    "mass_concrete": "大体积混凝土",
}


def _select_enum(
    label: str,
    options: list[str],
    labels: dict[str, str],
    data: dict[str, Any],
    key: str,
    default: str,
    *,
    help_text: str | None = None,
) -> str:
    cur = str(data.get(key, default))
    idx = options.index(cur) if cur in options else options.index(default)
    val = st.selectbox(
        label,
        options,
        index=idx,
        format_func=lambda x: labels.get(x, x),
        key=f"sfc_{key}",
        help=help_text,
    )
    data[key] = val
    return val


def render_environment_sidebar_inputs(data: dict[str, Any]) -> None:
    """
    环境与工艺侧栏：环境条件 / 养护制度 / 施工工艺 / 结构条件。
    写入 data，供 normalize 与 derive 使用。
    """
    st.caption(
        "工程开裂环境场 + 施工工艺场：用于蒸发/温差/收缩解释与温度应力联动；"
        "**不进入**主模型训练特征。"
    )

    st.markdown("**环境条件**")
    data["temperature"] = st.number_input(
        "环境温度 (°C)",
        min_value=5.0,
        max_value=40.0,
        value=float(data.get("temperature", 20.0)),
        step=0.5,
        key="sfc_temperature",
    )
    data["humidity"] = st.number_input(
        "相对湿度 (%)",
        min_value=30.0,
        max_value=100.0,
        value=float(data.get("humidity", 60.0)),
        step=1.0,
        key="sfc_humidity",
    )
    data["wind_speed_ms"] = st.number_input(
        "环境风速 (m/s)",
        min_value=0.0,
        max_value=15.0,
        value=float(data.get("wind_speed_ms", 1.5)),
        step=0.1,
        key="sfc_wind_speed_ms",
        help="风速越大，表面蒸发越快，塑性收缩与温差梯度风险越高。",
    )
    data["day_night_temp_diff_c"] = st.number_input(
        "昼夜温差 ΔT_day-night (°C)",
        min_value=0.0,
        max_value=30.0,
        value=float(data.get("day_night_temp_diff_c", 10.0)),
        step=0.5,
        key="sfc_day_night_temp_diff_c",
        help="用于温度应力、热胀冷缩循环与表层拉应力解释。",
    )
    _select_enum(
        "暴晒 / 太阳辐射等级",
        _SOLAR_OPTIONS,
        _SOLAR_LABELS,
        data,
        "solar_exposure_level",
        "medium",
    )

    st.markdown("---")
    st.markdown("**养护制度**")
    data["curing_days"] = st.number_input(
        "养护龄期 (天)",
        min_value=1,
        max_value=90,
        value=int(data.get("curing_days", 28)),
        step=1,
        key="sfc_curing_days",
    )
    _select_enum(
        "养护方式",
        _CURING_OPTIONS,
        _CURING_LABELS,
        data,
        "curing_method",
        "water",
    )

    st.markdown("---")
    st.markdown("**施工工艺**")
    cast_opts = list(CASTING_METHOD_MAP.keys())
    cur_cast = str(data.get("casting_method", cast_opts[0]))
    data["casting_method"] = st.selectbox(
        "浇筑方式",
        cast_opts,
        index=cast_opts.index(cur_cast) if cur_cast in cast_opts else 0,
        key="sfc_casting_method",
    )
    _select_enum(
        "振捣质量",
        _VIBRATION_OPTIONS,
        _VIBRATION_LABELS,
        data,
        "vibration_quality",
        "normal",
    )
    _select_enum(
        "泌水倾向",
        _BLEEDING_OPTIONS,
        _BLEEDING_LABELS,
        data,
        "bleeding_risk",
        "medium",
    )
    _select_enum(
        "施工季节",
        _SEASON_OPTIONS,
        _SEASON_LABELS,
        data,
        "construction_season",
        "spring",
    )

    st.markdown("---")
    st.markdown("**结构条件**")
    _select_enum(
        "构件类型",
        _MEMBER_OPTIONS,
        _MEMBER_LABELS,
        data,
        "member_type",
        "slab",
        help_text="不同构件散热、约束与裂缝模式不同；联动温度应力解释。",
    )
