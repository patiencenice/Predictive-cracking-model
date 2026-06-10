"""
预测用输入默认值与归一化：API/旧版请求缺字段时自动补齐，避免 KeyError。
"""

from __future__ import annotations

import math
from typing import Any

from src.features import (
    ADMIXTURE_MAP,
    CASTING_METHOD_MAP,
    CONCRETE_TYPE_MAP,
    FIBER_MATERIAL_MAP,
    FIBER_TYPE_MAP,
    SAND_TYPE_MAP,
    SLAG_GRADE_MAP,
    STRENGTH_GRADE_ENC,
)
from src.lab_strength_residual.lab_mix_features import WATER_REDUCER_TYPE_MAP

# 与表单一致的完整键集合
PREDICTION_INPUT_DEFAULTS: dict[str, Any] = {
    "fiber_material": "钢纤维",
    "fiber_content": 1.0,
    "aspect_ratio": 50.0,
    "tensile_strength": 1200.0,
    "fiber_type": "端钩型",
    "fiber_length_mm": 35.0,
    "fiber_elastic_modulus_gpa": 200.0,
    "fiber_dispersion_quality": "normal",
    "fiber_interface_bond": "medium",
    "fiber_function_role": "multi_function",
    "fiber_orientation_factor": "",
    "fiber_pullout_behavior": "",
    "fiber_bridge_factor": "",
    "strength_grade": "C30",
    "concrete_type": "普通混凝土",
    "binder_content": 480.0,
    "cement_content": 400.0,
    "fly_ash": 60.0,
    "slag_grade": "S95",
    "slag_powder": 40.0,
    "mixing_water": 192.0,
    "w_b_ratio": 0.4,
    "sand_type": "天然砂",
    "sand_content": 720.0,
    "sand_ratio": 40.0,
    "stone_content": 1100.0,
    "admixture": "减水剂",
    "admixture_dosage": 3.0,
    "water_reducer_type": "无",
    "water_reduction_rate_pct": 20.0,
    "water_reduction_rate_unknown": False,
    "curing_days": 28,
    "temperature": 20.0,
    "humidity": 60.0,
    "wind_speed_ms": 1.5,
    "day_night_temp_diff_c": 10.0,
    "solar_exposure_level": "medium",
    "curing_method": "water",
    "vibration_quality": "normal",
    "bleeding_risk": "medium",
    "construction_season": "spring",
    "member_type": "slab",
    "casting_method": "泵送",
    # 温度应力 Phase 1（仅解释链；不进入主特征）
    "delta_T_inner_outer": None,
    "restraint_level": "unknown",
    "wall_thickness_mm": None,
    "thermal_expansion_alpha": None,
    "elastic_modulus_E_user": None,
    "slip_layer_present": "unknown",
    # Phase 1 扩展可选（解释/验证；缺失=None，不填 0）
    "splitting_tensile_strength_mpa": None,
    "flexural_strength_mpa": None,
    "core_peak_temperature_c": None,
    "surface_temperature_c": None,
    "time_to_peak_temperature_h": None,
    "cooling_rate_c_per_h": None,
    "restraint_percent": None,
    "restraint_code": "unknown",
    "thermal_crack_observed": "unknown",
    "thermal_crack_time_h": None,
    "thermal_crack_width_mm": None,
    "apparent_crack_filtered": "unknown",
}

_THERMAL_OPTIONAL_NUMERIC_KEYS = frozenset(
    {
        "delta_T_inner_outer",
        "wall_thickness_mm",
        "thermal_expansion_alpha",
        "elastic_modulus_E_user",
        "splitting_tensile_strength_mpa",
        "flexural_strength_mpa",
        "core_peak_temperature_c",
        "surface_temperature_c",
        "time_to_peak_temperature_h",
        "cooling_rate_c_per_h",
        "restraint_percent",
        "thermal_crack_time_h",
        "thermal_crack_width_mm",
        "fiber_length_mm",
        "fiber_elastic_modulus_gpa",
        "wind_speed_ms",
        "day_night_temp_diff_c",
    }
)

_THERMAL_OPTIONAL_ENUM_KEYS = frozenset(
    {
        "restraint_code",
        "thermal_crack_observed",
        "apparent_crack_filtered",
    }
)

_FIBER_ENGINEERING_ENUM_KEYS = frozenset(
    {
        "fiber_dispersion_quality",
        "fiber_interface_bond",
        "fiber_function_role",
    }
)

_FIBER_ENUM_ALLOWED: dict[str, frozenset[str]] = {
    "fiber_dispersion_quality": frozenset({"good", "normal", "poor"}),
    "fiber_interface_bond": frozenset({"weak", "medium", "strong"}),
    "fiber_function_role": frozenset(
        {
            "plastic_crack_control",
            "thermal_crack_control",
            "toughness_enhancement",
            "multi_function",
        }
    ),
}

_ENV_ENGINEERING_ENUM_KEYS = frozenset(
    {
        "curing_method",
        "solar_exposure_level",
        "vibration_quality",
        "bleeding_risk",
        "construction_season",
        "member_type",
    }
)

_ENV_ENUM_ALLOWED: dict[str, frozenset[str]] = {
    "curing_method": frozenset({"natural", "water", "membrane", "steam", "sealed"}),
    "solar_exposure_level": frozenset({"low", "medium", "high"}),
    "vibration_quality": frozenset({"poor", "normal", "good"}),
    "bleeding_risk": frozenset({"low", "medium", "high"}),
    "construction_season": frozenset({"spring", "summer", "autumn", "winter"}),
    "member_type": frozenset({"slab", "wall", "beam", "column", "mass_concrete"}),
}

_FLOAT_KEYS = frozenset(
    {
        "fiber_content",
        "aspect_ratio",
        "tensile_strength",
        "w_b_ratio",
        "cement_content",
        "sand_ratio",
        "binder_content",
        "fly_ash",
        "slag_powder",
        "stone_content",
        "sand_content",
        "mixing_water",
        "admixture_dosage",
        "water_reduction_rate_pct",
        "temperature",
        "humidity",
        "wind_speed_ms",
        "day_night_temp_diff_c",
    }
)
_INT_KEYS = frozenset({"curing_days"})
_BOOL_KEYS = frozenset({"water_reduction_rate_unknown"})


def normalize_prediction_inputs(raw: dict[str, Any]) -> dict[str, Any]:
    """合并默认值，并对数值字段做安全转换（兼容 JSON 字符串）。"""
    out: dict[str, Any] = dict(PREDICTION_INPUT_DEFAULTS)
    for key, val in raw.items():
        if val is None:
            continue
        if key not in PREDICTION_INPUT_DEFAULTS:
            continue
        if key == "restraint_level":
            s = str(val).strip().lower()
            out[key] = s if s in ("low", "medium", "high", "unknown") else "unknown"
            continue
        if key == "slip_layer_present":
            if isinstance(val, bool):
                out[key] = val
                continue
            s = str(val).strip().lower()
            if s in ("yes", "true", "1", "是", "y"):
                out[key] = True
            elif s in ("no", "false", "0", "否", "n"):
                out[key] = False
            else:
                out[key] = "unknown"
            continue
        if key in _THERMAL_OPTIONAL_ENUM_KEYS:
            s = str(val).strip().lower()
            if key == "restraint_code":
                if s in ("unknown", "", "nan", "none"):
                    out[key] = "unknown"
                elif s.upper() in ("R0", "R50", "R100"):
                    out[key] = s.upper()
                else:
                    out[key] = "unknown"
            elif key in ("thermal_crack_observed", "apparent_crack_filtered"):
                if s in ("unknown", "", "nan", "none"):
                    out[key] = "unknown"
                elif s in ("yes", "true", "1", "是", "y"):
                    out[key] = True
                elif s in ("no", "false", "0", "否", "n"):
                    out[key] = False
                else:
                    out[key] = "unknown"
            continue
        if key in _FIBER_ENGINEERING_ENUM_KEYS:
            s = str(val).strip().lower()
            allowed = _FIBER_ENUM_ALLOWED.get(key, frozenset())
            out[key] = s if s in allowed else PREDICTION_INPUT_DEFAULTS.get(key, "normal")
            continue
        if key in _ENV_ENGINEERING_ENUM_KEYS:
            s = str(val).strip().lower()
            allowed = _ENV_ENUM_ALLOWED.get(key, frozenset())
            out[key] = s if s in allowed else PREDICTION_INPUT_DEFAULTS.get(key)
            continue
        if key in _THERMAL_OPTIONAL_NUMERIC_KEYS:
            if isinstance(val, str) and not val.strip():
                out[key] = None
                continue
            if isinstance(val, str):
                try:
                    out[key] = float(val.replace(",", "."))
                except (TypeError, ValueError):
                    out[key] = None
                continue
            try:
                fv = float(val)
            except (TypeError, ValueError):
                out[key] = None
                continue
            out[key] = fv if math.isfinite(fv) else None
            continue
        if key in _BOOL_KEYS:
            if isinstance(val, str):
                out[key] = val.strip().lower() in ("1", "true", "yes", "on")
            else:
                out[key] = bool(val)
            continue
        if key in _INT_KEYS:
            try:
                out[key] = int(float(val))
            except (TypeError, ValueError):
                out[key] = PREDICTION_INPUT_DEFAULTS[key]
        elif key in _FLOAT_KEYS:
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                out[key] = PREDICTION_INPUT_DEFAULTS[key]
        elif key == "strength_grade" and isinstance(val, str):
            out[key] = val.strip().upper()
        else:
            out[key] = val
    return out


def validate_choice_fields(user_inputs: dict[str, Any]) -> tuple[bool, str]:
    """检查各下拉选项是否在允许集合内。"""
    if user_inputs["fiber_material"] not in FIBER_MATERIAL_MAP:
        return False, f"未知纤维材质: {user_inputs['fiber_material']!r}"
    if user_inputs["fiber_type"] not in FIBER_TYPE_MAP:
        return False, f"未知纤维外形: {user_inputs['fiber_type']!r}"
    if user_inputs["strength_grade"] not in STRENGTH_GRADE_ENC:
        return False, f"未知强度等级: {user_inputs['strength_grade']!r}"
    if user_inputs["concrete_type"] not in CONCRETE_TYPE_MAP:
        return False, f"未知混凝土类型: {user_inputs['concrete_type']!r}"
    if user_inputs["admixture"] not in ADMIXTURE_MAP:
        return False, f"未知外加剂类型: {user_inputs['admixture']!r}"
    if user_inputs["casting_method"] not in CASTING_METHOD_MAP:
        return False, f"未知浇筑方式: {user_inputs['casting_method']!r}"
    if user_inputs["slag_grade"] not in SLAG_GRADE_MAP:
        return False, f"未知矿粉等级: {user_inputs['slag_grade']!r}"
    if user_inputs["sand_type"] not in SAND_TYPE_MAP:
        return False, f"未知砂类型: {user_inputs['sand_type']!r}"
    wt = user_inputs.get("water_reducer_type", "无")
    if wt not in WATER_REDUCER_TYPE_MAP:
        return False, f"未知减水剂类型: {wt!r}"
    rl = user_inputs.get("restraint_level", "unknown")
    if rl not in ("low", "medium", "high", "unknown"):
        return False, f"未知约束等级 restraint_level: {rl!r}"
    slip = user_inputs.get("slip_layer_present", "unknown")
    if slip not in ("unknown", "yes", "no", True, False):
        return False, f"未知滑移层选项 slip_layer_present: {slip!r}"
    rc = user_inputs.get("restraint_code", "unknown")
    if rc not in ("unknown", "R0", "R50", "R100"):
        return False, f"未知试验约束档 restraint_code: {rc!r}"
    for bk in ("thermal_crack_observed", "apparent_crack_filtered"):
        bv = user_inputs.get(bk, "unknown")
        if bv not in ("unknown", True, False):
            return False, f"未知选项 {bk}: {bv!r}"
    for fk, allowed in _FIBER_ENUM_ALLOWED.items():
        fv = str(user_inputs.get(fk, "")).strip().lower()
        if fv and fv not in allowed:
            return False, f"未知纤维工程枚举 {fk}: {fv!r}"
    for ek, allowed in _ENV_ENUM_ALLOWED.items():
        ev = str(user_inputs.get(ek, "")).strip().lower()
        if ev and ev not in allowed:
            return False, f"未知环境工程枚举 {ek}: {ev!r}"
    return True, "OK"
