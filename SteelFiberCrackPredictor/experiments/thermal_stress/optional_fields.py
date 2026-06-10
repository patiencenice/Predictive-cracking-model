"""
温度应力 Phase 1 — 扩展可选输入（解释/验证用，不进 FEATURE_COLUMNS）。

所有字段可选；缺失不静默填 0；不修改 derive 中 thermal_stress_index 与 restraint_factor_R 算法。
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from experiments.thermal_stress.derive import _series_float, _series_str_lower, _truthy_slip
from src.lab_formula_gb import ftk_standard_mpa_from_fcu_k

# 试验约束档（与 Phase1 restraint_level 枚举独立）

OPTIONAL_INPUT_FIELDS: tuple[str, ...] = (
    "splitting_tensile_strength_mpa",
    "flexural_strength_mpa",
    "core_peak_temperature_c",
    "surface_temperature_c",
    "time_to_peak_temperature_h",
    "cooling_rate_c_per_h",
    "restraint_percent",
    "restraint_code",
    "thermal_crack_observed",
    "thermal_crack_time_h",
    "thermal_crack_width_mm",
    "apparent_crack_filtered",
)


def _series_restraint_code(row: pd.Series) -> str | None:
    if "restraint_code" not in row.index:
        return None
    raw = row["restraint_code"]
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip().upper()
    if not s or s in ("UNKNOWN", "NAN", "NONE"):
        return None
    if s in ("R0", "R50", "R100"):
        return s
    m = re.match(r"^R(0|50|100)$", s, re.I)
    return f"R{m.group(1)}" if m else None


def _series_optional_bool(row: pd.Series, key: str) -> bool | None:
    if key not in row.index:
        return None
    v = row[key]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (bool,)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("unknown", "", "nan", "none"):
        return None
    if s in ("1", "true", "yes", "y", "是", "有"):
        return True
    if s in ("0", "false", "no", "n", "否", "无"):
        return False
    return _truthy_slip(v)


def resolve_f_t_for_eta(row: pd.Series) -> tuple[float | None, str, dict[str, Any] | None]:
    """
    η 分母 f_t 优先级：
    1. splitting_tensile_strength_mpa（实测劈裂抗拉）
    2. flexural_strength_mpa（实测抗折，备选代理，非标准 ft）
    3. cube_strength_mpa → GB50010 经验 ftk
    """
    ft_split = _series_float(row, "splitting_tensile_strength_mpa")
    if ft_split is not None and ft_split > 1e-6:
        return float(ft_split), "measured_splitting_tensile_mpa", None

    ft_flex = _series_float(row, "flexural_strength_mpa")
    if ft_flex is not None and ft_flex > 1e-6:
        return (
            float(ft_flex),
            "measured_flexural_mpa_fallback",
            {"note": "抗折强度非标准劈裂抗拉 ft，仅作无实测 ft 时的备选解释代理"},
        )

    fcu = _series_float(row, "cube_strength_mpa")
    if fcu is not None:
        f_t, detail = ftk_standard_mpa_from_fcu_k(fcu)
        return f_t, "GB50010_ftk_from_cube_strength_mpa", detail

    return None, "missing", None


def f_t_source_label_zh(source: str) -> str:
    return {
        "measured_splitting_tensile_mpa": "实测劈裂抗拉强度",
        "measured_flexural_mpa_fallback": "实测抗折（备选代理）",
        "GB50010_ftk_from_cube_strength_mpa": "强度等级经验估算（GB50010）",
        "missing": "缺失",
    }.get(source, source)


def derive_thermal_optional_context(row: pd.Series) -> dict[str, Any]:
    """
    扩展可选输入的只读上下文（解释/验证数据集用）。
    不参与 thermal_stress_index 基础公式；缺失置 missing_flag=1，不填 0。
    """
    out: dict[str, Any] = {}

    for key in OPTIONAL_INPUT_FIELDS:
        if key in (
            "restraint_code",
        ):
            val = _series_restraint_code(row)
        elif key in ("thermal_crack_observed", "apparent_crack_filtered"):
            val = _series_optional_bool(row, key)
        else:
            val = _series_float(row, key)
        out[key] = val
        if key == "restraint_code":
            out[f"{key}_missing_flag"] = 0 if val is not None else 1
        elif key in ("thermal_crack_observed", "apparent_crack_filtered"):
            out[f"{key}_missing_flag"] = 0 if val is not None else 1
        else:
            out[f"{key}_missing_flag"] = 0 if val is not None else 1

    # 温度路径：可选解释量（不强制进入 σ_T* 基线公式）
    core = _series_float(row, "core_peak_temperature_c")
    surf = _series_float(row, "surface_temperature_c")
    if core is not None and surf is not None:
        out["core_surface_delta_c"] = float(core - surf)
        out["core_surface_delta_missing_flag"] = 0
    else:
        out["core_surface_delta_c"] = None
        out["core_surface_delta_missing_flag"] = 1

    t_peak = _series_float(row, "time_to_peak_temperature_h")
    cool = _series_float(row, "cooling_rate_c_per_h")
    out["temperature_path_summary_zh"] = _temperature_path_summary_zh(
        core, surf, t_peak, cool, out.get("core_surface_delta_c")
    )

    # 试验约束：与 restraint_factor_R 区分
    rp = out.get("restraint_percent")
    rc = out.get("restraint_code")
    r_phase = _series_str_lower(row, "restraint_level")
    out["restraint_test_note_zh"] = (
        "试验约束率 restraint_percent / restraint_code 为 C30 仪器试验原始标签，"
        "不等同于 Phase1 解释链 restraint_level → restraint_factor_R。"
    )
    if rc is not None or rp is not None:
        parts = []
        if rc is not None:
            parts.append(f"restraint_code={rc}")
        if rp is not None:
            parts.append(f"restraint_percent={rp}%")
        if r_phase is not None and r_phase != "unknown":
            parts.append(f"（对照：解释链 restraint_level={r_phase}）")
        out["restraint_test_display_zh"] = "；".join(parts)
    else:
        out["restraint_test_display_zh"] = None

    # 裂缝观测：仅验证 η 关系
    crack_obs = out.get("thermal_crack_observed")
    if crack_obs is not None or out.get("thermal_crack_width_mm") is not None:
        out["crack_validation_note_zh"] = (
            "裂缝观测字段仅用于后续 η–开裂关系验证，不接入主预测。"
        )
    else:
        out["crack_validation_note_zh"] = None

    return out


def _temperature_path_summary_zh(
    core: float | None,
    surf: float | None,
    t_peak: float | None,
    cool: float | None,
    delta: float | None,
) -> str | None:
    parts: list[str] = []
    if core is not None:
        parts.append(f"芯部峰值温度 {core:.2f} ℃")
    if surf is not None:
        parts.append(f"表面温度 {surf:.2f} ℃")
    if delta is not None:
        parts.append(f"芯表温差 {delta:.2f} ℃")
    if t_peak is not None:
        parts.append(f"达峰时间 {t_peak:.2f} h")
    if cool is not None:
        parts.append(f"冷却速率 {cool:.2f} ℃/h")
    return "；".join(parts) if parts else None
