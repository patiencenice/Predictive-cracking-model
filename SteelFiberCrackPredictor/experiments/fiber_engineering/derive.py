"""
工程抗裂纤维体系 — Phase 1 派生（解释层，不进 FEATURE_COLUMNS）。
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from experiments.thermal_stress.derive import _series_float, _series_str_lower

# 材质默认 E_f (GPa) 与 l_f (mm) — 仅作 UI 初值参考，derive 不静默覆盖用户输入
MATERIAL_FIBER_DEFAULTS: dict[str, dict[str, float]] = {
    "钢纤维": {"fiber_elastic_modulus_gpa": 200.0, "fiber_length_mm": 35.0},
    "玄武岩纤维": {"fiber_elastic_modulus_gpa": 90.0, "fiber_length_mm": 12.0},
    "聚丙烯纤维": {"fiber_elastic_modulus_gpa": 4.0, "fiber_length_mm": 12.0},
    "玻璃纤维": {"fiber_elastic_modulus_gpa": 85.0, "fiber_length_mm": 12.0},
}

_ASPECT_RATIO_REF_MIN = 30.0
_ASPECT_RATIO_REF_MAX = 100.0
_E_F_REF_GPA = 200.0

_DISPERSION_SCORE = {"good": 1.0, "normal": 0.65, "poor": 0.35}
_BOND_SCORE = {"weak": 0.45, "medium": 0.7, "strong": 0.95}

_FUNCTION_ROLE_ZH = {
    "plastic_crack_control": "塑性抗裂",
    "thermal_crack_control": "温度抗裂",
    "toughness_enhancement": "韧性增强",
    "multi_function": "复合功能",
}


def _clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _norm_aspect_ratio(ar: float | None) -> float | None:
    if ar is None or ar <= 0:
        return None
    t = (float(ar) - _ASPECT_RATIO_REF_MIN) / max(
        _ASPECT_RATIO_REF_MAX - _ASPECT_RATIO_REF_MIN, 1.0
    )
    return _clamp01(t)


def _level_zh(score: float) -> str:
    if score >= 0.72:
        return "较高"
    if score >= 0.45:
        return "中等"
    return "偏低"


def derive_fiber_engineering_features(row: pd.Series) -> dict[str, Any]:
    """
    由侧栏/CSV 输入派生纤维工程解释量。
    不修改 row；缺失不填 0；指数类不可算时 missing_flag=1。
    """
    out: dict[str, Any] = {}

    mat = row.get("fiber_material")
    mat_s = str(mat).strip() if mat is not None and not (isinstance(mat, float) and pd.isna(mat)) else ""

    E_f = _series_float(row, "fiber_elastic_modulus_gpa")
    l_f = _series_float(row, "fiber_length_mm")
    vol_pct = _series_float(row, "fiber_content")
    ar = _series_float(row, "aspect_ratio")

    out["fiber_elastic_modulus_gpa"] = E_f
    out["fiber_length_mm"] = l_f
    out["fiber_elastic_modulus_gpa_missing_flag"] = 0 if E_f is not None else 1
    out["fiber_length_mm_missing_flag"] = 0 if l_f is not None else 1

    # fiber_diameter_mm = l_f / aspect_ratio
    d_mm: float | None = None
    if l_f is not None and ar is not None and ar > 1e-6:
        d_mm = float(l_f / ar)
    out["fiber_diameter_mm"] = d_mm
    out["fiber_diameter_mm_missing_flag"] = 0 if d_mm is not None else 1

    ar_norm = _norm_aspect_ratio(ar)
    out["aspect_ratio_norm"] = ar_norm
    out["aspect_ratio_norm_missing_flag"] = 0 if ar_norm is not None else 1

    fci: float | None = None
    if vol_pct is not None and E_f is not None and ar_norm is not None:
        vol_frac = float(vol_pct) / 100.0
        e_norm = _clamp01(float(E_f) / _E_F_REF_GPA)
        fci = float(vol_frac * e_norm * ar_norm)
    out["fiber_constraint_index"] = fci if fci is not None else -1.0
    out["fiber_constraint_index_missing_flag"] = 0 if fci is not None else 1

    disp = _series_str_lower(row, "fiber_dispersion_quality") or "normal"
    if disp not in _DISPERSION_SCORE:
        disp = "normal"
        out["fiber_dispersion_illegal_flag"] = 1
    else:
        out["fiber_dispersion_illegal_flag"] = 0
    out["fiber_dispersion_quality"] = disp

    bond = _series_str_lower(row, "fiber_interface_bond") or "medium"
    if bond not in _BOND_SCORE:
        bond = "medium"
        out["fiber_interface_bond_illegal_flag"] = 1
    else:
        out["fiber_interface_bond_illegal_flag"] = 0
    out["fiber_interface_bond"] = bond

    role = _series_str_lower(row, "fiber_function_role") or "multi_function"
    if role not in _FUNCTION_ROLE_ZH:
        role = "multi_function"
        out["fiber_function_role_illegal_flag"] = 1
    else:
        out["fiber_function_role_illegal_flag"] = 0
    out["fiber_function_role"] = role
    out["fiber_function_role_zh"] = _FUNCTION_ROLE_ZH.get(role, role)

    # 工程倾向评分（解释层）
    e_score = _clamp01(float(E_f) / _E_F_REF_GPA) if E_f is not None else 0.0
    len_score = _clamp01(float(l_f) / 40.0) if l_f is not None else 0.0
    bridge_score = (
        0.35 * e_score
        + 0.25 * len_score
        + 0.20 * _BOND_SCORE[bond]
        + 0.20 * _DISPERSION_SCORE[disp]
    )
    if fci is not None:
        bridge_score = 0.6 * bridge_score + 0.4 * _clamp01(fci / 0.015)
    bridge_score = _clamp01(bridge_score)
    out["bridge_capacity_score"] = round(bridge_score, 4)

    thermal_coord = _clamp01(0.55 * e_score + 0.45 * _DISPERSION_SCORE[disp])
    out["thermal_coordination_score"] = round(thermal_coord, 4)

    interface_score = _clamp01(_BOND_SCORE[bond] * (0.7 + 0.3 * e_score))
    out["interface_synergy_score"] = round(interface_score, 4)

    out["fiber_engineering_summary"] = _build_summary(
        mat_s, role, disp, bond, E_f, l_f, fci, bridge_score, thermal_coord
    )
    out["thermal_fiber_note_zh"] = _thermal_fiber_note_zh(E_f, disp, thermal_coord)
    out["bridge_explanation_zh"] = _bridge_explanation_blocks(E_f, l_f, bond, disp)

    return out


def _build_summary(
    mat: str,
    role: str,
    disp: str,
    bond: str,
    E_f: float | None,
    l_f: float | None,
    fci: float | None,
    bridge: float,
    thermal: float,
) -> dict[str, str]:
    sys_type = f"{mat or '—'} · {_FUNCTION_ROLE_ZH.get(role, role)}"
    disp_risk = {
        "good": "分散良好，局部弱区风险低",
        "normal": "分散一般，宜关注施工均匀性",
        "poor": "存在团聚/离析风险，可能出现假性高风险区",
    }[disp]
    return {
        "system_type_zh": sys_type,
        "crack_resistance_tendency_zh": f"工程抗裂倾向：{_level_zh(bridge)}",
        "thermal_constraint_capacity_zh": f"温度变形协调能力：{_level_zh(thermal)}",
        "expected_bridging_zh": f"预计桥联能力：{_level_zh(bridge)}",
        "dispersion_risk_zh": disp_risk,
        "interface_bond_zh": {"weak": "偏弱", "medium": "中等", "strong": "较强"}[bond],
        "fiber_constraint_index_display": (
            f"{fci:.4f}" if fci is not None and math.isfinite(fci) else "—"
        ),
        "E_f_display": f"{E_f:.1f} GPa" if E_f is not None else "—",
        "l_f_display": f"{l_f:.1f} mm" if l_f is not None else "—",
    }


def _thermal_fiber_note_zh(
    E_f: float | None, disp: str, thermal_score: float
) -> str:
    parts = [
        "高模量纤维可提高变形协调能力，降低温度应力集中风险。"
    ]
    if E_f is not None and E_f >= 80.0 and disp == "good":
        parts.append("当前输入下：纤维约束体系较稳定。")
    elif E_f is not None and E_f >= 80.0:
        parts.append("模量较高，但分散等级一般，宜复核现场均匀性。")
    elif disp == "poor":
        parts.append("分散欠佳时，温度收缩约束可能在局部弱区集中。")
    if thermal_score >= 0.72:
        parts.append("温度协调评分偏高。")
    return " ".join(parts)


def _bridge_explanation_blocks(
    E_f: float | None,
    l_f: float | None,
    bond: str,
    disp: str,
) -> list[dict[str, str]]:
    """纤维桥联能力解释卡内容。"""
    cards: list[dict[str, str]] = []

    e_txt = (
        f"E_f≈{E_f:.0f} GPa：高模量纤维在约束温变时更有效地协调基体变形，"
        "比单纯提高抗拉强度更能抑制温度收缩应力集中。"
        if E_f is not None and E_f >= 50
        else "模量偏低（如 PPF）：主要抑制早期塑性裂缝，对温度收缩约束贡献有限。"
        if E_f is not None
        else "未提供 E_f：无法评估模量对桥联与温度协调的贡献。"
    )
    cards.append(
        {
            "title": "高模量 · 温度协调",
            "body": e_txt,
            "effects": "裂缝扩展↓（间接）· 温度应力释放↑ · 宽度发展受抑",
        }
    )

    l_txt = (
        f"l_f≈{l_f:.1f} mm：较长纤维跨越裂缝尺度能力更强，"
        "长径比 alone 不足以反映有效桥联几何。"
        if l_f is not None
        else "未提供纤维长度：桥联尺度解释不完整。"
    )
    cards.append(
        {
            "title": "长纤维 · 跨越能力",
            "body": l_txt,
            "effects": "裂缝扩展↓ · 裂缝宽度↓ · 残余抗拉↑",
        }
    )

    bond_txt = {
        "strong": "强界面：拉拔耗能高，裂缝尖端应力重分布更充分。",
        "medium": "中等界面：常规工程纤维-浆体粘结，桥联稳定但仍有优化空间。",
        "weak": "弱界面：易早期脱粘，裂缝宽度与密度控制偏弱。",
    }[bond]
    cards.append(
        {
            "title": "界面协同 · ITZ/拉拔",
            "body": bond_txt,
            "effects": "裂缝扩展↓ · 裂缝密度↓ · 桥联效率↑",
        }
    )

    disp_txt = {
        "good": "分散良好：纤维网络均匀，减少局部无纤维弱区与假性高风险。",
        "normal": "分散一般：可能存在轻微不均匀，宜结合施工与振捣复核。",
        "poor": "团聚风险：局部纤维堆叠与弱区并存，裂缝与温度应力易在薄弱带集中。",
    }[disp]
    cards.append(
        {
            "title": "分散质量 · 均匀性",
            "body": disp_txt,
            "effects": "裂缝密度↓ · 假性高风险↓ · 温度应力均匀化",
        }
    )
    return cards
