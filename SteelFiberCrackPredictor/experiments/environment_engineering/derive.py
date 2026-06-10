"""
工程开裂环境场 + 施工工艺场 — Phase 1 派生（解释层，不进 FEATURE_COLUMNS）。
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from experiments.thermal_stress.derive import _series_float, _series_str_lower

_SOLAR_NUMERIC = {"low": 1.0, "medium": 2.0, "high": 3.0}
_CURING_SCORE = {
    "natural": 0.50,
    "water": 0.82,
    "membrane": 0.88,
    "steam": 0.94,
    "sealed": 0.86,
}
_VIBRATION_SCORE = {"poor": 0.35, "normal": 0.65, "good": 0.92}
_BLEEDING_SCORE = {"low": 0.25, "medium": 0.55, "high": 0.85}
_SEASON_THERMAL = {"spring": 0.55, "summer": 0.90, "autumn": 0.60, "winter": 0.85}
_MEMBER_THERMAL = {
    "slab": 0.55,
    "wall": 0.65,
    "beam": 0.50,
    "column": 0.45,
    "mass_concrete": 0.95,
}

_CURING_ZH = {
    "natural": "自然养护",
    "water": "洒水养护",
    "membrane": "覆膜养护",
    "steam": "蒸汽养护",
    "sealed": "密封养护",
}
_SOLAR_ZH = {"low": "阴凉", "medium": "普通日晒", "high": "强暴晒"}
_MEMBER_ZH = {
    "slab": "板",
    "wall": "墙",
    "beam": "梁",
    "column": "柱",
    "mass_concrete": "大体积混凝土",
}


def _clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _level_zh(score: float) -> str:
    if score >= 0.72:
        return "偏高"
    if score >= 0.45:
        return "中等"
    return "偏低"


def _adequacy_zh(score: float) -> str:
    if score >= 0.80:
        return "较充分"
    if score >= 0.55:
        return "一般"
    return "偏弱"


def derive_environment_engineering_features(row: pd.Series) -> dict[str, Any]:
    """由侧栏输入派生环境/工艺解释量；不修改 row。"""
    out: dict[str, Any] = {}

    temp = _series_float(row, "temperature")
    humidity = _series_float(row, "humidity")
    wind = _series_float(row, "wind_speed_ms")
    day_night = _series_float(row, "day_night_temp_diff_c")

    out["wind_speed_ms"] = wind
    out["day_night_temp_diff_c"] = day_night
    out["wind_speed_ms_missing_flag"] = 0 if wind is not None else 1
    out["day_night_temp_diff_c_missing_flag"] = 0 if day_night is not None else 1

    solar = _series_str_lower(row, "solar_exposure_level") or "medium"
    if solar not in _SOLAR_NUMERIC:
        solar = "medium"
        out["solar_exposure_level_illegal_flag"] = 1
    else:
        out["solar_exposure_level_illegal_flag"] = 0
    out["solar_exposure_level"] = solar

    curing = _series_str_lower(row, "curing_method") or "water"
    if curing not in _CURING_SCORE:
        curing = "water"
        out["curing_method_illegal_flag"] = 1
    else:
        out["curing_method_illegal_flag"] = 0
    out["curing_method"] = curing

    vibration = _series_str_lower(row, "vibration_quality") or "normal"
    if vibration not in _VIBRATION_SCORE:
        vibration = "normal"
        out["vibration_quality_illegal_flag"] = 1
    else:
        out["vibration_quality_illegal_flag"] = 0
    out["vibration_quality"] = vibration

    bleeding = _series_str_lower(row, "bleeding_risk") or "medium"
    if bleeding not in _BLEEDING_SCORE:
        bleeding = "medium"
        out["bleeding_risk_illegal_flag"] = 1
    else:
        out["bleeding_risk_illegal_flag"] = 0
    out["bleeding_risk"] = bleeding

    season = _series_str_lower(row, "construction_season") or "spring"
    if season not in _SEASON_THERMAL:
        season = "spring"
        out["construction_season_illegal_flag"] = 1
    else:
        out["construction_season_illegal_flag"] = 0
    out["construction_season"] = season

    member = _series_str_lower(row, "member_type") or "slab"
    if member not in _MEMBER_THERMAL:
        member = "slab"
        out["member_type_illegal_flag"] = 1
    else:
        out["member_type_illegal_flag"] = 0
    out["member_type"] = member

    dryness = None
    if humidity is not None:
        dryness = _clamp01(1.0 - float(humidity) / 100.0)
    out["humidity_dryness_factor"] = dryness

    evap: float | None = None
    if temp is not None and wind is not None and dryness is not None:
        evap = float(temp) * float(wind) * dryness
    out["evaporation_risk_index"] = evap if evap is not None else -1.0
    out["evaporation_risk_index_missing_flag"] = 0 if evap is not None else 1

    solar_n = _SOLAR_NUMERIC[solar]
    tgrad: float | None = None
    if day_night is not None:
        tgrad = float(day_night) * solar_n
    out["thermal_gradient_risk"] = tgrad if tgrad is not None else -1.0
    out["thermal_gradient_risk_missing_flag"] = 0 if tgrad is not None else 1

    shrink: float | None = None
    if wind is not None and temp is not None and dryness is not None:
        wind_n = _clamp01(float(wind) / 15.0)
        temp_n = _clamp01((float(temp) - 5.0) / 35.0)
        shrink = float(wind_n + temp_n + dryness)
    out["surface_shrinkage_risk"] = shrink if shrink is not None else -1.0
    out["surface_shrinkage_risk_missing_flag"] = 0 if shrink is not None else 1

    evap_score = _clamp01((evap or 0.0) / 120.0) if evap is not None else None
    tgrad_score = _clamp01((tgrad or 0.0) / 25.0) if tgrad is not None else None
    shrink_score = _clamp01((shrink or 0.0) / 2.5) if shrink is not None else None

    curing_days = _series_float(row, "curing_days")
    curing_base = _CURING_SCORE[curing]
    if curing_days is not None and curing_days >= 14:
        curing_base = min(1.0, curing_base + 0.06)
    elif curing_days is not None and curing_days < 7:
        curing_base = max(0.0, curing_base - 0.12)

    surface_tendency = None
    if shrink_score is not None:
        surface_tendency = _clamp01(
            0.45 * shrink_score
            + 0.25 * _BLEEDING_SCORE[bleeding]
            + 0.15 * (1.0 - _VIBRATION_SCORE[vibration])
            + 0.15 * (evap_score or 0.0)
        )

    thermal_tendency = None
    if tgrad_score is not None:
        thermal_tendency = _clamp01(
            0.50 * tgrad_score
            + 0.25 * _MEMBER_THERMAL[member]
            + 0.15 * _SEASON_THERMAL[season]
            + 0.10 * (1.0 - curing_base)
        )

    out["evaporation_risk_score"] = round(evap_score, 4) if evap_score is not None else None
    out["thermal_gradient_risk_score"] = (
        round(tgrad_score, 4) if tgrad_score is not None else None
    )
    out["surface_shrinkage_risk_score"] = (
        round(shrink_score, 4) if shrink_score is not None else None
    )
    out["curing_adequacy_score"] = round(curing_base, 4)

    out["environment_engineering_summary"] = _build_summary(
        evap_score,
        tgrad_score,
        curing_base,
        surface_tendency,
        thermal_tendency,
        curing,
        solar,
        member,
    )
    out["environment_driver_bullets_zh"] = _driver_bullets(
        temp,
        humidity,
        wind,
        day_night,
        solar,
        curing,
        vibration,
        bleeding,
        evap_score,
        tgrad_score,
        shrink_score,
    )
    out["environment_thermal_linkage_note_zh"] = _thermal_linkage_note(
        day_night, solar, member
    )

    return out


def _build_summary(
    evap_score: float | None,
    tgrad_score: float | None,
    curing_score: float,
    surface_tendency: float | None,
    thermal_tendency: float | None,
    curing: str,
    solar: str,
    member: str,
) -> dict[str, str]:
    return {
        "evaporation_risk_zh": (
            f"蒸发风险：{_level_zh(evap_score)}"
            if evap_score is not None
            else "蒸发风险：—"
        ),
        "thermal_gradient_risk_zh": (
            f"温差风险：{_level_zh(tgrad_score)}"
            if tgrad_score is not None
            else "温差风险：—"
        ),
        "curing_adequacy_zh": f"养护充分性：{_adequacy_zh(curing_score)}（{_CURING_ZH[curing]}）",
        "surface_crack_tendency_zh": (
            f"表层开裂倾向：{_level_zh(surface_tendency)}"
            if surface_tendency is not None
            else "表层开裂倾向：—"
        ),
        "thermal_crack_tendency_zh": (
            f"热裂缝倾向：{_level_zh(thermal_tendency)}"
            if thermal_tendency is not None
            else "热裂缝倾向：—"
        ),
        "member_type_zh": _MEMBER_ZH.get(member, member),
        "solar_exposure_zh": _SOLAR_ZH.get(solar, solar),
        "surface_moisture_loss_zh": (
            f"表层失水风险：{_level_zh(evap_score)}"
            if evap_score is not None
            else "表层失水风险：—"
        ),
    }


def _driver_bullets(
    temp: float | None,
    humidity: float | None,
    wind: float | None,
    day_night: float | None,
    solar: str,
    curing: str,
    vibration: str,
    bleeding: str,
    evap_score: float | None,
    tgrad_score: float | None,
    shrink_score: float | None,
) -> list[str]:
    bullets: list[str] = []

    if day_night is not None and tgrad_score is not None:
        bullets.append(
            f"温度梯度：昼夜温差 ΔT≈{day_night:.1f}°C，"
            f"暴晒等级「{_SOLAR_ZH.get(solar, solar)}」，"
            f"表层—内部温差与热胀冷缩循环风险{_level_zh(tgrad_score)}。"
        )
    if humidity is not None and curing:
        bullets.append(
            f"水分迁移：环境 RH≈{humidity:.0f}%，养护方式为「{_CURING_ZH.get(curing, curing)}」，"
            "影响早期失水速率与表层干燥收缩。"
        )
    if wind is not None and temp is not None and evap_score is not None:
        bullets.append(
            f"蒸发：T≈{temp:.1f}°C、风速≈{wind:.1f} m/s，"
            f"表面蒸发与塑性期失水风险{_level_zh(evap_score)}。"
        )
    if shrink_score is not None:
        bullets.append(
            f"塑性收缩：温—风—湿度组合下塑性期体积收缩与表层拉应力风险{_level_zh(shrink_score)}。"
        )
    if vibration != "good" or bleeding != "low":
        vib_txt = {"poor": "振捣不足", "normal": "振捣一般", "good": "振捣良好"}[vibration]
        bleed_txt = {"low": "低", "medium": "中", "high": "高"}[bleeding]
        bullets.append(
            f"表层弱区：{vib_txt}、泌水倾向{bleed_txt}，"
            "可能形成孔隙/泌水弱层，塑性裂缝与表层拉应力更易发展。"
        )
    else:
        bullets.append(
            "表层拉应力：干燥—温变耦合下，表层混凝土抗拉能力尚低时，"
            "拉应力易在表面集中并诱发微裂缝。"
        )
    return bullets


def _thermal_linkage_note(
    day_night: float | None,
    solar: str,
    member: str,
) -> str | None:
    dt_large = day_night is not None and float(day_night) >= 8.0
    if dt_large and solar == "high" and member == "mass_concrete":
        return (
            "大体积构件内部散热受限，表里温差可能增大，"
            "温度拉应力风险提高。"
        )
    if dt_large and member == "mass_concrete":
        return (
            "大体积混凝土构件表里温差与约束应力的工程解释权重较高，"
            "宜结合温控与保温措施复核。"
        )
    if solar == "high" and day_night is not None and float(day_night) >= 6.0:
        return (
            "强暴晒叠加较大昼夜温差，表层温升与温度梯度显著，"
            "宜关注表层开裂与热裂缝风险。"
        )
    return None
