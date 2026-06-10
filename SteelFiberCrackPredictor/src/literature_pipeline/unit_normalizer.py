"""单位归一：将文献中可能出现的其他单位换算为训练统一单位。"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 列名后缀约定：如 crack_width_um 表示微米，本模块可根据元数据列 unit_crack_width 换算


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[name], errors="coerce")


def normalize_crack_width_mm(df: pd.DataFrame) -> pd.DataFrame:
    """
    crack_width 统一到 mm。
    - 可选列 crack_width_um：微米，直接 /1000 合并进 crack_width。
    - 可选列 unit_crack_width：mm | um | cm，对 crack_width 列做换算。
    """
    out = df.copy()
    w = _col(out, "crack_width")
    if "crack_width_um" in out.columns:
        w = w.fillna(_col(out, "crack_width_um") / 1000.0)
    if "unit_crack_width" in out.columns:
        u = out["unit_crack_width"].astype(str).str.lower().str.strip()
        w = np.where(u == "um", w / 1000.0, w)
        w = np.where(u == "cm", w * 10.0, w)
    out["crack_width"] = w
    return out


def normalize_crack_density_per_m2(df: pd.DataFrame) -> pd.DataFrame:
    """
    crack_density 统一到 条/m²。
    若 unit_crack_density 为 per_cm2，则 ×1e4。
    """
    out = df.copy()
    if "crack_density" not in out.columns:
        out["crack_density"] = np.nan
        return out
    d = _col(out, "crack_density")
    if "unit_crack_density" in out.columns:
        u = out["unit_crack_density"].astype(str).str.lower().str.strip()
        d = d.where(~u.str.contains("cm2", na=False), d * 10000.0)
    out["crack_density"] = d
    return out


def normalize_strength_mpa(df: pd.DataFrame) -> pd.DataFrame:
    """cube_strength_mpa、tensile_strength 等到 MPa（若标注为 GPa 则 ×1000）。"""
    out = df.copy()
    for col in ("cube_strength_mpa", "tensile_strength"):
        if col not in out.columns:
            continue
        v = _col(out, col)
        ucol = f"unit_{col}"
        if ucol in out.columns:
            u = out[ucol].astype(str).str.lower()
            v = v.mask(u.str.contains("gpa"), v * 1000.0)
        out[col] = v
    return out


def normalize_binder_kg_m3(df: pd.DataFrame) -> pd.DataFrame:
    """胶材、水泥、粉煤灰、矿粉、水、砂石等到 kg/m³（若给 g/L 则数值相同量级需注意，此处仅处理明显标记）。"""
    out = df.copy()
    cols = [
        "binder_content",
        "cement_content",
        "fly_ash",
        "slag_powder",
        "mixing_water",
        "sand_content",
        "stone_content",
    ]
    for col in cols:
        if col not in out.columns:
            continue
        v = _col(out, col)
        ucol = f"unit_{col}"
        if ucol in out.columns:
            u = out[ucol].astype(str).str.lower()
            # kg/L ≈ 1000 × 体积浓度；文献极少如此标注，保留占位
            v = v.mask(u.str.contains("kg/l"), v * 1000.0)
        out[col] = v
    return out


def normalize_curing_days_d(df: pd.DataFrame) -> pd.DataFrame:
    """curing_days 统一到 d；若给 hours 列则 /24。"""
    out = df.copy()
    if "curing_days" in out.columns:
        out["curing_days"] = pd.to_numeric(out["curing_days"], errors="coerce")
    if "curing_hours" in out.columns:
        h = _col(out, "curing_hours")
        if "curing_days" in out.columns:
            out["curing_days"] = out["curing_days"].fillna(h / 24.0)
        else:
            out["curing_days"] = h / 24.0
    return out


def normalize_all(df: pd.DataFrame) -> pd.DataFrame:
    """按顺序应用各类单位换算。"""
    x = normalize_strength_mpa(df)
    x = normalize_binder_kg_m3(x)
    x = normalize_curing_days_d(x)
    x = normalize_crack_width_mm(x)
    x = normalize_crack_density_per_m2(x)
    return x
