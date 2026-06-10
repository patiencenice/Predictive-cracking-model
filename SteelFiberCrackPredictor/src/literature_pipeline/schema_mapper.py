"""字段映射：将论文表格中的异名列映射到本系统 FEATURE_COLUMNS + 目标 + 溯源列。"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, FIBER_MATERIAL_MAP, FIBER_TYPE_MAP
from src.literature_pipeline.constants import (
    LABEL_META_COLUMNS,
    PROVENANCE_COLUMNS,
    TARGET_COLUMNS,
)

# 常见异名 -> 标准列名（小写匹配）
COLUMN_ALIASES: dict[str, str] = {
    # 纤维
    "vf": "fiber_content",
    "vol": "fiber_content",
    "fiber_vol": "fiber_content",
    "fiber volume": "fiber_content",
    "体积掺量": "fiber_content",
    "钢纤维掺量": "fiber_content",
    "l/d": "aspect_ratio",
    "aspect ratio": "aspect_ratio",
    "长径比": "aspect_ratio",
    "fu": "tensile_strength",
    "fts": "tensile_strength",
    "抗拉强度": "tensile_strength",
    "fiber tensile": "tensile_strength",
    # 强度 / 胶材
    "fcu": "cube_strength_mpa",
    "fc": "cube_strength_mpa",
    "cube strength": "cube_strength_mpa",
    "binder": "binder_content",
    "胶材": "binder_content",
    "cement": "cement_content",
    "水泥": "cement_content",
    "fa": "fly_ash",
    "粉煤灰": "fly_ash",
    "ggbs": "slag_powder",
    "矿粉": "slag_powder",
    "water": "mixing_water",
    "用水量": "mixing_water",
    "w/b": "w_b_ratio",
    "w/c": "w_b_ratio",
    "水胶比": "w_b_ratio",
    "sand": "sand_content",
    "细骨料": "sand_content",
    "sand ratio": "sand_ratio",
    "砂率": "sand_ratio",
    "coarse": "stone_content",
    "粗骨料": "stone_content",
    # 环境
    "t": "temperature",
    "temp": "temperature",
    "rh": "humidity",
    "湿度": "humidity",
    "age": "curing_days",
    "龄期": "curing_days",
    # 目标
    "wmax": "crack_width",
    "crack width": "crack_width",
    "max crack": "crack_width",
    "最大裂缝宽度": "crack_width",
    "crack density": "crack_density",
    "裂缝密度": "crack_density",
    "risk": "cracking_risk",
    "开裂等级": "cracking_risk",
    # 溯源
    "doi": "source_doi",
    "table": "source_table",
    "fig": "source_figure",
    "group": "source_group",
    "batch": "source_group",
    "source_batch": "source_group",
}


def _normalize_col_name(c: str) -> str:
    s = str(c).strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s


def map_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """将原始列名映射为标准名；未识别的列保留原名（后续可人工补映射）。"""
    rename: dict[str, str] = {}
    for c in raw.columns:
        key = _normalize_col_name(c)
        if key in COLUMN_ALIASES:
            rename[c] = COLUMN_ALIASES[key]
            continue
        # 直接等于标准名
        if key in FEATURE_COLUMNS or key in TARGET_COLUMNS or key in PROVENANCE_COLUMNS:
            rename[c] = key
            continue
        if key in LABEL_META_COLUMNS:
            rename[c] = key
            continue
        # 中文括号等
        for alias, std in COLUMN_ALIASES.items():
            if alias in key.replace(" ", "_"):
                rename[c] = std
                break
    return raw.rename(columns=rename)


def _parse_fiber_type_material(df: pd.DataFrame) -> pd.DataFrame:
    """若存在文本列 fiber_type / fiber_material，则编码为 *_enc。"""
    out = df.copy()
    if "fiber_type" in out.columns and "fiber_type_enc" not in out.columns:
        out["fiber_type_enc"] = out["fiber_type"].map(
            lambda x: FIBER_TYPE_MAP.get(str(x).strip(), 0)
            if pd.notna(x)
            else 0
        )
    if "fiber_material" in out.columns and "fiber_material_enc" not in out.columns:
        out["fiber_material_enc"] = out["fiber_material"].map(
            lambda x: FIBER_MATERIAL_MAP.get(str(x).strip(), 0)
            if pd.notna(x)
            else 0
        )
    return out


def fill_feature_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    文献中常缺若干编码列：用中性默认值填充，避免无法与推理维数对齐。
    用户应在 quality_scoring 中因「缺配合比」被降权。
    """
    out = df.copy()
    defaults: dict[str, float] = {
        "fiber_type_enc": 0.0,
        "fiber_material_enc": 0.0,
        "strength_grade_enc": 3.0,  # 默认 C30（与 STRENGTH_GRADE_ORDER 索引一致）
        "concrete_type_enc": 0.0,
        "slag_grade_enc": 1.0,
        "sand_type_enc": 0.0,
        "admixture_enc": 0.0,
        "casting_method_enc": 0.0,
        "admixture_dosage": 0.0,
    }
    for k, v in defaults.items():
        if k not in out.columns:
            out[k] = v
        else:
            out[k] = pd.to_numeric(out[k], errors="coerce").fillna(v)
    # cube_strength_mpa：缺省或由 enc 反推
    from src.features import STRENGTH_GRADE_ORDER, STRENGTH_GRADE_TO_MPA

    enc = out["strength_grade_enc"].astype(int).clip(0, len(STRENGTH_GRADE_ORDER) - 1)
    grades = [STRENGTH_GRADE_ORDER[int(i)] for i in enc.to_numpy()]
    filled = pd.Series(
        [float(STRENGTH_GRADE_TO_MPA[g]) for g in grades], index=out.index
    )
    if "cube_strength_mpa" not in out.columns:
        out["cube_strength_mpa"] = filled
    else:
        out["cube_strength_mpa"] = pd.to_numeric(
            out["cube_strength_mpa"], errors="coerce"
        ).fillna(filled)
    return out


def compute_interaction_feature(df: pd.DataFrame) -> pd.DataFrame:
    """与线上一致：fiber_content_x_aspect_ratio = fiber_content * aspect_ratio。"""
    out = df.copy()
    fc = pd.to_numeric(out["fiber_content"], errors="coerce")
    ar = pd.to_numeric(out["aspect_ratio"], errors="coerce")
    out["fiber_content_x_aspect_ratio"] = fc * ar
    return out


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """输出含 FEATURE_COLUMNS + 目标 + 溯源等完整列的 DataFrame。"""
    m = map_columns(df)
    m = _parse_fiber_type_material(m)
    m = fill_feature_defaults(m)
    m = compute_interaction_feature(m)
    # 确保 FEATURE_COLUMNS 均存在
    for c in FEATURE_COLUMNS:
        if c not in m.columns:
            m[c] = np.nan
    return m


def drop_rows_missing_critical_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    缺失严重或标签不清：剔除 crack_width 为空、或 crack_width_definition_id 缺失且未标记默认口径的行。
    返回 (清洗后表, 剔除原因列表)。
    """
    reasons: list[str] = []
    out = df.copy()
    if "crack_width" not in out.columns:
        return out.iloc[0:0], ["无 crack_width 列，全部丢弃"]

    m = pd.to_numeric(out["crack_width"], errors="coerce")
    bad = m.isna()
    n_bad = int(bad.sum())
    if n_bad:
        reasons.append(f"剔除 crack_width 无法转为数值的 {n_bad} 行")
    out = out.loc[~bad].copy()

    # 若定义 id 全空：在 label_standardizer 中会填默认，此处不删
    return out, reasons
