"""合并用户试验数据与文献结构化数据，统一 source_doi 占位与列顺序。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS
from src.literature_pipeline.constants import (
    LABEL_META_COLUMNS,
    PROVENANCE_COLUMNS,
    QUALITY_COLUMNS,
    TARGET_COLUMNS,
    USER_SOURCE_DOI_PLACEHOLDER,
    USER_SOURCE_GROUP_DEFAULT,
)


def _all_output_columns() -> list[str]:
    """训练用列 + 溯源 + 标签元数据 + 质量（特征列只出现一次）。"""
    extra = list(PROVENANCE_COLUMNS) + list(LABEL_META_COLUMNS) + list(QUALITY_COLUMNS)
    # 去重保持顺序
    seen: set[str] = set()
    out: list[str] = []
    for c in list(FEATURE_COLUMNS) + list(TARGET_COLUMNS) + extra:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def merge_user_and_literature(
    user_csv: Path,
    literature_csv: Path,
    *,
    user_doi: str = USER_SOURCE_DOI_PLACEHOLDER,
) -> pd.DataFrame:
    """
    读取用户表与文献表；为用户表各行填充 source_doi=USER_LOCAL_LAB（若缺）。
    列并集对齐；缺失列填 NaN。
    """
    u = pd.read_csv(user_csv)
    l = pd.read_csv(literature_csv)
    if "source_doi" not in u.columns:
        u["source_doi"] = user_doi
    else:
        u["source_doi"] = u["source_doi"].fillna(user_doi)

    # —— source_group：文献缺省用 source_doi；用户缺省用 USER_LOCAL_LAB/DEFAULT 或保留 BATCH_* ——
    ld = l["source_doi"].astype(str).str.strip()
    if "source_group" not in l.columns:
        l["source_group"] = ld
    else:
        lg = l["source_group"].astype(str).str.strip()
        le = lg.isna() | (lg == "") | (lg == "nan")
        l.loc[le, "source_group"] = ld[le]

    if "source_group" not in u.columns:
        u["source_group"] = USER_SOURCE_GROUP_DEFAULT
    else:
        ug = u["source_group"].astype(str).str.strip()
        ue = ug.isna() | (ug == "") | (ug == "nan")
        u.loc[ue, "source_group"] = USER_SOURCE_GROUP_DEFAULT

    merged = pd.concat([u, l], ignore_index=True)
    for c in _all_output_columns():
        if c not in merged.columns:
            merged[c] = np.nan
    # 按标准列顺序输出（多出的列放后）
    ordered = [c for c in _all_output_columns() if c in merged.columns]
    rest = [c for c in merged.columns if c not in ordered]
    return merged[ordered + rest]


def save_merged(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
