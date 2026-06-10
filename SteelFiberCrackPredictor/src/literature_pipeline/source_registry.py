"""来源登记：从结构化表中汇总「文献/规范/数据库」级元数据，生成 source_provenance.csv。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.literature_pipeline.constants import PROVENANCE_COLUMNS


def aggregate_sources(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 source_doi 聚合：样本条数、表格/图引用列表、首条标题、**distinct source_group** 等。
    source_doi 级溯源逻辑不变；新增列便于核对「一文多组」交叉验证分组。
    """
    if "source_doi" not in df.columns:
        raise ValueError("DataFrame 须含 source_doi 列")

    def _join_unique(series: pd.Series) -> str:
        vals = [str(x).strip() for x in series.dropna().unique() if str(x).strip()]
        return "; ".join(sorted(set(vals)))[:2000]

    g = df.groupby("source_doi", dropna=False)
    rows: list[dict[str, Any]] = []
    for doi, part in g:
        row: dict[str, Any] = {
            "source_doi": doi,
            "n_rows": int(len(part)),
            "source_tables": _join_unique(part["source_table"])
            if "source_table" in part.columns
            else "",
            "source_figures": _join_unique(part["source_figure"])
            if "source_figure" in part.columns
            else "",
        }
        # 该 DOI 下出现过的所有 source_group（交叉验证分组粒度）
        if "source_group" in part.columns:
            row["source_groups"] = _join_unique(part["source_group"])
        else:
            row["source_groups"] = ""
        if "source_paper_title" in part.columns:
            row["source_paper_title"] = str(part["source_paper_title"].iloc[0])
        else:
            row["source_paper_title"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def enrich_row_level_notes(df: pd.DataFrame) -> pd.DataFrame:
    """为行级数据保留完整溯源列（若缺失则填空字符串）。"""
    out = df.copy()
    for c in PROVENANCE_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    return out


def save_source_provenance_csv(df: pd.DataFrame, out_path: Path) -> Path:
    """写出 source_provenance.csv（按 DOI 聚合）。"""
    agg = aggregate_sources(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path
