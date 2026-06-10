"""
lab_strength 训练 CSV 可选「追溯 / 分层」列：不参与闸门硬必需、不进入特征矩阵。

用途：文献键、表号、抽取批次、数据层级、行级审计 id；便于合并保留与报告汇总，不替代标签与 source_group。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# 与 data/lab_strength/LAB_STRENGTH_TRACING_SPEC.md 一致；仅追加列名，勿改既有训练列语义。
TRACING_COLUMN_NAMES: tuple[str, ...] = (
    "literature_key",  # 文献稳定键：DOI/内部编号等，非空时建议全局唯一或「论文+表」粒度
    "table_or_figure_ref",  # 原文定位：如 Table 3、Fig.2(b)
    "extraction_batch_id",  # 抽取/整理批次，便于复现与 diff
    "data_tier",  # 分层：建议 A_lab_native | B_literature_verified | C_literature_extracted | 空
    "row_uid",  # 可选：行级稳定 id（合并去重审计）；勿臆造，人工或上游脚本写入
)

# compressive 协议治理列（只做审计/筛选，不进特征矩阵）
COMPRESSIVE_PROTOCOL_GOV_COLUMNS: tuple[str, ...] = (
    "lab_specimen",  # 试件类型；建议与 lab_experiment.SPECIMEN_TYPES 枚举一致
    "lab_cube_edge_mm",  # 立方体边长 mm；150 为 strict 口径目标
    "lab_loading_compression",  # 抗压加载方式；建议与 lab_experiment.LOADING_COMPRESSION 一致
    "lab_curing_regime",  # 养护制度：standard | non_standard | unknown
    "lab_curing_note",  # 养护条件原文说明（追溯）
    "cube_strength_mpa_semantics",  # cube_strength_mpa 语义：fcu_k_design | cube_test_mean | cube_test_representative | unknown
    "cube_strength_mpa_source_note",  # cube_strength_mpa 来源/换算说明（追溯）
    "lab_protocol_closed_flag_compressive",  # compressive 协议闭合标识：1 闭合；0/空 未闭合
)


def ensure_optional_trace_columns(df: pd.DataFrame) -> pd.DataFrame:
    """为缺失的可选追溯列补空字符串（不写盘、不推断内容）。"""
    out = df.copy()
    for c in tuple(list(TRACING_COLUMN_NAMES) + list(COMPRESSIVE_PROTOCOL_GOV_COLUMNS)):
        if c not in out.columns:
            out[c] = ""
        else:
            out[c] = out[c].fillna("")
    return out


def summarize_tracing_columns(df: pd.DataFrame) -> dict[str, Any]:
    """供训练报告与闸门 warnings：仅统计，不改数据。"""
    present = [c for c in TRACING_COLUMN_NAMES if c in df.columns]
    out: dict[str, Any] = {
        "tracing_columns_defined": list(TRACING_COLUMN_NAMES),
        "compressive_protocol_gov_columns_defined": list(
            COMPRESSIVE_PROTOCOL_GOV_COLUMNS
        ),
        "tracing_columns_present_in_csv": present,
        "compressive_protocol_gov_columns_present_in_csv": [
            c for c in COMPRESSIVE_PROTOCOL_GOV_COLUMNS if c in df.columns
        ],
        "n_rows": int(len(df)),
    }
    if "data_tier" in df.columns:
        s = df["data_tier"].fillna("").astype(str).str.strip()
        s = s.replace("", "<empty>")
        vc = s.value_counts()
        out["data_tier_counts"] = {str(k): int(v) for k, v in vc.items()}
        out["n_rows_data_tier_nonempty"] = int((df["data_tier"].fillna("").astype(str).str.strip() != "").sum())
    else:
        out["data_tier_counts"] = {}
        out["n_rows_data_tier_nonempty"] = 0
    for c in ("literature_key", "table_or_figure_ref", "extraction_batch_id", "row_uid"):
        if c in df.columns:
            nn = df[c].fillna("").astype(str).str.strip() != ""
            out[f"n_rows_{c}_nonempty"] = int(nn.sum())
        else:
            out[f"n_rows_{c}_nonempty"] = 0
    for c in COMPRESSIVE_PROTOCOL_GOV_COLUMNS:
        if c in df.columns:
            nn = df[c].fillna("").astype(str).str.strip() != ""
            out[f"n_rows_{c}_nonempty"] = int(nn.sum())
        else:
            out[f"n_rows_{c}_nonempty"] = 0
    return out
