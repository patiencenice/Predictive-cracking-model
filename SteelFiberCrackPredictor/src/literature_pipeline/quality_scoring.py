"""样本质量分：依据可追溯性、配合比完整性、单位与试验方法明确性，输出 0~1 分与 sample_weight。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.literature_pipeline.constants import USER_SOURCE_DOI_PLACEHOLDER


def score_row(
    row: pd.Series,
    *,
    peer_reviewed_hint: bool | None = None,
) -> float:
    """
    单条样本质量分（启发式）：
    - 有明确 DOI 且非占位：+0.2
    - 有 source_table 或 source_figure：+0.15
    - 配合比核心列（binder, w_b_ratio, fiber_content）齐全：+0.25
    - test_method / specimen_size 非空：+0.15
    - crack_width_definition_id 非 UNSPECIFIED：+0.15
    - peer_reviewed_hint（可选元数据列 peer_reviewed=1）：+0.1
    封顶 1.0。
    """
    s = 0.0
    doi = str(row.get("source_doi", "") or "").strip()
    if doi and doi != USER_SOURCE_DOI_PLACEHOLDER:
        s += 0.2
    elif doi == USER_SOURCE_DOI_PLACEHOLDER:
        s += 0.15  # 自有试验仍可信，略低于有 DOI 文献

    st = str(row.get("source_table", "") or "").strip()
    sf = str(row.get("source_figure", "") or "").strip()
    if st or sf:
        s += 0.15

    need = ["binder_content", "w_b_ratio", "fiber_content"]
    ok = sum(
        1
        for k in need
        if k in row.index and pd.notna(row[k]) and str(row[k]).strip() != ""
    )
    if ok == len(need):
        s += 0.25
    elif ok >= 2:
        s += 0.12

    if str(row.get("test_method", "") or "").strip():
        s += 0.075
    if str(row.get("specimen_size", "") or "").strip():
        s += 0.075

    cid = str(row.get("crack_width_definition_id", "") or "")
    if cid and cid != "CW_UNSPECIFIED":
        s += 0.15

    if peer_reviewed_hint is True:
        s += 0.1
    elif "peer_reviewed" in row.index:
        try:
            if int(float(row["peer_reviewed"])) == 1:
                s += 0.1
        except (TypeError, ValueError):
            pass

    return float(min(1.0, s))


def add_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """写入 source_quality_score 与 sample_weight（与质量分线性相关，便于 XGBoost）。"""
    out = df.copy()
    scores = [score_row(out.iloc[i]) for i in range(len(out))]
    out["source_quality_score"] = scores
    # 权重：0.3~1.0，避免 0 导致样本被完全忽略
    out["sample_weight"] = np.clip(0.3 + 0.7 * out["source_quality_score"], 0.3, 1.0)
    return out


def build_data_quality_report(
    df: pd.DataFrame,
    drop_reasons: list[str],
    *,
    definition_filter_stats: dict[str, Any] | None = None,
    mixed_definition_warning: str | None = None,
) -> dict:
    """生成 data_quality_report.json 摘要。"""
    rep: dict[str, Any] = {
        "n_rows": int(len(df)),
        "mean_quality": float(df["source_quality_score"].mean())
        if "source_quality_score" in df.columns and len(df)
        else None,
        "drop_reasons": drop_reasons,
        "crack_width_definition_id_counts": {},
    }
    if "crack_width_definition_id" in df.columns:
        rep["crack_width_definition_id_counts"] = {
            str(k): int(v)
            for k, v in df["crack_width_definition_id"]
            .fillna("CW_UNSPECIFIED")
            .astype(str)
            .str.strip()
            .replace("", "CW_UNSPECIFIED")
            .value_counts()
            .items()
        }
    if definition_filter_stats:
        rep["crack_width_family_filter"] = definition_filter_stats.get("family")
        rep["n_samples_before_definition_filter"] = definition_filter_stats.get(
            "n_before"
        )
        rep["n_samples_after_definition_filter"] = definition_filter_stats.get(
            "n_after"
        )
        rep["dropped_by_definition_filter"] = definition_filter_stats.get("dropped", 0)
        rep["crack_width_definition_id_counts_before_filter"] = definition_filter_stats.get(
            "counts_before", {}
        )
        rep["crack_width_definition_id_counts_after_filter"] = definition_filter_stats.get(
            "counts_after", {}
        )
    if mixed_definition_warning:
        rep["mixed_definition_warning"] = mixed_definition_warning
    return rep
