"""crack_width_definition_id 过滤与混训告警（与 label_standardizer 中定义 ID 一致）。"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from src.literature_pipeline.label_standardizer import CRACK_WIDTH_DEFINITIONS


def ensure_crack_width_definition_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """无该列或空值时补 CW_UNSPECIFIED，便于与用户旧表合并后再过滤。"""
    out = df.copy()
    if "crack_width_definition_id" not in out.columns:
        out["crack_width_definition_id"] = "CW_UNSPECIFIED"
    else:
        s = out["crack_width_definition_id"].fillna("CW_UNSPECIFIED").astype(str).str.strip()
        out["crack_width_definition_id"] = s.replace("", "CW_UNSPECIFIED")
    return out


def definition_id_counts(df: pd.DataFrame) -> dict[str, int]:
    """统计各 crack_width_definition_id 样本数（空视为 CW_UNSPECIFIED）。"""
    if "crack_width_definition_id" not in df.columns:
        return {}
    s = df["crack_width_definition_id"].fillna("CW_UNSPECIFIED").astype(str).str.strip()
    s = s.replace("", "CW_UNSPECIFIED")
    return {str(k): int(v) for k, v in s.value_counts().items()}


def filter_by_crack_width_family(
    df: pd.DataFrame,
    family: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    仅保留指定标签家族；family 为 None 时不删行。
    返回 (筛选后表, 统计信息)。
    """
    counts_before = definition_id_counts(df)
    n_before = int(len(df))
    if family is None or str(family).strip() == "":
        return df.copy(), {
            "family": None,
            "n_before": n_before,
            "n_after": n_before,
            "dropped": 0,
            "counts_before": counts_before,
            "counts_after": counts_before,
        }
    fam = str(family).strip()
    if "crack_width_definition_id" not in df.columns:
        raise ValueError("DataFrame 缺少 crack_width_definition_id 列，无法按家族过滤")
    col = df["crack_width_definition_id"].fillna("CW_UNSPECIFIED").astype(str).str.strip()
    col = col.replace("", "CW_UNSPECIFIED")
    mask = col == fam
    out = df.loc[mask].copy()
    counts_after = definition_id_counts(out)
    return out, {
        "family": fam,
        "n_before": n_before,
        "n_after": int(len(out)),
        "dropped": n_before - int(len(out)),
        "counts_before": counts_before,
        "counts_after": counts_after,
    }


def mixed_definition_warning_text(counts: dict[str, int]) -> str | None:
    """若存在多个不同 definition_id，返回强警告文案；否则 None。"""
    if len(counts) <= 1:
        return None
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    return (
        "【强警告】检测到多个 crack_width_definition_id 同时存在（混训风险）："
        + ", ".join(parts)
        + "。请使用 --crack-width-family <ID> 仅保留单一测量口径，或拆分数据后再训练/交叉验证。"
    )


def emit_mixed_warning_if_needed(counts: dict[str, int], *, stream=sys.stderr) -> None:
    """未指定过滤且多定义时打印强警告。"""
    msg = mixed_definition_warning_text(counts)
    if msg:
        print("\n" + "=" * 72, file=stream)
        print(msg, file=stream)
        print("=" * 72 + "\n", file=stream)


def known_family_help() -> str:
    """合法定义 ID 列表说明（供 --help）。"""
    return ", ".join(sorted(CRACK_WIDTH_DEFINITIONS.keys()))
