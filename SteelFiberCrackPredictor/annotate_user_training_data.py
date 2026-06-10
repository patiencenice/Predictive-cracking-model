"""
为用户训练数据批量补全 crack_width_definition_id / source_doi / source_group。

用法示例：
  py annotate_user_training_data.py ^
    --input data/training_data.csv ^
    --output data/training_data.annotated.csv ^
    --default-crack-width-family CW_MAX_SURFACE_MM ^
    --default-source-doi USER_LOCAL_LAB

可选：
  --source-group-cols batch,series,date
  当原表缺少 source_group 时，按这些列拼接成：
    <default_source_doi>/<batch>-<series>-<date>
  若不提供该参数，则固定填：
    <default_source_doi>/DEFAULT
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_group_cols(raw: str | None) -> list[str]:
    """解析 --source-group-cols，支持逗号分隔。"""
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _build_source_group_from_cols(
    df: pd.DataFrame,
    group_cols: list[str],
    default_source_doi: str,
) -> pd.Series:
    """
    按指定列拼接 source_group。
    例：USER_LOCAL_LAB/BATCH_001-SERIES_A-2026-04-01
    """
    if not group_cols:
        return pd.Series(
            [f"{default_source_doi}/DEFAULT"] * len(df),
            index=df.index,
            dtype="object",
        )

    # 缺失列时用占位 "NA"；不报错，保证批量可落地。
    parts: list[pd.Series] = []
    for c in group_cols:
        if c in df.columns:
            s = df[c].astype(str).str.strip().replace("", "NA")
        else:
            s = pd.Series(["NA"] * len(df), index=df.index, dtype="object")
        parts.append(s)

    merged = parts[0]
    for p in parts[1:]:
        merged = merged + "-" + p
    return default_source_doi + "/" + merged


def annotate_user_training_data_df(
    df: pd.DataFrame,
    *,
    default_crack_width_family: str,
    default_source_doi: str,
    source_group_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    核心补全逻辑（可被 run_literature_pipeline.py 直接 import 调用）。
    只补空值，不覆盖已有非空值。
    """
    out = df.copy()
    before_cols = set(out.columns)
    n_rows = len(out)
    has_col_cw = "crack_width_definition_id" in out.columns
    has_col_doi = "source_doi" in out.columns

    # 1) crack_width_definition_id：无列则新增，有空值则补默认值
    if not has_col_cw:
        out["crack_width_definition_id"] = default_crack_width_family
        used_default_cw = True
    else:
        before_cw = out["crack_width_definition_id"].fillna("").astype(str).str.strip()
        used_default_cw = bool(before_cw.eq("").any())
        s = (
            out["crack_width_definition_id"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", default_crack_width_family)
        )
        out["crack_width_definition_id"] = s

    # 2) source_doi：无列则新增，有空值则补默认值
    if not has_col_doi:
        out["source_doi"] = default_source_doi
        used_default_doi = True
    else:
        before_doi = out["source_doi"].fillna("").astype(str).str.strip()
        used_default_doi = bool(before_doi.eq("").any())
        s = (
            out["source_doi"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", default_source_doi)
        )
        out["source_doi"] = s

    # 3) source_group：无列则按策略生成；有列则仅补空值
    if "source_group" not in out.columns:
        out["source_group"] = _build_source_group_from_cols(
            df=out,
            group_cols=source_group_cols,
            default_source_doi=default_source_doi,
        )
        source_group_mode = (
            f"from_cols:{','.join(source_group_cols)}"
            if source_group_cols
            else "fixed_default"
        )
    else:
        current = out["source_group"].fillna("").astype(str).str.strip()
        empty = current.eq("")
        if empty.any():
            fallback = _build_source_group_from_cols(
                df=out,
                group_cols=source_group_cols,
                default_source_doi=default_source_doi,
            )
            current.loc[empty] = fallback.loc[empty]
        out["source_group"] = current
        source_group_mode = (
            f"fill_empty_from_cols:{','.join(source_group_cols)}"
            if source_group_cols
            else "fill_empty_fixed_default"
        )

    after_cols = set(out.columns)
    added_cols = sorted(list(after_cols - before_cols))
    sg_top20 = out["source_group"].value_counts().head(20).to_dict()
    default_group = f"{default_source_doi}/DEFAULT"
    default_group_count = int((out["source_group"] == default_group).sum())
    default_group_ratio = (default_group_count / n_rows) if n_rows > 0 else 0.0
    summary: dict[str, Any] = {
        "n_rows": int(n_rows),
        "added_columns": added_cols,
        "source_group_mode": source_group_mode,
        "used_default_crack_width_family": bool(used_default_cw),
        "used_default_source_doi": bool(used_default_doi),
        "crack_width_definition_id_counts": {
            str(k): int(v)
            for k, v in out["crack_width_definition_id"].value_counts().to_dict().items()
        },
        "source_doi_counts_top10": {
            str(k): int(v)
            for k, v in out["source_doi"].value_counts().head(10).to_dict().items()
        },
        "source_group_nunique": int(out["source_group"].nunique(dropna=True)),
        "source_group_counts_top20": {str(k): int(v) for k, v in sg_top20.items()},
        "default_group_label": default_group,
        "default_group_count": default_group_count,
        "default_group_ratio": float(default_group_ratio),
        "has_many_default_groups": bool(default_group_ratio >= 0.5),
    }
    return out, summary


def annotate_user_training_data_file(
    *,
    input_path: Path,
    output_path: Path,
    default_crack_width_family: str,
    default_source_doi: str,
    source_group_cols: list[str],
    allow_overwrite_output: bool = True,
) -> dict[str, Any]:
    """
    文件级封装：读取输入 CSV -> 补全 -> 写输出 CSV（不覆盖输入文件）。
    返回摘要信息，供上层脚本记录到 data_quality_report。
    """
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("为避免覆盖原文件，输出路径不能与输入路径相同")

    if output_path.exists() and not allow_overwrite_output:
        raise FileExistsError(f"输出文件已存在: {output_path}")

    df = pd.read_csv(input_path)
    annotated, summary = annotate_user_training_data_df(
        df,
        default_crack_width_family=default_crack_width_family,
        default_source_doi=default_source_doi,
        source_group_cols=source_group_cols,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="补全用户训练数据标签列")
    ap.add_argument("--input", type=Path, required=True, help="输入 CSV")
    ap.add_argument("--output", type=Path, required=True, help="输出 CSV（新文件）")
    ap.add_argument(
        "--default-crack-width-family",
        type=str,
        default="CW_MAX_SURFACE_MM",
        help="crack_width_definition_id 默认值",
    )
    ap.add_argument(
        "--default-source-doi",
        type=str,
        default="USER_LOCAL_LAB",
        help="source_doi 默认值",
    )
    ap.add_argument(
        "--source-group-cols",
        type=str,
        default="",
        help="当缺少 source_group 时，按这些列拼接生成（逗号分隔）",
    )
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"未找到输入文件: {args.input}")
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("为避免覆盖原文件，--output 不能与 --input 相同")

    group_cols = _parse_group_cols(args.source_group_cols)
    summary = annotate_user_training_data_file(
        input_path=args.input,
        output_path=args.output,
        default_crack_width_family=args.default_crack_width_family,
        default_source_doi=args.default_source_doi,
        source_group_cols=group_cols,
        allow_overwrite_output=True,
    )

    # 输出补全前后统计
    print("=== 注释补全统计 ===")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"总行数: {summary['n_rows']}")
    print(f"新增列: {summary['added_columns'] if summary['added_columns'] else '无'}")
    print(f"source_group 生成策略: {summary['source_group_mode']}")
    print("crack_width_definition_id 分布: " + str(summary["crack_width_definition_id_counts"]))
    print("source_doi 分布(Top10): " + str(summary["source_doi_counts_top10"]))
    print("source_group 分组数: " + str(summary["source_group_nunique"]))


if __name__ == "__main__":
    main()

