"""
将 lab_strength 训练 CSV 并入现有训练集（concat + 去重）。

基表/新表可缺少减水剂扩展六列；在闸门前会按 lab_mix_features.ensure_lab_mix_extra_columns_in_dataframe
逐行补齐（占位与 missing_flag 与 lab_mix_extra_row_vector 一致，不臆造真实减水率）。

合并时除硬列外对**扩展列取并集**（仅一侧存在的列另一侧补空），保留 needs_manual_review、追溯列等，见 provenance_columns / data/lab_strength/LAB_STRENGTH_TRACING_SPEC.md。

用法:
  py scripts/merge_lab_strength_training_csv.py --new path/prepared.csv \\
      --base data/lab_strength_training.csv --out data/lab_strength_training_merged.csv

若 --base 不存在，可改用 data/lab_strength_training.example.csv 作为起点（需自行确认）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.lab_strength_residual.lab_mix_features import (
    LAB_STRENGTH_FEATURE_COLUMNS,
    ensure_lab_mix_extra_columns_in_dataframe,
)
from src.lab_strength_residual.provenance_columns import ensure_optional_trace_columns
from src.lab_strength_residual.training_data_gate import validate_for_lab_strength_training

# 合并去重键与闸门硬列一致；其余列（如 needs_manual_review、追溯列）并集保留。
REQUIRED_MERGE_COLS: list[str] = list(LAB_STRENGTH_FEATURE_COLUMNS) + [
    "compressive_true",
    "flexural_true",
    "source_group",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="并入 lab_strength 训练 CSV")
    ap.add_argument("--new", type=Path, required=True, help="新增、已通过闸门的数据 CSV")
    ap.add_argument(
        "--base",
        type=Path,
        default=_ROOT / "data" / "lab_strength_training.csv",
        help="已有训练集 CSV",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出路径（默认覆盖 --base 所指文件）",
    )
    ap.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="读取 CSV 编码",
    )
    args = ap.parse_args()

    out_path = args.out or args.base

    base_path = args.base
    if not base_path.exists():
        print(f"基表不存在: {base_path}")
        print("请先准备基表，或显式指定 --base data/lab_strength_training.example.csv")
        raise SystemExit(2)

    new_df = pd.read_csv(args.new, encoding=args.encoding)
    base_df = pd.read_csv(base_path, encoding=args.encoding)

    new_df = ensure_lab_mix_extra_columns_in_dataframe(new_df)
    base_df = ensure_lab_mix_extra_columns_in_dataframe(base_df)
    new_df = ensure_optional_trace_columns(new_df)
    base_df = ensure_optional_trace_columns(base_df)

    g_new = validate_for_lab_strength_training(new_df)
    if not g_new["ok"]:
        print("新表未通过闸门，拒绝合并。问题:")
        for p in g_new["problems"]:
            print(" -", p)
        raise SystemExit(1)

    g_base = validate_for_lab_strength_training(base_df)
    if not g_base["ok"]:
        print("基表未通过闸门，拒绝合并。问题:")
        for p in g_base["problems"]:
            print(" -", p)
        raise SystemExit(1)

    extra_union = sorted(
        set(new_df.columns) | set(base_df.columns) - set(REQUIRED_MERGE_COLS)
    )
    for c in extra_union:
        if c not in new_df.columns:
            new_df[c] = pd.NA
        if c not in base_df.columns:
            base_df[c] = pd.NA
    cols = [c for c in REQUIRED_MERGE_COLS if c in new_df.columns and c in base_df.columns]
    cols = cols + [c for c in extra_union if c in new_df.columns and c in base_df.columns]

    a = new_df[cols].copy()
    b = base_df[cols].copy()
    merged = pd.concat([b, a], axis=0, ignore_index=True)
    dedup_on = [c for c in REQUIRED_MERGE_COLS if c in merged.columns]
    before = len(merged)
    merged = merged.drop_duplicates(subset=dedup_on, keep="first")
    after = len(merged)

    merged = ensure_optional_trace_columns(merged)

    g_merged = validate_for_lab_strength_training(merged)
    if not g_merged["ok"]:
        print("合并去重后的表未通过闸门，拒绝写盘。问题:")
        for p in g_merged.get("problems") or []:
            print(" -", p)
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("合并完成:", out_path.resolve())
    print(f"行数: 基 {len(b)} + 新 {len(a)} → 合并 {before} → 去重后 {after}")


if __name__ == "__main__":
    main()
