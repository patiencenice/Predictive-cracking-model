"""
为 lab_strength 训练 CSV 补齐减水剂扩展六列（与 lab_mix_extra_row_vector 一致），再跑 training_data_gate。

water_reduction_rate_pct=-1 表示未知占位，不是真实 0%%；减水率未知时不计算 adjusted_w_b_ratio（-1 + missing_flag=1）。

用法（在 SteelFiberCrackPredictor 目录下）:
  py scripts/prepare_lab_strength_missing_columns.py --in data/lab_strength_training_merged.csv
  py scripts/prepare_lab_strength_missing_columns.py --in data/a.csv --out data/b.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.lab_strength_residual.lab_mix_features import ensure_lab_mix_extra_columns_in_dataframe
from src.lab_strength_residual.provenance_columns import ensure_optional_trace_columns
from src.lab_strength_residual.training_data_gate import validate_for_lab_strength_training


def main() -> None:
    ap = argparse.ArgumentParser(description="补齐 lab_strength 减水剂六列并校验闸门")
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="输入 CSV")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 CSV（默认覆盖 --in）",
    )
    ap.add_argument("--encoding", type=str, default="utf-8-sig", help="读写编码")
    args = ap.parse_args()
    out_path = args.out or args.inp

    df = pd.read_csv(args.inp, encoding=args.encoding)
    df = ensure_lab_mix_extra_columns_in_dataframe(df)
    df = ensure_optional_trace_columns(df)
    gate = validate_for_lab_strength_training(df)
    if not gate["ok"]:
        print("闸门未通过:")
        for pr in gate.get("problems") or []:
            print(" -", pr)
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding=args.encoding)
    print("已写入:", out_path.resolve())
    print("行数:", len(df))


if __name__ == "__main__":
    main()
