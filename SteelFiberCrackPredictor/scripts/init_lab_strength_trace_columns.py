"""
为 lab_strength 训练 CSV 追加可选追溯列（空值），不修改标签与特征数值。

用法（在 SteelFiberCrackPredictor 目录下）:
  py scripts/init_lab_strength_trace_columns.py --in data/lab_strength_training_merged.csv
  py scripts/init_lab_strength_trace_columns.py --in data/a.csv --out data/b.csv

说明见 data/lab_strength/LAB_STRENGTH_TRACING_SPEC.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.lab_strength_residual.provenance_columns import ensure_optional_trace_columns
from src.lab_strength_residual.training_data_gate import validate_for_lab_strength_training


def main() -> None:
    ap = argparse.ArgumentParser(description="追加 lab_strength 可选追溯列（空）")
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--encoding", type=str, default="utf-8-sig")
    args = ap.parse_args()
    out_path = args.out or args.inp

    df = pd.read_csv(args.inp, encoding=args.encoding)
    df = ensure_optional_trace_columns(df)
    gate = validate_for_lab_strength_training(df)
    if not gate["ok"]:
        print("闸门未通过:")
        for p in gate.get("problems") or []:
            print(" -", p)
        raise SystemExit(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding=args.encoding)
    print("已写入:", out_path.resolve(), "行数:", len(df))


if __name__ == "__main__":
    main()
