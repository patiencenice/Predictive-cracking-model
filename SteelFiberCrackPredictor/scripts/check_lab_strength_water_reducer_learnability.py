"""
只读：对 lab_strength 训练 CSV 做减水剂类型/减水率可学习性检查，写出轻量 JSON。

用法（在 SteelFiberCrackPredictor 目录下）:
  py scripts/check_lab_strength_water_reducer_learnability.py --in data/lab_strength_training_merged.csv
  py scripts/check_lab_strength_water_reducer_learnability.py --in data/a.csv --out outputs/wr_learnability.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.lab_strength_residual.water_reducer_learnability import (
    build_water_reducer_learnability_report,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="减水剂字段可学习性只读检查")
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="训练 CSV")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON 输出路径（默认打印到 stdout）",
    )
    ap.add_argument("--encoding", type=str, default="utf-8-sig")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, encoding=args.encoding)
    rep = build_water_reducer_learnability_report(df)
    rep["input_csv"] = str(args.inp.resolve())

    text = json.dumps(rep, ensure_ascii=False, indent=2)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print("已写入:", args.out.resolve())


if __name__ == "__main__":
    main()
