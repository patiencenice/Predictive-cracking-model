"""
对易梦论文表做 lab_strength_residual 训练前闸门检查（不训练）。

默认尝试读取（按顺序）:
  1) data/lab_strength/paper_yimeng_28d_final_for_lab_strength_residual.csv
  2) data/paper_yimeng_28d_final_for_lab_strength_residual.csv

编码依次尝试 utf-8-sig、utf-8、gbk、gb18030。

用法:
  py scripts/gate_paper_yimeng_lab_strength.py
  py scripts/gate_paper_yimeng_lab_strength.py --input path/to.csv
  py scripts/gate_paper_yimeng_lab_strength.py --out outputs/lab_strength/paper_yimeng_gate_report.json
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

from src.lab_strength_residual.training_data_gate import validate_for_lab_strength_training

DEFAULT_CANDIDATES = [
    _ROOT / "data" / "lab_strength" / "paper_yimeng_28d_final_for_lab_strength_residual.csv",
    _ROOT / "data" / "paper_yimeng_28d_final_for_lab_strength_residual.csv",
]

ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030")


def try_read_csv(path: Path) -> tuple[pd.DataFrame | None, str | None, str | None]:
    last_err = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc), enc, None
        except Exception as e:
            last_err = f"{enc}: {type(e).__name__}: {e}"
    return None, None, last_err


def resolve_input_path(cli: Path | None) -> tuple[Path | None, str | None]:
    if cli is not None and cli.exists():
        return cli, None
    if cli is not None:
        return None, f"指定文件不存在: {cli}"
    for p in DEFAULT_CANDIDATES:
        if p.exists():
            return p, None
    return None, "未找到默认路径中的任一路径: " + ", ".join(str(p) for p in DEFAULT_CANDIDATES)


def main() -> None:
    ap = argparse.ArgumentParser(description="易梦论文表 lab_strength 闸门")
    ap.add_argument("--input", type=Path, default=None, help="CSV 路径（默认同上候选）")
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "outputs" / "lab_strength" / "paper_yimeng_gate_report.json",
        help="JSON 报告输出路径",
    )
    args = ap.parse_args()

    path, err = resolve_input_path(args.input)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    envelope: dict[str, object] = {
        "resolved_input": str(path) if path else None,
        "requested_input": str(args.input) if args.input else None,
    }

    if path is None:
        envelope["status"] = "error"
        envelope["error"] = err
        envelope["gate"] = None
        out_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(envelope, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    df, enc, read_err = try_read_csv(path)
    if df is None:
        envelope["status"] = "error"
        envelope["error"] = read_err
        envelope["gate"] = None
        out_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(envelope, ensure_ascii=False, indent=2))
        raise SystemExit(3)

    envelope["read_encoding"] = enc
    envelope["status"] = "ok"
    envelope["gate"] = validate_for_lab_strength_training(df)

    out_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")

    if not envelope["gate"]["ok"]:
        print("=== 检查未通过：问题清单（不训练）===")
        for i, p in enumerate(envelope["gate"]["problems"], 1):
            print(f"{i}. {p}")
        print()
        print("完整 JSON 已写入:", out_path.resolve())
        raise SystemExit(1)

    print("检查通过。warnings 见 JSON:", out_path.resolve())
    print(json.dumps(envelope["gate"]["warnings"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
