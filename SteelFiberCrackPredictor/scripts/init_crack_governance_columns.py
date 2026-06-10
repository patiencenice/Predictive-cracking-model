"""
为主开裂训练 CSV 追加空治理列（sidecar metadata），不修改特征/标签数值。

不调用 annotate_user_training_data.py；不填 source_group / data_tier /
needs_manual_review / crack_width_definition_id 等任何默认值。

用法（在 SteelFiberCrackPredictor 目录下）:
  py scripts/init_crack_governance_columns.py --dry-run
  py scripts/init_crack_governance_columns.py
  py scripts/init_crack_governance_columns.py --in data/training_data.csv --out data/training_data.csv
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

# 治理列：不进 FEATURE_COLUMNS；追加时统一为空字符串
CRACK_GOVERNANCE_COLUMN_NAMES: tuple[str, ...] = (
    "source_group",
    "data_tier",
    "needs_manual_review",
    "crack_width_definition_id",
    "literature_key",
    "source_doi",
    "table_or_figure_ref",
)

DEFAULT_TARGETS: tuple[Path, ...] = (
    _ROOT / "data" / "training_data.csv",
    _ROOT / "data" / "training_data.example.csv",
    _ROOT / "data" / "training_data.real_template.csv",
)


def ensure_crack_governance_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """仅追加缺失列，空值；不覆盖已有非空单元格。"""
    out = df.copy()
    added: list[str] = []
    for c in CRACK_GOVERNANCE_COLUMN_NAMES:
        if c not in out.columns:
            out[c] = ""
            added.append(c)
        else:
            out[c] = out[c].fillna("")
    return out, added


def process_file(
    path: Path,
    *,
    out_path: Path | None,
    encoding: str,
    dry_run: bool,
) -> dict:
    if not path.exists():
        return {
            "path": str(path.resolve()),
            "status": "missing",
            "n_cols_before": None,
            "n_cols_after": None,
            "added_columns": [],
        }
    df = pd.read_csv(path, encoding=encoding)
    n_before = len(df.columns)
    df2, added = ensure_crack_governance_columns(df)
    n_after = len(df2.columns)
    target = out_path or path
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        df2.to_csv(target, index=False, encoding=encoding)
    return {
        "path": str(path.resolve()),
        "out_path": str(target.resolve()) if not dry_run else None,
        "status": "dry_run" if dry_run else "written",
        "n_rows": int(len(df)),
        "n_cols_before": int(n_before),
        "n_cols_after": int(n_after),
        "added_columns": added,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="主开裂训练 CSV：追加空治理列（不填默认值）"
    )
    ap.add_argument(
        "--in",
        dest="inp",
        type=Path,
        default=None,
        help="单文件模式；省略则处理 data/training_data*.csv 三个默认目标",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--encoding", type=str, default="utf-8-sig")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只报告将追加的列，不写盘",
    )
    args = ap.parse_args()

    if args.inp is not None:
        targets = [(args.inp, args.out)]
    else:
        targets = [(p, None) for p in DEFAULT_TARGETS]

    report = {
        "governance_columns_defined": list(CRACK_GOVERNANCE_COLUMN_NAMES),
        "dry_run": bool(args.dry_run),
        "files": [],
    }
    for inp, out in targets:
        report["files"].append(
            process_file(
                inp,
                out_path=out,
                encoding=args.encoding,
                dry_run=args.dry_run,
            )
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    missing = [f for f in report["files"] if f.get("status") == "missing"]
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
