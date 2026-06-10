"""
导出主开裂训练表人工治理补值模板（只读源表，不自动填写治理值）。

用法（在 SteelFiberCrackPredictor 目录下）:
  py scripts/export_crack_governance_fill_template.py
  py scripts/export_crack_governance_fill_template.py --csv data/training_data.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_init_path = Path(__file__).resolve().parent / "init_crack_governance_columns.py"
_spec = importlib.util.spec_from_file_location(
    "init_crack_governance_columns", _init_path
)
_init_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_init_mod)
CRACK_GOVERNANCE_COLUMN_NAMES: tuple[str, ...] = _init_mod.CRACK_GOVERNANCE_COLUMN_NAMES

DEFAULT_CSV = _ROOT / "data" / "training_data.csv"
DEFAULT_OUT = _ROOT / "outputs" / "crack_governance" / "crack_governance_fill_template.csv"
DEFAULT_META = _ROOT / "outputs" / "crack_governance" / "crack_governance_fill_template_meta.json"

# 供人工对照的原始输入/标签列（来自 training_data，非治理列）
SNAPSHOT_INPUT_COLUMNS: tuple[str, ...] = (
    "strength_grade",
    "fiber_type",
    "fiber_content",
    "aspect_ratio",
    "w_b_ratio",
    "cube_strength_mpa",
    "curing_days",
)

SNAPSHOT_LABEL_COLUMNS: tuple[str, ...] = (
    "crack_width",
    "crack_density",
    "cracking_risk",
)

META_FILL_COLUMNS: tuple[str, ...] = (
    "fill_note",
    "reviewer",
    "review_date",
)

ENC_TO_GRADE: dict[int, str] = {}
ENC_TO_FIBER_TYPE: dict[int, str] = {}


def _load_label_maps() -> None:
    from src.features import FIBER_TYPE_MAP, STRENGTH_GRADE_ENC

    global ENC_TO_GRADE, ENC_TO_FIBER_TYPE
    ENC_TO_GRADE = {int(v): k for k, v in STRENGTH_GRADE_ENC.items()}
    ENC_TO_FIBER_TYPE = {int(v): k for k, v in FIBER_TYPE_MAP.items()}


def _decode_grade(enc: object) -> str:
    try:
        return ENC_TO_GRADE.get(int(enc), "")
    except (TypeError, ValueError):
        return ""


def _decode_fiber_type(enc: object) -> str:
    try:
        return ENC_TO_FIBER_TYPE.get(int(enc), "")
    except (TypeError, ValueError):
        return ""


def build_fill_template(df: pd.DataFrame) -> pd.DataFrame:
    """从训练表构建人工模板；治理列与 meta 列一律留空。"""
    n = len(df)
    out = pd.DataFrame({"row_index": list(range(n))})

    out["strength_grade"] = (
        df["strength_grade_enc"].map(_decode_grade)
        if "strength_grade_enc" in df.columns
        else ""
    )
    out["fiber_type"] = (
        df["fiber_type_enc"].map(_decode_fiber_type)
        if "fiber_type_enc" in df.columns
        else ""
    )
    for c in ("fiber_content", "aspect_ratio", "w_b_ratio", "cube_strength_mpa", "curing_days"):
        if c in df.columns:
            out[c] = df[c]
        else:
            out[c] = ""

    for c in SNAPSHOT_LABEL_COLUMNS:
        if c in df.columns:
            out[c] = df[c]
        else:
            out[c] = ""

    for c in CRACK_GOVERNANCE_COLUMN_NAMES:
        out[c] = ""

    for c in META_FILL_COLUMNS:
        out[c] = ""

    return out


def export_template(
    *,
    input_csv: Path,
    output_csv: Path,
    encoding: str,
) -> dict:
    df = pd.read_csv(input_csv, encoding=encoding)
    _load_label_maps()
    template = build_fill_template(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_csv, index=False, encoding=encoding)
    return {
        "input_csv": str(input_csv.resolve()),
        "output_csv": str(output_csv.resolve()),
        "row_count": int(len(template)),
        "columns": list(template.columns),
        "governance_columns_left_empty": list(CRACK_GOVERNANCE_COLUMN_NAMES),
        "meta_columns_left_empty": list(META_FILL_COLUMNS),
        "auto_filled_governance_values": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="导出主开裂治理人工补值模板 CSV（治理列不自动填写）"
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--meta-out", type=Path, default=DEFAULT_META)
    ap.add_argument("--encoding", type=str, default="utf-8-sig")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"输入不存在: {args.csv}")
        raise SystemExit(1)

    meta = export_template(
        input_csv=args.csv,
        output_csv=args.out,
        encoding=args.encoding,
    )
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"已写入模板: {args.out.resolve()}")
    print(f"已写入元数据: {args.meta_out.resolve()}")


if __name__ == "__main__":
    main()
