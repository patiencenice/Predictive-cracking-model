"""
只读审计：主开裂训练表治理列（sidecar metadata）。

不训练、不跑 GroupKFold、不回写 CSV。

用法（在 SteelFiberCrackPredictor 目录下）:
  py scripts/diagnose_crack_training_governance.py
  py scripts/diagnose_crack_training_governance.py --csv data/training_data.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

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
CRACK_GOVERNANCE_COLUMN_NAMES = _init_mod.CRACK_GOVERNANCE_COLUMN_NAMES

DEFAULT_CSV = _ROOT / "data" / "training_data.csv"
DEFAULT_OUT = _ROOT / "outputs" / "crack_governance" / "crack_training_governance.json"

VALID_DATA_TIERS = frozenset(
    {
        "A_lab_native",
        "B_literature_verified",
        "C_literature_extracted",
    }
)

VALID_CRACK_WIDTH_DEFINITION_IDS = frozenset(
    {
        "CW_MAX_SURFACE_MM",
        "CW_MEAN_MM",
        "CW_INNER_MM",
        "CW_UNSPECIFIED",
    }
)

LABEL_COLUMNS = ("crack_width", "crack_density", "cracking_risk")


def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


def _nonempty_mask(series: pd.Series, *, col: str) -> pd.Series:
    if col == "needs_manual_review":
        return pd.to_numeric(series, errors="coerce").notna()
    return series.map(_norm_str) != ""


def _field_presence(df: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in df.columns:
        return {
            "present": False,
            "n_nonempty": 0,
            "nonempty_ratio": 0.0,
        }
    nn = _nonempty_mask(df[col], col=col)
    n = int(len(df))
    c = int(nn.sum())
    return {
        "present": True,
        "n_nonempty": c,
        "nonempty_ratio": round(c / n, 6) if n else 0.0,
    }


def _value_counts_str(series: pd.Series, *, empty_label: str = "<empty>") -> dict[str, int]:
    s = series.map(_norm_str)
    s = s.replace("", empty_label)
    return {str(k): int(v) for k, v in s.value_counts().items()}


def _manual_review_series(df: pd.DataFrame) -> pd.Series:
    if "needs_manual_review" not in df.columns:
        return pd.Series([math.nan] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df["needs_manual_review"], errors="coerce")


def _labels_complete_mask(df: pd.DataFrame) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    for c in LABEL_COLUMNS:
        if c not in df.columns:
            return pd.Series(False, index=df.index)
        if c == "cracking_risk":
            v = pd.to_numeric(df[c], errors="coerce")
            ok &= v.notna()
        else:
            v = pd.to_numeric(df[c], errors="coerce")
            ok &= v.notna() & v.map(lambda x: math.isfinite(float(x)))
    return ok


def _tier_counts_exclusive(df: pd.DataFrame) -> dict[str, Any]:
    """A/B/C/暂缓/未分级（互斥计数，仅诊断用，不写回 CSV）。"""
    n = len(df)
    tier = (
        df["data_tier"].map(_norm_str)
        if "data_tier" in df.columns
        else pd.Series([""] * n, index=df.index)
    )
    mr = _manual_review_series(df)
    sg = (
        df["source_group"].map(_norm_str)
        if "source_group" in df.columns
        else pd.Series([""] * n, index=df.index)
    )
    def_id = (
        df["crack_width_definition_id"].map(_norm_str)
        if "crack_width_definition_id" in df.columns
        else pd.Series([""] * n, index=df.index)
    )
    labels_ok = _labels_complete_mask(df)

    hold = mr.isna() | (mr >= 0.5)
    illegal_tier = tier.ne("") & ~tier.isin(VALID_DATA_TIERS)

    a_mask = (
        ~hold
        & tier.eq("A_lab_native")
        & sg.ne("")
        & def_id.isin(VALID_CRACK_WIDTH_DEFINITION_IDS)
        & def_id.ne("CW_UNSPECIFIED")
        & labels_ok
    )
    b_mask = ~hold & ~a_mask & tier.eq("B_literature_verified")
    c_mask = (
        ~hold
        & ~a_mask
        & ~b_mask
        & (tier.eq("C_literature_extracted") | illegal_tier)
    )
    unclassified = ~hold & ~a_mask & ~b_mask & ~c_mask & tier.eq("")

    return {
        "note": "互斥计数；空 needs_manual_review 归入暂缓；不自动标 A。",
        "hold_pending": int(hold.sum()),
        "tier_A_candidate": int(a_mask.sum()),
        "tier_B": int(b_mask.sum()),
        "tier_C_or_illegal_tier": int(c_mask.sum()),
        "unclassified_tier_empty": int(unclassified.sum()),
        "illegal_tier_row_count": int(illegal_tier.sum()),
    }


def _group_audit(df: pd.DataFrame) -> dict[str, Any]:
    if "source_group" not in df.columns:
        return {
            "source_group_present": False,
            "note": "无 source_group 列，无法做分组审计。",
        }
    sg = df["source_group"].map(_norm_str)
    nonempty = sg.ne("")
    vc = sg[nonempty].value_counts()
    singletons = sorted(vc[vc == 1].index.astype(str).tolist())
    return {
        "source_group_present": True,
        "n_rows_with_nonempty_source_group": int(nonempty.sum()),
        "n_groups": int(vc.shape[0]),
        "singleton_group_count": len(singletons),
        "singleton_groups": singletons[:50],
        "group_size_top20": {str(k): int(v) for k, v in vc.head(20).items()},
        "max_group_share": round(float(vc.max() / len(df)), 6) if len(df) and len(vc) else None,
        "suggested_holdout_groups": [],
    }


def _literature_coverage(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in ("literature_key", "table_or_figure_ref", "source_doi"):
        out[c] = _field_presence(df, c)
    lk = df["literature_key"].map(_norm_str) if "literature_key" in df.columns else pd.Series([""] * len(df))
    tf = (
        df["table_or_figure_ref"].map(_norm_str)
        if "table_or_figure_ref" in df.columns
        else pd.Series([""] * len(df))
    )
    both = (lk != "") & (tf != "")
    out["rows_with_both_literature_key_and_table_ref"] = int(both.sum())
    if "source_group" in df.columns:
        sg = df["source_group"].map(_norm_str)
        mask = sg.ne("")
        cross = (
            df.loc[mask, ["source_group"]]
            .assign(
                literature_key=lk.loc[mask],
                table_or_figure_ref=tf.loc[mask],
            )
            .groupby("source_group", dropna=False)
            .agg(
                n_rows=("source_group", "size"),
                n_lit_key_nonempty=("literature_key", lambda s: (s != "").sum()),
                n_table_ref_nonempty=("table_or_figure_ref", lambda s: (s != "").sum()),
            )
            .reset_index()
        )
        out["source_group_literature_cross_sample"] = cross.head(15).to_dict(orient="records")
    return out


def diagnose_crack_training_governance(df: pd.DataFrame, *, input_csv: Path) -> dict[str, Any]:
    n = len(df)
    tier_s = df["data_tier"].map(_norm_str) if "data_tier" in df.columns else pd.Series([""] * n)
    illegal = tier_s.ne("") & ~tier_s.isin(VALID_DATA_TIERS)
    def_s = (
        df["crack_width_definition_id"].map(_norm_str)
        if "crack_width_definition_id" in df.columns
        else pd.Series([""] * n)
    )
    illegal_def = def_s.ne("") & ~def_s.isin(VALID_CRACK_WIDTH_DEFINITION_IDS)

    return {
        "input_csv": str(input_csv.resolve()),
        "row_count": int(n),
        "governance_columns_expected": list(CRACK_GOVERNANCE_COLUMN_NAMES),
        "column_presence": {
            c: c in df.columns for c in CRACK_GOVERNANCE_COLUMN_NAMES
        },
        "nonempty_rate": {
            c: _field_presence(df, c) for c in CRACK_GOVERNANCE_COLUMN_NAMES
        },
        "data_tier_distribution": _value_counts_str(
            df["data_tier"] if "data_tier" in df.columns else pd.Series([""] * n)
        ),
        "illegal_data_tier_values": sorted(tier_s[illegal].unique().tolist()),
        "needs_manual_review_distribution": _value_counts_str(
            df["needs_manual_review"]
            if "needs_manual_review" in df.columns
            else pd.Series([""] * n),
            empty_label="<empty>",
        ),
        "crack_width_definition_id_distribution": _value_counts_str(
            df["crack_width_definition_id"]
            if "crack_width_definition_id" in df.columns
            else pd.Series([""] * n)
        ),
        "illegal_crack_width_definition_id_values": sorted(
            def_s[illegal_def].unique().tolist()
        ),
        "tier_ABC_hold_counts": _tier_counts_exclusive(df),
        "group_audit": _group_audit(df),
        "literature_coverage": _literature_coverage(df),
        "label_columns_present": {c: c in df.columns for c in LABEL_COLUMNS},
        "note": "本报告仅审计 sidecar 治理列；不改变 train_model / FEATURE_COLUMNS。",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="主开裂训练表治理列只读诊断")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--encoding", type=str, default="utf-8-sig")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"输入不存在: {args.csv}")
        raise SystemExit(1)

    df = pd.read_csv(args.csv, encoding=args.encoding)
    report = diagnose_crack_training_governance(df, input_csv=args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"已写入: {args.out.resolve()}")


if __name__ == "__main__":
    main()
