"""
真实接入 lab_strength 训练前的最小数据质量检查（独立脚本，不改主流程）。

用法（在 SteelFiberCrackPredictor 目录下）:
  py check_lab_strength_training_csv.py
  py check_lab_strength_training_csv.py --input data/lab_strength_training.csv
  py check_lab_strength_training_csv.py --input path/to.csv --out outputs/lab_strength/my_check.json

默认输入: data/lab_strength_training.csv
默认报告: outputs/lab_strength/lab_strength_training_check_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REQUIRED_COLUMNS = [
    "compressive_true",
    "flexural_true",
    "fiber_content",
    "aspect_ratio",
    "w_b_ratio",
    "source_group",
]

# 与训练链路一致：若存在下列列则一并做数值可解析性检查（缺失列不报错）
OPTIONAL_NUMERIC_FOR_TRAINING = [
    "cube_strength_mpa",
    "tensile_strength",
    "cement_content",
    "curing_days",
    "temperature",
    "humidity",
]

# 与侧栏/训练表单常用范围一致（仅作提醒，不作为 hard fail）
REFERENCE_RANGES: dict[str, tuple[float, float, str]] = {
    "fiber_content": (0.5, 3.0, "体积掺量 %，与预测表单范围一致"),
    "aspect_ratio": (30.0, 100.0, "长径比，与预测表单范围一致"),
    "w_b_ratio": (0.3, 0.5, "水胶比，与预测表单范围一致"),
}


def _safe_read_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"文件不存在: {path}"
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, f"读取 CSV 失败: {type(e).__name__}: {e}"


def _missing_required(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def _coerce_numeric_series(s: pd.Series, name: str) -> tuple[pd.Series, list[str]]:
    """返回 (numeric_series, 无法解析的原始行索引列表的字符串说明)。"""
    bad: list[str] = []
    out = pd.to_numeric(s, errors="coerce")
    mask = s.notna() & out.isna()
    if mask.any():
        idx = _index_sample(mask, 20)
        bad.append(
            f"{name}: 非空但不可解析为数值的行 index={idx}{'...' if mask.sum() > 20 else ''}"
        )
    return out, bad


def _index_sample(mask: pd.Series, limit: int = 100) -> list[int]:
    return [int(i) for i in mask.index[mask].tolist()[:limit]]


def _row_missing_fraction(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    sub = df[[c for c in cols if c in df.columns]]
    if sub.empty:
        return pd.Series(0.0, index=df.index)
    return sub.isna().mean(axis=1)


def _warnings_parameter_ranges(df: pd.DataFrame) -> dict[str, Any]:
    """Warning：三列相对典型工程/表单范围的偏离统计（不 fail）。"""
    out: dict[str, Any] = {
        "type": "parameter_reference_ranges",
        "note": "以下为与当前预测侧栏常用上下界一致的参考区间，仅提醒录入是否异常；不纳入 passes_minimal_gates。",
        "columns": {},
    }
    for col, (lo, hi, desc) in REFERENCE_RANGES.items():
        if col not in df.columns:
            out["columns"][col] = {"skipped": True, "reason": "列不存在"}
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        valid = v.dropna()
        if len(valid) == 0:
            out["columns"][col] = {
                "reference_min": lo,
                "reference_max": hi,
                "description": desc,
                "n_numeric": 0,
                "note": "无数值可比对",
            }
            continue
        below = valid < lo
        above = valid > hi
        outside = (below | above).to_numpy()
        idx_out = valid.index.to_numpy()[outside][:30]
        out["columns"][col] = {
            "reference_min": lo,
            "reference_max": hi,
            "description": desc,
            "n_numeric": int(len(valid)),
            "n_below_reference": int(below.sum()),
            "n_above_reference": int(above.sum()),
            "n_outside_reference": int(outside.sum()),
            "row_indices_outside_sample": [int(i) for i in idx_out.tolist()],
        }
    return out


def _warnings_flexural_ge_compressive(df: pd.DataFrame) -> dict[str, Any]:
    """Warning：抗折 MPa 不应大于等于抗压 MPa 的常见物理关系（可疑行）。"""
    base: dict[str, Any] = {
        "type": "suspicious_flexural_ge_compressive",
        "note": "通常抗折强度（弯拉/断裂模量级）显著低于立方体抗压；若 flexural_true >= compressive_true 请核对单位或标签是否对调。",
    }
    if "flexural_true" not in df.columns or "compressive_true" not in df.columns:
        base["skipped"] = True
        base["reason"] = "缺少 compressive_true 或 flexural_true"
        return base
    fc = pd.to_numeric(df["compressive_true"], errors="coerce")
    ff = pd.to_numeric(df["flexural_true"], errors="coerce")
    both = fc.notna() & ff.notna()
    susp = both & (ff >= fc)
    base["n_rows_checked_both_numeric"] = int(both.sum())
    base["n_suspicious_rows"] = int(susp.sum())
    base["row_indices_suspicious_sample"] = _index_sample(pd.Series(susp, index=df.index), 40)
    return base


def _warnings_source_group_max_share(df: pd.DataFrame) -> dict[str, Any]:
    """Warning：最大组样本占比，识别组分布失衡。"""
    base: dict[str, Any] = {"type": "source_group_max_share"}
    if "source_group" not in df.columns or len(df) == 0:
        base["skipped"] = True
        base["reason"] = "无 source_group 或无数据"
        return base
    sg = df["source_group"].astype(str).replace("nan", "NA")
    vc = sg.value_counts(dropna=False)
    if vc.empty:
        base["skipped"] = True
        base["reason"] = "source_group 全为空"
        return base
    top_label = str(vc.index[0])
    top_n = int(vc.iloc[0])
    n = int(len(df))
    share = float(top_n / n) if n else 0.0
    base["n_rows"] = n
    base["n_unique_groups"] = int(vc.shape[0])
    base["largest_group_label"] = top_label
    base["largest_group_count"] = top_n
    base["largest_group_share"] = round(share, 6)
    base["note"] = (
        "最大组占比过高时，GroupKFold/分组评估可能过度代表单一来源；"
        "建议结合业务检查是否需合并小类或分层抽样。"
    )
    return base


def _collect_warnings(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "parameter_reference_ranges": _warnings_parameter_ranges(df),
        "suspicious_flexural_ge_compressive": _warnings_flexural_ge_compressive(df),
        "source_group_max_share": _warnings_source_group_max_share(df),
    }


def _duplicate_report(df: pd.DataFrame, subset_cols: list[str]) -> dict[str, Any]:
    use = [c for c in subset_cols if c in df.columns]
    if len(use) < 2:
        return {"subset": use, "n_duplicate_rows_excluding_first": 0, "note": "用于判重的列过少"}
    dup_any = df.duplicated(subset=use, keep=False)
    dup_extra = df.duplicated(subset=use, keep="first")
    return {
        "subset": use,
        "n_rows_in_duplicate_groups_keep_false": int(dup_any.sum()),
        "n_duplicate_rows_excluding_first_occurrence": int(dup_extra.sum()),
        "duplicate_row_indices_sample": [int(i) for i in df.index[dup_any].tolist()[:30]],
    }


def run_checks(
    df: pd.DataFrame,
    *,
    missing_row_frac_threshold: float = 0.5,
    min_source_groups: int = 2,
    recommended_min_groups_for_groupkfold: int = 5,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
    }

    miss = _missing_required(df)
    report["required_columns_present"] = len(miss) == 0
    report["missing_required_columns"] = miss

    numeric_issues: list[str] = []
    numeric_cols_checked: list[str] = []

    cols_to_numeric = [
        c for c in REQUIRED_COLUMNS if c != "source_group" and c in df.columns
    ]
    cols_to_numeric += [c for c in OPTIONAL_NUMERIC_FOR_TRAINING if c in df.columns]

    for c in cols_to_numeric:
        numeric_cols_checked.append(c)
        _, bad = _coerce_numeric_series(df[c], c)
        numeric_issues.extend(bad)

    report["numeric_columns_checked"] = numeric_cols_checked
    report["numeric_coercion_ok"] = len(numeric_issues) == 0
    report["numeric_coercion_issues"] = numeric_issues

    # 缺失过多的行（在「已存在列」上统计缺失比例）
    check_missing_on = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if check_missing_on:
        frac = _row_missing_fraction(df, check_missing_on)
        bad_rows = frac > missing_row_frac_threshold
        report["missing_value_row_check"] = {
            "columns_used": check_missing_on,
            "threshold_missing_frac": missing_row_frac_threshold,
            "n_rows_above_threshold": int(bad_rows.sum()),
            "row_indices_sample": _index_sample(bad_rows, 30),
        }
    else:
        report["missing_value_row_check"] = {"note": "无必需列可用于缺失行统计"}

    if "source_group" in df.columns:
        sg = df["source_group"].astype(str).replace("nan", "NA")
        nuniq = int(sg.nunique(dropna=False))
        vc = sg.value_counts(dropna=False)
        small_groups = vc[vc < 2]
        report["source_group"] = {
            "n_unique": nuniq,
            "min_samples_per_group_ge_2": bool((vc >= 2).all()) if len(vc) else False,
            "n_groups_with_lt_2_samples": int((vc < 2).sum()),
            "recommended_for_groupkfold_5folds": nuniq
            >= recommended_min_groups_for_groupkfold,
            "min_group_count_for_cv_note": f"GroupKFold 5 折通常需要至少约 {recommended_min_groups_for_groupkfold} 个组（当前 {nuniq}）",
            "sufficient_for_any_grouped_cv": nuniq >= min_source_groups,
        }
    else:
        report["source_group"] = {"note": "缺少 source_group 列"}

    dup_subset = [c for c in REQUIRED_COLUMNS if c in df.columns]
    report["duplicates"] = _duplicate_report(df, dup_subset)

    pos: dict[str, Any] = {}
    for c in ("compressive_true", "flexural_true"):
        if c not in df.columns:
            pos[c] = {"present": False}
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        finite = v.dropna()
        non_pos = finite <= 0
        pos[c] = {
            "present": True,
            "n_non_positive_among_numeric": int(non_pos.sum()),
            "min_value": float(finite.min()) if len(finite) else None,
            "row_indices_non_positive_sample": [
                int(i) for i in finite.index[non_pos].tolist()[:20]
            ],
        }
    report["positive_strength_labels"] = pos

    # 总闸门（启发式）
    gates: list[bool] = [
        bool(report["required_columns_present"]),
        bool(report["numeric_coercion_ok"]),
    ]
    pc = pos.get("compressive_true", {})
    if pc.get("present"):
        gates.append(int(pc.get("n_non_positive_among_numeric", -1)) == 0)
    pf = pos.get("flexural_true", {})
    if pf.get("present"):
        gates.append(int(pf.get("n_non_positive_among_numeric", -1)) == 0)
    gates.append(bool(report.get("source_group", {}).get("sufficient_for_any_grouped_cv", True)))
    report["passes_minimal_gates"] = all(gates)

    report["warnings"] = _collect_warnings(df)

    return report


def main() -> None:
    p = argparse.ArgumentParser(description="lab_strength 训练 CSV 数据质量检查")
    p.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "data" / "lab_strength_training.csv",
        help="待检查的 CSV 路径",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "outputs" / "lab_strength" / "lab_strength_training_check_report.json",
        help="JSON 报告输出路径",
    )
    p.add_argument(
        "--missing-row-frac",
        type=float,
        default=0.5,
        help="单行在已选列上缺失比例超过该阈值则标记",
    )
    args = p.parse_args()

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    envelope: dict[str, Any] = {
        "input_path": str(args.input.resolve()),
        "output_path": str(out_path.resolve()),
    }

    df, err = _safe_read_csv(args.input)
    if df is None:
        envelope["status"] = "error"
        envelope["error"] = err
        envelope["checks"] = None
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(envelope, f, ensure_ascii=False, indent=2)
        print(envelope["error"])
        print("已写入报告:", out_path.resolve())
        raise SystemExit(2)

    envelope["status"] = "ok"
    envelope["checks"] = run_checks(df, missing_row_frac_threshold=args.missing_row_frac)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False, indent=2)

    print("检查完成。passes_minimal_gates:", envelope["checks"]["passes_minimal_gates"])
    print("报告:", out_path.resolve())


if __name__ == "__main__":
    main()
