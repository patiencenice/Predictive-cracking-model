"""
只读诊断：compressive 任务的协议显式化检查、窄场景候选集统计、最差组复核清单。

不训练、不改模型、不改主链路。

用法（在 SteelFiberCrackPredictor 目录）:
  py scripts/diagnose_compressive_protocol_candidates.py
  py scripts/diagnose_compressive_protocol_candidates.py --in data/lab_strength_training_merged.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IN = _ROOT / "data" / "lab_strength_training_merged.csv"
DEFAULT_GROUP_ERR = _ROOT / "outputs" / "lab_strength" / "lab_strength_group_error_by_source_group.json"
DEFAULT_OOF_DIAG = _ROOT / "outputs" / "lab_strength" / "lab_strength_oof_diagnosis.json"
DEFAULT_OUT = _ROOT / "outputs" / "lab_strength" / "compressive_protocol_diagnosis.json"
DEFAULT_MANUAL_FILL_OUT = _ROOT / "outputs" / "lab_strength" / "compressive_manual_fill_checklist.json"

FOCUS_GROUPS = [
    "XUkun/C33_BASE",
    "XUkun/C35_BASE",
    "XUkun/C37_BASE",
    "XUkun/C41_BASE",
]

KEY_FIELDS_FOR_CANDIDATE = [
    "lab_specimen",
    "lab_cube_edge_mm",
    "lab_loading_compression",
    "lab_curing_regime",
    "cube_strength_mpa_semantics",
    "lab_protocol_closed_flag_compressive",
]


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


def _field_presence_summary(df: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in df.columns:
        return {
            "present": False,
            "n_nonempty": 0,
            "nonempty_ratio": 0.0,
            "auditability": "missing_column",
            "note": "列缺失，无法审计。",
        }
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        nonempty = s.notna()
    else:
        nonempty = s.map(_norm_str) != ""
    n_nonempty = int(nonempty.sum())
    ratio = float(n_nonempty / len(df)) if len(df) else 0.0
    top_values: list[Any] = []
    try:
        vals = s[nonempty].astype(str).value_counts().head(5)
        top_values = [{"value": str(k), "count": int(v)} for k, v in vals.items()]
    except Exception:
        pass
    return {
        "present": True,
        "n_nonempty": n_nonempty,
        "nonempty_ratio": round(ratio, 6),
        "top_values_sample": top_values,
        "auditability": "ok" if n_nonempty > 0 else "present_but_empty",
    }


def _detect_curing_condition_columns(df: pd.DataFrame) -> list[str]:
    hits: list[str] = []
    for c in df.columns:
        k = c.lower()
        if "curing" in k and c != "curing_days":
            hits.append(c)
        elif "humidity" in k:
            hits.append(c)
        elif "temperature" in k:
            hits.append(c)
        elif "养护" in c:
            hits.append(c)
    return sorted(set(hits))


def _detect_cube_strength_semantic_columns(df: pd.DataFrame) -> list[str]:
    hits: list[str] = []
    for c in df.columns:
        k = c.lower()
        if "cube_strength" in k and (
            "semantic" in k
            or "meaning" in k
            or "definition" in k
            or "note" in k
            or "source" in k
            or "basis" in k
        ):
            hits.append(c)
        if ("fcu" in k or "cube" in k) and ("note" in k or "semantic" in k or "meaning" in k):
            hits.append(c)
    return sorted(set(hits))


def _standard_curing_mask(df: pd.DataFrame, curing_cols: list[str]) -> tuple[pd.Series, dict[str, Any]]:
    if "lab_curing_regime" in df.columns:
        s = df["lab_curing_regime"].map(_norm_str).str.lower()
        out = s.eq("standard")
        return out, {
            "has_explicit_standard_curing_field": True,
            "mode": "lab_curing_regime",
            "matched_columns": ["lab_curing_regime"],
            "n_rows_marked_standard": int(out.sum()),
        }
    if not curing_cols:
        return pd.Series(False, index=df.index), {
            "has_explicit_standard_curing_field": False,
            "note": "未找到可用于判定标准养护的显式字段；不将默认值视为闭合。",
        }
    out = pd.Series(False, index=df.index)
    matched_cols: list[str] = []
    keys = ("标准养护", "standard", "std")
    for c in curing_cols:
        s = df[c].map(_norm_str).str.lower()
        m = pd.Series(False, index=df.index)
        for kw in keys:
            m = m | s.str.contains(kw, na=False)
        if m.any():
            matched_cols.append(c)
        out = out | m
    return out, {
        "has_explicit_standard_curing_field": True,
        "matched_columns": matched_cols,
        "n_rows_marked_standard": int(out.sum()),
    }


def _protocol_masks(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    mr_ok = pd.to_numeric(df.get("needs_manual_review", 0), errors="coerce").fillna(0) < 0.5
    c28 = pd.to_numeric(df.get("curing_days", math.nan), errors="coerce").eq(28.0)

    has_spec = "lab_specimen" in df.columns
    has_edge = "lab_cube_edge_mm" in df.columns
    has_load = "lab_loading_compression" in df.columns
    cube_sem_cols = _detect_cube_strength_semantic_columns(df)

    spec_150_cube = pd.Series(False, index=df.index)
    if has_spec and has_edge:
        spec = df["lab_specimen"].map(_norm_str)
        edge = pd.to_numeric(df["lab_cube_edge_mm"], errors="coerce")
        spec_150_cube = spec.str.contains("立方体", na=False) & edge.eq(150.0)

    load_ok = pd.Series(False, index=df.index)
    if has_load:
        load_ok = df["lab_loading_compression"].map(_norm_str) != ""

    curing_cols = _detect_curing_condition_columns(df)
    standard_curing, curing_meta = _standard_curing_mask(df, curing_cols)

    # 协议闭合优先使用显式 flag；否则退回字段条件判定（仍不使用默认值补齐）
    if "lab_protocol_closed_flag_compressive" in df.columns:
        pc = pd.to_numeric(
            df["lab_protocol_closed_flag_compressive"], errors="coerce"
        ).fillna(0)
        protocol_closed = pc >= 0.5
        protocol_closed_mode = "explicit_flag"
    else:
        protocol_closed = (
            pd.Series(has_spec and has_edge and has_load, index=df.index)
            & spec_150_cube
            & load_ok
            & standard_curing
        )
        protocol_closed_mode = "derived_from_fields"

    tier = df.get("data_tier", pd.Series([""] * n, index=df.index)).map(_norm_str)
    tier_a = tier.eq("A")
    tier_empty_like = tier.eq("")

    strict = c28 & spec_150_cube & standard_curing & protocol_closed & tier_a & mr_ok
    relaxed = c28 & spec_150_cube & standard_curing & protocol_closed & (tier_a | tier_empty_like) & mr_ok

    def _candidate_stats(mask: pd.Series) -> dict[str, Any]:
        sg = sorted(df.loc[mask, "source_group"].astype(str).unique().tolist()) if "source_group" in df.columns else []
        n_rows = int(mask.sum())
        fit = n_rows >= 30 and len(sg) >= 5
        return {
            "n_rows": n_rows,
            "source_groups": sg,
            "is_suitable_for_standalone_model_now": fit,
            "conclusion": (
                "可考虑单独建模（样本量与分组规模初步可用）"
                if fit
                else "暂不建议单独建模（样本量或分组规模不足）"
            ),
        }

    return {
        "field_presence": {
            "lab_specimen": _field_presence_summary(df, "lab_specimen"),
            "lab_cube_edge_mm": _field_presence_summary(df, "lab_cube_edge_mm"),
            "lab_loading_compression": _field_presence_summary(df, "lab_loading_compression"),
            "curing_days": _field_presence_summary(df, "curing_days"),
            "curing_condition_fields": {
                "detected_columns": curing_cols,
                "presence": {c: _field_presence_summary(df, c) for c in curing_cols},
                "standard_curing_detection": curing_meta,
            },
            "cube_strength_semantic_fields": {
                "detected_columns": sorted(
                    set(
                        cube_sem_cols
                        + (
                            ["cube_strength_mpa_semantics"]
                            if "cube_strength_mpa_semantics" in df.columns
                            else []
                        )
                    )
                ),
                "note": (
                    "未检测到 cube_strength_mpa 语义说明列。"
                    if not cube_sem_cols and "cube_strength_mpa_semantics" not in df.columns
                    else "检测到可用于解释 cube_strength_mpa 语义的列。"
                ),
            },
            "lab_protocol_closed_flag_compressive": _field_presence_summary(
                df, "lab_protocol_closed_flag_compressive"
            ),
        },
        "candidate_sets": {
            "strict": _candidate_stats(strict),
            "relaxed": _candidate_stats(relaxed),
        },
        "protocol_closed_judgement_mode": protocol_closed_mode,
    }


def _group_formula_bias_map(group_err: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    arr = (
        group_err.get("compressive", {})
        .get("formula_only", [])
    )
    if not isinstance(arr, list):
        return out
    for x in arr:
        if not isinstance(x, dict):
            continue
        sg = str(x.get("source_group", "")).strip()
        if not sg:
            continue
        out[sg] = {
            "n": x.get("n"),
            "mae_formula_only": x.get("mae"),
            "pred_minus_true_formula_only": x.get("mean_pred_minus_true"),
        }
    return out


def _worst_group_review(
    df: pd.DataFrame,
    focus_groups: list[str],
    group_err: dict[str, Any],
) -> list[dict[str, Any]]:
    bias_map = _group_formula_bias_map(group_err)
    protocol_cols = ["lab_specimen", "lab_cube_edge_mm", "lab_loading_compression", "curing_days"]
    out: list[dict[str, Any]] = []
    for sg in focus_groups:
        sub = df[df["source_group"].astype(str) == sg].copy() if "source_group" in df.columns else pd.DataFrame()
        row = {
            "source_group": sg,
            "n_rows": int(len(sub)),
            "cube_strength_mpa_values": [],
            "compressive_true_values": [],
            "formula_bias": bias_map.get(sg, {}),
            "protocol_field_missing_summary": {},
            "suggest_needs_manual_review": False,
            "suspected_primary_issue": "insufficient_data",
        }
        if len(sub) == 0:
            out.append(row)
            continue
        row["cube_strength_mpa_values"] = [
            float(v) for v in pd.to_numeric(sub.get("cube_strength_mpa"), errors="coerce").dropna().tolist()
        ]
        row["compressive_true_values"] = [
            float(v) for v in pd.to_numeric(sub.get("compressive_true"), errors="coerce").dropna().tolist()
        ]
        miss: dict[str, Any] = {}
        for c in protocol_cols:
            if c not in sub.columns:
                miss[c] = {"present": False, "n_nonempty": 0}
                continue
            ss = sub[c]
            nonempty = ss.notna() if pd.api.types.is_numeric_dtype(ss) else (ss.map(_norm_str) != "")
            miss[c] = {"present": True, "n_nonempty": int(nonempty.sum())}
        row["protocol_field_missing_summary"] = miss

        bias = row["formula_bias"].get("pred_minus_true_formula_only")
        has_protocol_gap = any((not v["present"]) or v["n_nonempty"] <= 0 for v in miss.values())
        has_semantic_gap = "cube_strength_mpa" in sub.columns and pd.to_numeric(sub["cube_strength_mpa"], errors="coerce").notna().any()
        if has_protocol_gap:
            row["suspected_primary_issue"] = "protocol_inconsistency_or_missing_fields"
            row["suggest_needs_manual_review"] = True
        elif isinstance(bias, (int, float)) and bias < -4.0 and has_semantic_gap:
            row["suspected_primary_issue"] = "label_or_cube_strength_semantics_mismatch"
            row["suggest_needs_manual_review"] = True
        elif isinstance(bias, (int, float)) and abs(bias) > 2.0:
            row["suspected_primary_issue"] = "group_systematic_bias_needs_traceability_review"
            row["suggest_needs_manual_review"] = True
        else:
            row["suspected_primary_issue"] = "no_severe_signal"
        out.append(row)
    return out


def _is_missing_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    s = str(v).strip()
    return s == "" or s.lower() == "nan"


def _build_manual_fill_checklist(
    df: pd.DataFrame,
    focus_groups: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for sg in focus_groups:
        sub = df[df["source_group"].astype(str) == sg].copy() if "source_group" in df.columns else pd.DataFrame()
        entry: dict[str, Any] = {
            "source_group": sg,
            "n_rows": int(len(sub)),
            "key_fields": [],
        }
        if len(sub) == 0:
            for f in KEY_FIELDS_FOR_CANDIDATE:
                entry["key_fields"].append(
                    {
                        "field_name": f,
                        "current_value": None,
                        "is_missing": True,
                        "is_key_for_compressive_candidate": True,
                        "must_fill_for_candidate": True,
                        "missing_should_trigger_manual_review": True,
                        "affects_candidate_set": "strict_and_relaxed",
                    }
                )
            rows.append(entry)
            continue

        # 每组当前多为单行；若多行则只读采样第1行并给出 distinct_values 供人工判断
        r0 = sub.iloc[0]
        for f in KEY_FIELDS_FOR_CANDIDATE:
            if f in sub.columns:
                vals = sub[f].tolist()
                distinct = []
                for v in vals:
                    sv = None if _is_missing_value(v) else str(v)
                    if sv not in distinct:
                        distinct.append(sv)
                cur = None if _is_missing_value(r0.get(f)) else r0.get(f)
                miss = _is_missing_value(cur)
            else:
                distinct = []
                cur = None
                miss = True

            entry["key_fields"].append(
                {
                    "field_name": f,
                    "current_value": cur,
                    "is_missing": bool(miss),
                    "distinct_values_in_group": distinct,
                    "is_key_for_compressive_candidate": True,
                    "must_fill_for_candidate": True,
                    "missing_should_trigger_manual_review": True,
                    "fill_purpose": "用于 compressive 协议闭合审计与 strict/relaxed 候选集判定",
                    "affects_candidate_set": "strict_and_relaxed",
                }
            )
        rows.append(entry)

    return {
        "focus_groups": focus_groups,
        "key_fields_for_candidate": KEY_FIELDS_FOR_CANDIDATE,
        "rows": rows,
        "note": "只读清单：不自动编造字段值，不自动将 lab_protocol_closed_flag_compressive 置为 1。",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="compressive 协议/候选集/最差组只读诊断")
    ap.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    ap.add_argument("--group-error", dest="group_error", type=Path, default=DEFAULT_GROUP_ERR)
    ap.add_argument("--oof-diag", dest="oof_diag", type=Path, default=DEFAULT_OOF_DIAG)
    ap.add_argument("--out", dest="out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--manual-fill-out",
        dest="manual_fill_out",
        type=Path,
        default=DEFAULT_MANUAL_FILL_OUT,
    )
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    group_err = _safe_read_json(args.group_error)
    oof_diag = _safe_read_json(args.oof_diag)

    proto = _protocol_masks(df)
    worst = _worst_group_review(df, FOCUS_GROUPS, group_err)
    manual_fill = _build_manual_fill_checklist(df, FOCUS_GROUPS)

    rep: dict[str, Any] = {
        "input_csv": str(args.inp),
        "n_rows_csv": int(len(df)),
        "compressive_protocol_audit": proto["field_presence"],
        "compressive_candidate_sets": proto["candidate_sets"],
        "compressive_worst_group_review": worst,
        "references": {
            "group_error_json": str(args.group_error),
            "oof_diagnosis_json": str(args.oof_diag),
            "oof_formula_bias_overview": {
                "formula_bias_mean_y_minus_formula": (
                    oof_diag.get("compressive", {}).get("formula_bias_mean_y_minus_formula")
                ),
                "note": oof_diag.get("compressive", {}).get("formula_bias_note"),
            },
        },
        "scope_note": "只读诊断，不修改训练逻辑、不触发训练。",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(args.manual_fill_out, "w", encoding="utf-8") as f:
        json.dump(manual_fill, f, ensure_ascii=False, indent=2)

    print("written:", args.out)
    print("written:", args.manual_fill_out)
    print("strict_n:", rep["compressive_candidate_sets"]["strict"]["n_rows"])
    print("relaxed_n:", rep["compressive_candidate_sets"]["relaxed"]["n_rows"])


if __name__ == "__main__":
    main()
