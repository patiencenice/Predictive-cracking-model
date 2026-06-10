"""
只读审计：训练或业务 CSV 中温度应力相关字段的覆盖与闭合情况。

不训练、不改写数据。默认输出：
  outputs/thermal_stress/thermal_input_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.thermal_stress.derive import derive_thermal_stress_features

A_FIELDS = [
    "ambient_temperature",
    "casting_temperature",
    "curing_temperature",
    "T_reference",
    "delta_T_user",
    "T_peak_observed",
    "delta_T_inner_outer",
    "wall_thickness_mm",
    "hydration_heat_proxy_level",
    "surface_insulation_level",
    "thermal_expansion_alpha",
    "elastic_modulus_E_user",
    "restraint_level",
    "segment_length_m",
    "slip_layer_present",
    "rock_stiffness_class",
    "rebar_constraint_class",
    "splitting_tensile_strength_mpa",
    "flexural_strength_mpa",
    "core_peak_temperature_c",
    "surface_temperature_c",
    "time_to_peak_temperature_h",
    "cooling_rate_c_per_h",
    "restraint_percent",
    "restraint_code",
    "thermal_crack_observed",
    "thermal_crack_time_h",
    "thermal_crack_width_mm",
    "apparent_crack_filtered",
]

ENUMS = {
    "restraint_level": {"low", "medium", "high"},
    "restraint_code": {"r0", "r50", "r100"},
    "rock_stiffness_class": {"soft", "medium", "hard"},
    "rebar_constraint_class": {"light", "medium", "heavy"},
    "hydration_heat_proxy_level": {"low", "medium", "high"},
    "surface_insulation_level": {"low", "medium", "high"},
}


def _non_null_rate(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().any():
        return float(sn.notna().mean())
    return float(s.notna() & (s.astype(str).str.strip() != ""))


def _illegal_enum_rows(df: pd.DataFrame, col: str, allowed: set[str]) -> int:
    if col not in df.columns:
        return 0
    bad = 0
    for v in df[col]:
        if pd.isna(v) or v is None:
            continue
        t = str(v).strip().lower()
        if not t or t == "nan":
            continue
        if t not in allowed:
            bad += 1
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description="温度应力 Phase1 输入诊断（只读）")
    ap.add_argument(
        "--csv",
        type=Path,
        default=_ROOT / "data" / "training_data.csv",
        help="待审计 CSV（默认 data/training_data.csv）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "outputs" / "thermal_stress" / "thermal_input_diagnosis.json",
        help="JSON 输出路径",
    )
    ap.add_argument("--encoding", type=str, default="utf-8-sig")
    ap.add_argument("--max-rows", type=int, default=200_000, help="最多读取行数")
    args = ap.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV 不存在: {args.csv}")

    df = pd.read_csv(args.csv, encoding=args.encoding, nrows=args.max_rows)
    n = len(df)

    field_presence = {c: bool(c in df.columns) for c in A_FIELDS}
    non_null_rate = {}
    for c in A_FIELDS:
        if c not in df.columns:
            non_null_rate[c] = 0.0
        else:
            non_null_rate[c] = _non_null_rate(df[c])

    illegal_enums = {
        col: _illegal_enum_rows(df, col, allowed) for col, allowed in ENUMS.items()
    }

    miss_counters: Counter[str] = Counter()
    dt_sources: Counter[str] = Counter()
    restraint_dist: Counter[str] = Counter()
    tsi_ok = 0
    bottleneck_hits: Counter[str] = Counter()

    for _, row in df.iterrows():
        feats = derive_thermal_stress_features(row)
        src = feats.get("delta_T_eff_source")
        if isinstance(src, str):
            dt_sources[src] += 1
        rl = row.get("restraint_level") if "restraint_level" in row.index else None
        if rl is not None and not (isinstance(rl, float) and math.isnan(rl)):
            restraint_dist[str(rl).strip().lower()] += 1
        else:
            restraint_dist["<null>"] += 1

        for k in (
            "delta_T_eff_missing_flag",
            "cooling_rate_missing_flag",
            "thermal_gradient_index_missing_flag",
            "E_norm_missing_flag",
            "alpha_norm_missing_flag",
            "restraint_factor_R_missing_flag",
            "thermal_stress_index_missing_flag",
            "thermal_crack_risk_index_missing_flag",
        ):
            if feats.get(k) == 1:
                miss_counters[k] += 1

        if feats.get("thermal_stress_index_missing_flag") == 0:
            tsi = feats.get("thermal_stress_index")
            try:
                tf = float(tsi)
            except (TypeError, ValueError):
                tf = float("nan")
            if math.isfinite(tf) and tf >= 0.0:
                tsi_ok += 1

        if feats.get("thermal_stress_index_missing_flag") == 1:
            if feats.get("delta_T_eff_missing_flag") == 1:
                bottleneck_hits["delta_T_path"] += 1
            if feats.get("E_norm_missing_flag") == 1:
                bottleneck_hits["E_norm (E_user 或 cube_strength_mpa)"] += 1
            if feats.get("alpha_norm_missing_flag") == 1:
                bottleneck_hits["thermal_expansion_alpha"] += 1
            if feats.get("restraint_factor_R_missing_flag") == 1:
                bottleneck_hits["restraint_level 缺失或非法"] += 1

    tsi_ratio = (tsi_ok / n) if n else 0.0

    blocking_sorted = sorted(
        bottleneck_hits.items(), key=lambda x: (-x[1], x[0])
    )

    report = {
        "input_csv": str(args.csv.resolve()),
        "row_count": n,
        "field_presence": field_presence,
        "non_null_rate": non_null_rate,
        "illegal_enum_nonconforming_row_counts": illegal_enums,
        "missing_flag_row_counts": dict(miss_counters),
        "thermal_stress_index_computable_row_count": tsi_ok,
        "thermal_stress_index_computable_ratio": round(tsi_ratio, 6),
        "restraint_level_value_distribution": dict(restraint_dist),
        "delta_T_eff_source_distribution": dict(dt_sources),
        "blocking_reasons_when_thermal_stress_index_missing": [
            {"reason": k, "rows": v} for k, v in blocking_sorted
        ],
        "note": "thermal_stress_index 闭合需要：温差路径 + E_norm + alpha_norm + 合法 restraint_level。",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("已写入:", args.out.resolve())


if __name__ == "__main__":
    main()
