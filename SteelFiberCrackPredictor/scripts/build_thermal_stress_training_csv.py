"""
由 C30 温度应力时序 CSV 构建残差训练表。

用法:
  py scripts/build_thermal_stress_training_csv.py
  py scripts/build_thermal_stress_training_csv.py --granularity summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.thermal_stress_residual.formula import C30_DEFAULTS
from src.thermal_stress_residual.restraint_map import restraint_factor_from_percent

TS_CSV = _ROOT / "data" / "thermal_stress" / "c30_temperature_stress_timeseries.csv"
SUM_CSV = _ROOT / "data" / "thermal_stress" / "c30_temperature_stress_summary.csv"
OUT_POINT = _ROOT / "data" / "thermal_stress" / "c30_thermal_stress_training_point.csv"
OUT_SUMMARY = _ROOT / "data" / "thermal_stress" / "c30_thermal_stress_training_summary.csv"


def _t_reference_per_sample(ts: pd.DataFrame) -> pd.Series:
    """每组取首个有效温度为 T_reference。"""
    refs: dict[str, float] = {}
    for sid, g in ts.groupby("sample_id"):
        temps = pd.to_numeric(g["specimen_temperature_c"], errors="coerce")
        valid = temps[np.isfinite(temps)]
        refs[str(sid)] = float(valid.iloc[0]) if len(valid) else float("nan")
    return refs


def build_point_frame(
    ts: pd.DataFrame,
    *,
    stride: int = 3,
    max_points_per_sample: int | None = 800,
) -> pd.DataFrame:
    refs = _t_reference_per_sample(ts)
    rows: list[dict] = []
    for sid, g in ts.groupby("sample_id"):
        g = g.sort_values("time_h")
        if stride > 1:
            g = g.iloc[::stride]
        if max_points_per_sample and len(g) > max_points_per_sample:
            idx = np.linspace(0, len(g) - 1, max_points_per_sample, dtype=int)
            g = g.iloc[idx]
        t_ref = refs.get(str(sid), float("nan"))
        t_max = float(pd.to_numeric(g["time_h"], errors="coerce").max())
        for _, row in g.iterrows():
            temp = pd.to_numeric(row.get("specimen_temperature_c"), errors="coerce")
            time_h = pd.to_numeric(row.get("time_h"), errors="coerce")
            stress = pd.to_numeric(row.get("axial_stress_mpa"), errors="coerce")
            defo = pd.to_numeric(row.get("deformation_um"), errors="coerce")
            if not (np.isfinite(temp) and np.isfinite(time_h) and np.isfinite(stress)):
                continue
            delta_t = float(temp - t_ref) if np.isfinite(t_ref) else float("nan")
            r_pct = row.get("restraint_percent")
            rows.append(
                {
                    "sample_id": sid,
                    "source_file": row.get("source_file"),
                    "strength_grade": row.get("strength_grade", "C30"),
                    "w_b_ratio": row.get("w_b_ratio"),
                    "restraint_code": row.get("restraint_code"),
                    "restraint_percent": r_pct,
                    "restraint_factor_R": restraint_factor_from_percent(r_pct),
                    "time_h": time_h,
                    "time_h_norm": time_h / t_max if t_max > 0 else 0.0,
                    "specimen_temperature_c": temp,
                    "T_reference": t_ref,
                    "delta_T_point": delta_t,
                    "axial_stress_mpa": stress,
                    "deformation_um": defo if np.isfinite(defo) else np.nan,
                    "cube_strength_mpa": C30_DEFAULTS["cube_strength_mpa"],
                    "thermal_expansion_alpha": C30_DEFAULTS["thermal_expansion_alpha"],
                    "granularity": "point",
                }
            )
    return pd.DataFrame(rows)


def build_summary_frame(sum_df: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    refs = _t_reference_per_sample(ts)
    rows: list[dict] = []
    for _, row in sum_df.iterrows():
        sid = str(row["sample_id"])
        t_ref = refs.get(sid, float("nan"))
        t_max = row.get("max_temperature_c")
        delta_t = float(t_max - t_ref) if pd.notna(t_max) and np.isfinite(t_ref) else float("nan")
        r_pct = row.get("restraint_percent")
        rows.append(
            {
                "sample_id": sid,
                "source_file": row.get("source_file"),
                "strength_grade": row.get("strength_grade", "C30"),
                "w_b_ratio": row.get("w_b_ratio"),
                "restraint_code": row.get("restraint_code"),
                "restraint_percent": r_pct,
                "restraint_factor_R": restraint_factor_from_percent(r_pct),
                "time_h": row.get("time_at_max_tensile_stress_h"),
                "time_h_norm": 1.0,
                "specimen_temperature_c": t_max,
                "T_reference": t_ref,
                "delta_T_point": delta_t,
                "axial_stress_mpa": row.get("max_tensile_stress_mpa"),
                "deformation_um": row.get("max_deformation_um"),
                "cube_strength_mpa": C30_DEFAULTS["cube_strength_mpa"],
                "thermal_expansion_alpha": C30_DEFAULTS["thermal_expansion_alpha"],
                "granularity": "summary_max_tensile",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", type=Path, default=TS_CSV)
    ap.add_argument("--summary", type=Path, default=SUM_CSV)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-points", type=int, default=800)
    ap.add_argument(
        "--granularity",
        choices=("both", "point", "summary"),
        default="both",
    )
    args = ap.parse_args()

    if not args.timeseries.exists():
        print(f"缺少时序 CSV：{args.timeseries}，请先运行 import_c30_temperature_stress.py", file=sys.stderr)
        sys.exit(1)

    ts = pd.read_csv(args.timeseries)
    meta = {"timeseries_rows": len(ts), "samples": ts["sample_id"].nunique() if len(ts) else 0}

    if args.granularity in ("both", "point"):
        pt = build_point_frame(ts, stride=args.stride, max_points_per_sample=args.max_points)
        OUT_POINT.parent.mkdir(parents=True, exist_ok=True)
        pt.to_csv(OUT_POINT, index=False, encoding="utf-8-sig")
        meta["point_training_csv"] = str(OUT_POINT.relative_to(_ROOT))
        meta["point_rows"] = len(pt)

    if args.granularity in ("both", "summary"):
        if not args.summary.exists():
            print(f"缺少 summary CSV：{args.summary}", file=sys.stderr)
            sys.exit(1)
        sum_df = pd.read_csv(args.summary)
        sm = build_summary_frame(sum_df, ts)
        sm.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
        meta["summary_training_csv"] = str(OUT_SUMMARY.relative_to(_ROOT))
        meta["summary_rows"] = len(sm)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
