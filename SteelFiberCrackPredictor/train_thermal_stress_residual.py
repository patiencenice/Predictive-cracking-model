"""
C30 温度应力：工程公式基线（σ_T = R·E·α·ΔT）+ 残差学习训练。

用法:
  py scripts/import_c30_temperature_stress.py
  py scripts/build_thermal_stress_training_csv.py
  py train_thermal_stress_residual.py
  py train_thermal_stress_residual.py --granularity summary --save-models
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from src.thermal_stress_residual.train_eval import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="C30 温度应力 公式+残差 训练")
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="训练 CSV（默认按 granularity 自动选择）",
    )
    p.add_argument(
        "--granularity",
        choices=("point", "summary"),
        default="point",
        help="point=时序点级（默认）；summary=峰值拉应力摘要",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "outputs" / "thermal_stress" / "residual_model",
        help="报告与模型目录",
    )
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()

    if args.data is None:
        if args.granularity == "summary":
            args.data = _ROOT / "data" / "thermal_stress" / "c30_thermal_stress_training_summary.csv"
        else:
            args.data = _ROOT / "data" / "thermal_stress" / "c30_thermal_stress_training_point.csv"

    if not args.data.exists():
        print(f"训练数据不存在：{args.data}", file=sys.stderr)
        print("请先运行 build_thermal_stress_training_csv.py", file=sys.stderr)
        sys.exit(1)

    rep = run_pipeline(
        args.data,
        args.out,
        target_col="axial_stress_mpa",
        save_models=args.save_models,
        n_splits=args.n_splits,
    )
    oof = rep["oof_global_metrics"]
    fo = oof["formula_only"]
    hg = oof["hgb"]
    print("--- OOF 指标（formula_only vs hgb）---")
    print(f"  formula_only  MAE={fo['mae']:.6f}  R2={fo['r2']:.6f}")
    print(f"  hgb            MAE={hg['mae']:.6f}  R2={hg['r2']:.6f}")
    print(f"  default_method={rep['default_method']}")
    print(f"  n_rows_used={rep['dataset_stats']['n_rows_used']}  cv={rep['cv_fold_metrics']['cv']}")
    print("报告:", (args.out / "thermal_stress_residual_report.json").resolve())


if __name__ == "__main__":
    main()
