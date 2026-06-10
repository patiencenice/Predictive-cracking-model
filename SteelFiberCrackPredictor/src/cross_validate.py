"""
在完整训练集上做 K 折交叉验证（不覆盖 models/ 下 pkl），生成论文级性能报表 JSON。

每折：训练集上 fit StandardScaler + 三个 XGBoost，在验证折上计算 MAE/RMSE/R² 与分类 Accuracy/F1。

用法:
  py -m src.cross_validate
  py -m src.cross_validate --csv data/training_data.csv --folds 5
  py -m src.cross_validate --out models/cv_metrics.json
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

from src.features import FEATURE_COLUMNS
from src.paths import CONFIG_YAML, MODELS_DIR, PROJECT_ROOT
from src.train_utils import cross_validate_models, load_model_config

TARGET_COLS = ("crack_width", "crack_density", "cracking_risk")


def main() -> None:
    ap = argparse.ArgumentParser(description="K 折交叉验证并导出 cv_metrics.json")
    ap.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "training_data.example.csv",
        help="与训练相同的 CSV",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=CONFIG_YAML,
        help="YAML 配置（与训练共用 XGBoost 超参）",
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=5,
        help="折数（需满足 样本数 ≥ folds×3）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=MODELS_DIR / "cv_metrics.json",
        help="报表输出路径（默认 models/cv_metrics.json）",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="划分随机种子",
    )
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"未找到 CSV: {args.csv}")

    df = pd.read_csv(args.csv)
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            raise SystemExit(f"CSV 缺少特征列: {c}")
    for c in TARGET_COLS:
        if c not in df.columns:
            raise SystemExit(f"CSV 缺少目标列: {c}")

    if len(df) < 8:
        raise SystemExit("样本过少，无法做交叉验证。")

    X = df[list(FEATURE_COLUMNS)].copy()
    y_w = df["crack_width"].astype(float)
    y_d = df["crack_density"].astype(float)
    y_r = df["cracking_risk"].astype(int)

    cfg = load_model_config(args.config)
    report = cross_validate_models(
        X,
        y_w,
        y_d,
        y_r,
        cfg,
        n_splits=args.folds,
        random_state=args.seed,
        verbose=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved CV report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
