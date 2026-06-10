"""
SHAP 分析脚本：对开裂风险分类器（或可扩展至回归）做树模型解释，图与摘要保存到 outputs/shap/。

依赖与训练数据：需已训练好的 models/*.pkl 及与特征列一致的 CSV。

用法:
  py shap_analysis.py
  py shap_analysis.py --task risk --max-background 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import shap
except ImportError as e:
    raise SystemExit("请先安装 shap: py -m pip install shap") from e

from src.features import FEATURE_COLUMNS
from src.paths import MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description="SHAP 树模型解释，输出到 outputs/shap/")
    ap.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "training_data.example.csv",
        help="用于背景分布与解释样本的特征表",
    )
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="模型目录",
    )
    ap.add_argument(
        "--task",
        choices=("risk", "width", "density"),
        default="risk",
        help="解释哪个任务的模型（默认开裂风险分类）",
    )
    ap.add_argument(
        "--max-background",
        type=int,
        default=400,
        help="背景样本最大行数（降低可加快计算）",
    )
    ap.add_argument(
        "--max-explain",
        type=int,
        default=80,
        help="条形图解释的样本行数上限",
    )
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"未找到 CSV: {args.csv}")

    df = pd.read_csv(args.csv)
    miss = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if miss:
        raise SystemExit(f"CSV 缺少列: {miss}")

    X = df[FEATURE_COLUMNS].astype(np.float64, copy=False)
    if len(X) > args.max_background:
        X_bg = X.sample(args.max_background, random_state=42)
    else:
        X_bg = X

    scaler = joblib.load(args.models_dir / "feature_scaler.pkl")
    X_bg_s = scaler.transform(X_bg)

    out_dir = OUTPUTS_DIR / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "risk":
        model = joblib.load(args.models_dir / "crack_classifier.pkl")
        # TreeExplainer：以背景数据近似特征缺失时的期望
        explainer = shap.TreeExplainer(model, X_bg_s)
        X_exp = X_bg_s[: args.max_explain]
        shap_values = explainer.shap_values(X_exp)

        # 多分类时 shap 返回 list；逐类对 |SHAP| 按样本维求平均，再对类求平均
        n_feat = len(FEATURE_COLUMNS)
        importance = np.zeros(n_feat, dtype=np.float64)
        if isinstance(shap_values, list):
            for sv in shap_values:
                importance += np.abs(np.asarray(sv, dtype=np.float64)).mean(axis=0)
            importance /= max(1, len(shap_values))
        else:
            importance = np.abs(np.asarray(shap_values, dtype=np.float64)).mean(axis=0)
        imp1 = np.ravel(np.asarray(importance, dtype=np.float64))[:n_feat]
        order = np.argsort(-imp1)[:20]
        feat_names = []
        vals = []
        for ii in order:
            i = int(ii)
            feat_names.append(FEATURE_COLUMNS[i])
            vals.append(float(imp1[i]))
        summary = {
            "task": "cracking_risk",
            "top_features": [
                {"feature": a, "mean_abs_shap": float(b)} for a, b in zip(feat_names, vals)
            ],
        }
        with open(out_dir / "shap_summary_risk.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 条形图：以第一类 SHAP 为例（多分类下可选）
        plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
            shap.summary_plot(
                shap_values[0],
                X_exp,
                feature_names=FEATURE_COLUMNS,
                show=False,
                max_display=20,
            )
        else:
            shap.summary_plot(
                shap_values,
                X_exp,
                feature_names=FEATURE_COLUMNS,
                show=False,
                max_display=20,
            )
        plt.tight_layout()
        p = out_dir / "shap_summary_risk.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p.resolve()}")

    elif args.task == "width":
        model = joblib.load(args.models_dir / "crack_regressor.pkl")
        explainer = shap.TreeExplainer(model, X_bg_s)
        X_exp = X_bg_s[: args.max_explain]
        sv = explainer.shap_values(X_exp)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv, X_exp, feature_names=FEATURE_COLUMNS, show=False, max_display=20
        )
        plt.tight_layout()
        p = out_dir / "shap_summary_width.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p.resolve()}")

    else:
        reg_path = args.models_dir / "crack_density_regressor.pkl"
        if not reg_path.exists():
            raise SystemExit("未找到 crack_density_regressor.pkl")
        model = joblib.load(reg_path)
        explainer = shap.TreeExplainer(model, X_bg_s)
        X_exp = X_bg_s[: args.max_explain]
        sv = explainer.shap_values(X_exp)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv, X_exp, feature_names=FEATURE_COLUMNS, show=False, max_display=20
        )
        plt.tight_layout()
        p = out_dir / "shap_summary_density.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p.resolve()}")


if __name__ == "__main__":
    main()
