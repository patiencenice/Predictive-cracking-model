"""
使用 CSV 训练钢纤维混凝土抗裂模型，与推理端 FEATURE_COLUMNS 严格对齐。

默认数据路径: data/training_data.csv
列要求:
  - 与 src.features.FEATURE_COLUMNS 同名的特征列（数值，编码与训练时一致；含 binder_content、fly_ash、slag_powder）
  - crack_width (mm), crack_density, cracking_risk (整数 0=低 1=中 2=高)

训练流程（内部使用 sklearn Pipeline: StandardScaler + XGBoost）：
  - models/ 保存拆分后的 scaler 与各任务 pkl，供 Streamlit / predictor 加载；
  - outputs/pipelines/ 保存完整 Pipeline；outputs/ 还写入 training_metrics.json、feature_importance.json。

示例:
  py -m src.train_model
  py -m src.train_model --csv data/my_lab.csv --config config/model_config.yaml

评估与 SHAP（结果默认在 outputs/）:
  py evaluate.py
  py shap_analysis.py --task risk

论文级 K 折评估（不写 pkl，输出 models/cv_metrics.json）:
  py -m src.cross_validate --csv data/my_lab.csv --folds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import FEATURE_COLUMNS
from src.paths import CONFIG_YAML, MODELS_DIR, PROJECT_ROOT
from src.train_utils import fit_models_save, load_model_config


TARGET_COLS = ("crack_width", "crack_density", "cracking_risk")


def main() -> None:
    ap = argparse.ArgumentParser(description="训练裂缝宽度/密度回归与开裂风险分类模型")
    ap.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "training_data.example.csv",
        help="训练 CSV 路径（默认可用示例表；正式数据可另存为 training_data.csv）",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=CONFIG_YAML,
        help="YAML 配置（超参与划分比例）",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=MODELS_DIR,
        help="模型输出目录",
    )
    args = ap.parse_args()

    if not args.csv.exists():
        ex = PROJECT_ROOT / "data" / "training_data.example.csv"
        raise SystemExit(
            f"未找到训练文件: {args.csv}\n"
            f"请准备 CSV（列见文档字符串），或从示例复制修改: {ex}"
        )

    df = pd.read_csv(args.csv)
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            raise SystemExit(f"CSV 缺少特征列: {c}（须与 FEATURE_COLUMNS 一致）")
    for c in TARGET_COLS:
        if c not in df.columns:
            raise SystemExit(f"CSV 缺少目标列: {c}")

    if len(df) < 8:
        raise SystemExit("样本过少（建议至少数十条）；划分测试集需要一定样本量。")

    X = df[list(FEATURE_COLUMNS)].copy()
    y_w = df["crack_width"].astype(float)
    y_d = df["crack_density"].astype(float)
    y_r = df["cracking_risk"].astype(int)

    cfg = load_model_config(args.config)
    fit_models_save(X, y_w, y_d, y_r, args.out_dir, cfg, verbose=True)


if __name__ == "__main__":
    main()
