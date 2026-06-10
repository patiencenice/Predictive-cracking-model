"""
从已训练 crack_width 回归模型计算 SHAP mean(|SHAP|)，导出 Origin 作图 CSV 与 PNG 预览。

不修改训练逻辑、不重训模型；不改动 predictor.py / train_model.py / FEATURE_COLUMNS / 模型权重。

用法:
  py scripts/export_origin_shap_crack_width.py
  py scripts/export_origin_shap_crack_width.py --csv data/training_data.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import shap
except ImportError as e:
    raise SystemExit("请先安装 shap: py -m pip install shap") from e

from src.features import FEATURE_COLUMNS
from src.mechanism import feature_label
from src.paths import CONFIG_YAML, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT
from src.train_utils import load_model_config

EVAL_DIR = OUTPUTS_DIR / "evaluation"
DEFAULT_CSV = PROJECT_ROOT / "data" / "training_data.csv"
OUT_CSV = EVAL_DIR / "origin_shap_crack_width_top10.csv"
OUT_PNG = EVAL_DIR / "origin_shap_crack_width_top10.png"
TARGET_COL = "crack_width"
TOP_N = 10


def _load_partition(
    csv_path: Path,
    *,
    test_size: float,
    random_state: int,
    stratify_flag: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            raise SystemExit(f"CSV 缺少特征列: {c}")
    if TARGET_COL not in df.columns:
        raise SystemExit(f"CSV 缺少目标列: {TARGET_COL}")

    X = df[FEATURE_COLUMNS].astype(np.float64, copy=False)
    y_r = df["cracking_risk"].astype(int)
    stratify = y_r if stratify_flag else None
    try:
        X_train, X_test = train_test_split(
            X,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test = train_test_split(
            X,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
    return X_train, X_test


def compute_mean_abs_shap(
    *,
    models_dir: Path,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    max_background: int,
) -> np.ndarray:
    scaler = joblib.load(models_dir / "feature_scaler.pkl")
    model = joblib.load(models_dir / "crack_regressor.pkl")

    if len(X_train) > max_background:
        X_bg = X_train.sample(max_background, random_state=42)
    else:
        X_bg = X_train

    X_bg_s = scaler.transform(X_bg)
    X_exp_s = scaler.transform(X_explain)

    explainer = shap.TreeExplainer(model, X_bg_s)
    sv = explainer.shap_values(X_exp_s)
    sv_arr = np.asarray(sv, dtype=np.float64)
    if sv_arr.ndim == 3:
        sv_arr = sv_arr.mean(axis=0)
    return np.abs(sv_arr).mean(axis=0)


def export_origin_shap_crack_width(
    *,
    csv_path: Path = DEFAULT_CSV,
    models_dir: Path = MODELS_DIR,
    out_csv: Path = OUT_CSV,
    out_png: Path = OUT_PNG,
    max_background: int = 400,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")

    cfg = load_model_config(CONFIG_YAML)
    tcfg = cfg.get("training", {})
    test_size = float(tcfg.get("test_size", 0.2))
    random_state = int(tcfg.get("random_state", 42))
    stratify_flag = bool(tcfg.get("stratify_risk", True))

    X_train, X_test = _load_partition(
        csv_path,
        test_size=test_size,
        random_state=random_state,
        stratify_flag=stratify_flag,
    )

    mean_abs = compute_mean_abs_shap(
        models_dir=models_dir,
        X_train=X_train,
        X_explain=X_test,
        max_background=max_background,
    )

    rows = []
    for i, feat in enumerate(FEATURE_COLUMNS):
        rows.append(
            {
                "feature": feat,
                "feature_zh": feature_label(feat),
                "mean_abs_shap": float(mean_abs[i]),
            }
        )
    full_df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    top_df = full_df.head(top_n)[["feature_zh", "mean_abs_shap"]].reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    _save_preview_png(top_df, out_png)
    return top_df


def _save_preview_png(top_df: pd.DataFrame, out_png: Path) -> None:
    labels = top_df["feature_zh"].tolist()[::-1]
    values = top_df["mean_abs_shap"].tolist()[::-1]

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values, color="#4472C4")
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title("裂缝宽度模型 — Top 10 特征 SHAP 平均绝对贡献（预览）")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="导出 crack_width SHAP Top10 至 Origin CSV 与 PNG 预览"
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="训练/测试划分用 CSV")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="模型目录")
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV, help="Origin CSV 输出路径")
    ap.add_argument("--out-png", type=Path, default=OUT_PNG, help="PNG 预览输出路径")
    ap.add_argument(
        "--max-background",
        type=int,
        default=400,
        help="SHAP 背景样本上限（来自训练集）",
    )
    ap.add_argument("--top-n", type=int, default=TOP_N, help="导出 Top N 特征")
    args = ap.parse_args()

    top_df = export_origin_shap_crack_width(
        csv_path=args.csv,
        models_dir=args.models_dir,
        out_csv=args.out_csv,
        out_png=args.out_png,
        max_background=args.max_background,
        top_n=args.top_n,
    )

    def _safe_print(msg: str) -> None:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write((msg + "\n").encode(enc, errors="replace"))

    _safe_print(f"已写入 CSV: {args.out_csv.resolve()}")
    _safe_print(f"已写入 PNG: {args.out_png.resolve()}")
    _safe_print(f"导出 Top {args.top_n} 特征（按 mean_abs_shap 降序）")
    _safe_print("\nTop 10 特征（feature_zh, mean_abs_shap）:")
    for _, row in top_df.iterrows():
        _safe_print(f"  {row['feature_zh']}\t{row['mean_abs_shap']:.6f}")


if __name__ == "__main__":
    main()
