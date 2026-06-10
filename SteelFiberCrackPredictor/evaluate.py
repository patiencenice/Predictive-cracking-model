"""
评估脚本：加载 models/ 下已保存的 scaler 与三模型，在 CSV 上复现训练时的划分方式，
计算回归（MAE/RMSE/R2）与分类（准确率、F1、混淆矩阵）指标，并保存到 outputs/。

重要 — 与本机 models/*.pkl 对齐的推荐命令（勿误用 example 表解释当前模型性能）:
  py evaluate.py --csv data/training_data.csv

  data/training_data.example.csv 仅用于示例或演示，不应用来解释当前正式 pkl 的性能。

无 --csv 时默认仍读取 training_data.example.csv（便于快速试跑脚本）。

用法:
  py evaluate.py --csv data/training_data.csv          # 推荐：与本机 pkl 对齐
  py evaluate.py                                        # 默认 example 表，仅试跑
  py evaluate.py --csv data/training_data.csv --models-dir models
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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import FEATURE_COLUMNS
from src.paths import CONFIG_YAML, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT
from src.train_utils import load_model_config

TARGET_COLS = ("crack_width", "crack_density", "cracking_risk")
EVAL_SUBDIR = "evaluation"
RISK_CLASS_LABELS = {0: "低风险", 1: "中风险", 2: "高风险"}


def _risk_label(class_id: int) -> str:
    return RISK_CLASS_LABELS.get(int(class_id), str(class_id))


def _save_regression_true_pred_csv(
    path: Path,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    sample_index: np.ndarray,
) -> None:
    """导出回归任务 y_true/y_pred 明细（含残差与样本索引）。"""
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    idx_arr = np.asarray(sample_index).ravel()
    pd.DataFrame(
        {
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
            "residual": y_pred_arr - y_true_arr,
            "sample_index": idx_arr.astype(int),
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def _save_cracking_risk_true_pred_csv(
    path: Path,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    sample_index: np.ndarray,
) -> None:
    """导出分类任务 y_true/y_pred 明细（含预测类别文字标签）。"""
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=int).ravel()
    idx_arr = np.asarray(sample_index).ravel()
    pd.DataFrame(
        {
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
            "y_pred_label": [_risk_label(v) for v in y_pred_arr],
            "sample_index": idx_arr.astype(int),
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def _save_confusion_matrix_csv(
    path: Path,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> None:
    """导出混淆矩阵 CSV：行=真实类别，列=预测类别，值=样本数。"""
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=int).ravel()
    labels = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    pd.DataFrame(
        cm,
        index=[str(label) for label in labels],
        columns=[str(label) for label in labels],
    ).to_csv(path, encoding="utf-8-sig")


def _save_r2_scatter_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r2: float,
    out_path: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """回归任务：横轴真实值、纵轴预测值，叠加 y=x 参考线并标注 R²。"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(y_true, y_pred, alpha=0.75, s=36, edgecolors="white", linewidths=0.3)
    # 坐标范围：取真值与预测的共同范围，并留少量边距
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    span = hi - lo
    pad = 0.05 * span if span > 0 else 0.05
    lim_lo, lim_hi = lo - pad, hi + pad
    # 理想吻合线 y=x（真实值=预测值）
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="gray", linestyle="--", lw=1.2)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # 左上角标注 R²（与 evaluate_report 中 test_r2 一致）
    ax.text(
        0.04,
        0.96,
        f"$R^2$ = {r2:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "评估已保存模型并输出指标到 outputs/。"
            "与本机 pkl 对齐请指定: py evaluate.py --csv data/training_data.csv"
        ),
        epilog=(
            "说明:\n"
            "  data/training_data.example.csv 仅用于示例或演示，"
            "不应用来解释当前正式 pkl 的性能。\n"
            "推荐: py evaluate.py --csv data/training_data.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "training_data.example.csv",
        help=(
            "评估用 CSV（默认 training_data.example.csv 仅供试跑）。"
            "与本机 pkl 对齐请用 data/training_data.csv"
        ),
    )
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="含 feature_scaler.pkl 与 crack_*.pkl 的目录",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUTPUTS_DIR,
        help="评估报表输出目录",
    )
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"未找到数据文件: {args.csv}")

    _example_csv = (PROJECT_ROOT / "data" / "training_data.example.csv").resolve()
    if args.csv.resolve() == _example_csv:
        print(
            "提示: 当前使用 data/training_data.example.csv（仅示例/演示），"
            "不应用来解释当前正式 pkl 的性能。"
            "推荐: py evaluate.py --csv data/training_data.csv",
            file=sys.stderr,
        )

    # 载入配置，保证划分比例与随机种子与训练一致
    cfg = load_model_config(CONFIG_YAML)
    tcfg = cfg.get("training", {})
    test_size = float(tcfg.get("test_size", 0.2))
    rs = int(tcfg.get("random_state", 42))
    stratify_flag = bool(tcfg.get("stratify_risk", True))

    df = pd.read_csv(args.csv)
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            raise SystemExit(f"CSV 缺少特征列: {c}")
    for c in TARGET_COLS:
        if c not in df.columns:
            raise SystemExit(f"CSV 缺少目标列: {c}")

    Xo = df[FEATURE_COLUMNS].astype(np.float64, copy=False)
    y_w = df["crack_width"].astype(float)
    y_d = df["crack_density"].astype(float)
    y_r = df["cracking_risk"].astype(int)

    stratify = y_r if stratify_flag else None
    sample_idx = np.arange(len(df))
    try:
        split = train_test_split(
            Xo,
            y_w,
            y_d,
            y_r,
            sample_idx,
            test_size=test_size,
            random_state=rs,
            stratify=stratify,
        )
    except ValueError:
        split = train_test_split(
            Xo,
            y_w,
            y_d,
            y_r,
            sample_idx,
            test_size=test_size,
            random_state=rs,
            stratify=None,
        )
    X_train, X_test, yw_tr, yw_te, yd_tr, yd_te, yr_tr, yr_te, _, idx_te = split

    # 加载训练阶段保存的预处理器与估计器（非 Pipeline 文件，与 Web 推理一致）
    scaler = joblib.load(args.models_dir / "feature_scaler.pkl")
    reg_w = joblib.load(args.models_dir / "crack_regressor.pkl")
    reg_d_path = args.models_dir / "crack_density_regressor.pkl"
    reg_d = joblib.load(reg_d_path) if reg_d_path.exists() else None
    clf = joblib.load(args.models_dir / "crack_classifier.pkl")

    X_test_s = scaler.transform(X_test)

    report: dict = {
        "csv": str(args.csv.resolve()),
        "models_dir": str(args.models_dir.resolve()),
        "n_test": int(len(X_test)),
        "partition": {"test_size": test_size, "random_state": rs},
    }

    # —— 回归：裂缝宽度 ——
    pred_w = reg_w.predict(X_test_s)
    report["crack_width"] = {
        "task": "regression",
        "test_mae": float(mean_absolute_error(yw_te, pred_w)),
        "test_rmse": float(np.sqrt(mean_squared_error(yw_te, pred_w))),
        "test_r2": float(r2_score(yw_te, pred_w)),
    }

    # —— 回归：裂缝密度 ——
    if reg_d is not None:
        pred_d = reg_d.predict(X_test_s)
        report["crack_density"] = {
            "task": "regression",
            "test_mae": float(mean_absolute_error(yd_te, pred_d)),
            "test_rmse": float(np.sqrt(mean_squared_error(yd_te, pred_d))),
            "test_r2": float(r2_score(yd_te, pred_d)),
        }
    else:
        report["crack_density"] = {"task": "regression", "skipped": True}

    # —— 分类：开裂风险 ——
    pred_r = clf.predict(X_test_s)
    report["cracking_risk"] = {
        "task": "multiclass_classification",
        "test_accuracy": float(accuracy_score(yr_te, pred_r)),
        "test_macro_f1": float(
            f1_score(yr_te, pred_r, average="macro", zero_division=0)
        ),
        "test_weighted_f1": float(
            f1_score(yr_te, pred_r, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            yr_te, pred_r, zero_division=0, output_dict=True
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "evaluate_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        # classification_report 内可能含 numpy 标量，统一可序列化
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # 混淆矩阵图（分类任务）
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        yr_te, pred_r, ax=ax, colorbar=True
    )
    # 标题使用英文避免无中文字体时乱码（报告内可另附中文说明）
    ax.set_title("Cracking risk confusion matrix (test set)")
    fig.tight_layout()
    cm_path = args.out / "confusion_matrix_cracking_risk.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # 回归任务 R² 散点图：直接使用上文测试集 y_*_te 与 pred_*，保证与报表一致
    r2_w = report["crack_width"]["test_r2"]
    path_r2_w = args.out / "r2_crack_width.png"
    _save_r2_scatter_plot(
        yw_te,
        pred_w,
        r2_w,
        path_r2_w,
        xlabel="True crack width (mm)",
        ylabel="Predicted crack width (mm)",
        title="crack_width: true vs pred (test)",
    )
    if reg_d is not None:
        r2_d = report["crack_density"]["test_r2"]
        path_r2_d = args.out / "r2_crack_density.png"
        _save_r2_scatter_plot(
            yd_te,
            pred_d,
            r2_d,
            path_r2_d,
            xlabel="True crack density (per m²)",
            ylabel="Predicted crack density (per m²)",
            title="crack_density: true vs pred (test)",
        )

    # —— 可视化明细 CSV（供 model_performance_panel 读取）——
    eval_dir = args.out / EVAL_SUBDIR
    eval_dir.mkdir(parents=True, exist_ok=True)

    path_tp_w = eval_dir / "crack_width_true_pred.csv"
    _save_regression_true_pred_csv(path_tp_w, yw_te, pred_w, idx_te)
    print(f"Saved: {path_tp_w.resolve()}")

    if reg_d is not None:
        path_tp_d = eval_dir / "crack_density_true_pred.csv"
        _save_regression_true_pred_csv(path_tp_d, yd_te, pred_d, idx_te)
        print(f"Saved: {path_tp_d.resolve()}")

    path_tp_r = eval_dir / "cracking_risk_true_pred.csv"
    _save_cracking_risk_true_pred_csv(path_tp_r, yr_te, pred_r, idx_te)
    print(f"Saved: {path_tp_r.resolve()}")

    path_cm_csv = eval_dir / "cracking_risk_confusion_matrix.csv"
    _save_confusion_matrix_csv(path_cm_csv, yr_te, pred_r)
    print(f"Saved: {path_cm_csv.resolve()}")

    print(f"Saved: {out_json.resolve()}")
    print(f"Saved: {cm_path.resolve()}")
    print(f"Saved: {path_r2_w.resolve()}")
    if reg_d is not None:
        print(f"Saved: {path_r2_d.resolve()}")


if __name__ == "__main__":
    main()
