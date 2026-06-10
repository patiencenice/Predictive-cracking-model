"""
图7-5 抗压、抗折强度预测值与实测值对比图（硕士论文用，CBM/JBE 风格）。

数据来源：outputs/lab_strength/lab_strength_oof_predictions.csv
默认预测路径与 lab_strength_residual_report.json 中 default_method_by_task 一致。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF_CSV = PROJECT_ROOT / "outputs" / "lab_strength" / "lab_strength_oof_predictions.csv"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "outputs" / "lab_strength" / "lab_strength_residual_report.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "thesis"


def _setup_thesis_rc() -> None:
    # 中文标签用微软雅黑；英文与数字在 YaHei 下亦可正常显示（论文常用组合）
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
            "axes.linewidth": 0.8,
            "grid.color": "#cccccc",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.2,
            "font.size": 11,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def _load_default_residual_models(report_path: Path) -> dict[str, str]:
    with report_path.open(encoding="utf-8") as f:
        rep = json.load(f)
    dm = rep.get("default_method_by_task") or {}
    out: dict[str, str] = {}
    for task in ("compressive", "flexural"):
        block = dm.get(task) or {}
        learner = block.get("residual_learner")
        out[task] = "formula_only" if learner in (None, "", "null") else str(learner)
    return out


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _task_oof_frame(
    df: pd.DataFrame, task: str, residual_model: str
) -> pd.DataFrame:
    sub = df[(df["task"] == task) & (df["residual_model"] == residual_model)].copy()
    if sub.empty:
        raise ValueError(f"OOF 数据中无 task={task!r}, residual_model={residual_model!r} 记录")
    sub = sub.sort_values("row_id").drop_duplicates(subset=["row_id"], keep="first")
    return sub


def _plot_panel(
    ax: plt.Axes,
    y_measured: np.ndarray,
    y_predicted: np.ndarray,
    *,
    panel_tag: str,
    panel_title: str,
    scatter_color: str,
    scatter_edge: str,
    fixed_limits: tuple[float, float] | None = None,
) -> None:
    y_m = np.asarray(y_measured, dtype=np.float64)
    y_p = np.asarray(y_predicted, dtype=np.float64)
    metrics = _regression_metrics(y_m, y_p)

    ax.scatter(
        y_m,
        y_p,
        s=26,
        c=scatter_color,
        edgecolors=scatter_edge,
        linewidths=0.4,
        alpha=0.75,
        zorder=3,
    )

    if fixed_limits is not None:
        lim_lo, lim_hi = fixed_limits
    else:
        lo = float(min(y_m.min(), y_p.min()))
        hi = float(max(y_m.max(), y_p.max()))
        span = hi - lo
        pad = 0.06 * span if span > 0 else 0.5
        lim_lo, lim_hi = lo - pad, hi + pad

    ax.plot(
        [lim_lo, lim_hi],
        [lim_lo, lim_hi],
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        zorder=2,
        label="y = x",
    )
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("实测值 / MPa", fontsize=11.5)
    ax.set_ylabel("预测值 / MPa", fontsize=11.5)
    ax.set_title(
        f"{panel_tag} {panel_title}",
        loc="left",
        fontweight="normal",
        fontsize=12.5,
        pad=6,
    )
    ax.grid(True, which="major")
    ax.set_axisbelow(True)

    ax.text(
        0.97,
        0.05,
        f"$R^2$ = {metrics['r2']:.3f}\n"
        f"MAE = {metrics['mae']:.3f} MPa\n"
        f"RMSE = {metrics['rmse']:.3f} MPa",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        linespacing=1.28,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="#cccccc",
            linewidth=0.5,
            alpha=0.72,
        ),
    )


def plot_fig7_5(
    oof_csv: Path,
    report_json: Path,
    out_dir: Path,
    *,
    dpi: int = 300,
) -> tuple[Path, Path, Path]:
    _setup_thesis_rc()
    df = pd.read_csv(oof_csv)
    models = _load_default_residual_models(report_json)

    comp = _task_oof_frame(df, "compressive", models["compressive"])
    flex = _task_oof_frame(df, "flexural", models["flexural"])

    # A4 双栏插图常用宽度约 160 mm ≈ 6.3 in；双子图横向排列（图题由 Word 正文撰写）
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.05), dpi=dpi)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.90, wspace=0.28)

    _plot_panel(
        axes[0],
        comp["y_true"].to_numpy(),
        comp["final_pred"].to_numpy(),
        panel_tag="(a)",
        panel_title="抗压强度预测值与实测值对比",
        scatter_color="#4C72B0",
        scatter_edge="#3a5a8c",
    )
    _plot_panel(
        axes[1],
        flex["y_true"].to_numpy(),
        flex["final_pred"].to_numpy(),
        panel_tag="(b)",
        panel_title="抗折强度预测值与实测值对比",
        scatter_color="#DD8452",
        scatter_edge="#b86a42",
        fixed_limits=(3.5, 7.2),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig7_5_strength_pred_vs_measured"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    svg_path = out_dir / f"{stem}.svg"
    save_kw = dict(facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, dpi=dpi, **save_kw)
    fig.savefig(pdf_path, **save_kw)
    fig.savefig(svg_path, **save_kw)
    plt.close(fig)
    return png_path, pdf_path, svg_path


def main() -> None:
    ap = argparse.ArgumentParser(description="绘制图7-5 抗压/抗折强度预测-实测对比图")
    ap.add_argument("--oof-csv", type=Path, default=DEFAULT_OOF_CSV)
    ap.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    png_path, pdf_path, svg_path = plot_fig7_5(
        args.oof_csv,
        args.report_json,
        args.out_dir,
        dpi=args.dpi,
    )
    models = _load_default_residual_models(args.report_json)
    print(f"默认预测路径: 抗压={models['compressive']}, 抗折={models['flexural']}")
    print(f"已保存: {png_path}")
    print(f"已保存: {pdf_path}")
    print(f"已保存: {svg_path}")


if __name__ == "__main__":
    main()
