"""
模型性能评价展示层（只读）。

读取历史离线指标与可选 evaluation 导出文件，在 Streamlit「⑤ 高级分析」中展示。
不参与训练、推理或特征工程。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.mechanism import load_cv_metrics_json, load_training_metrics_json
from src.paths import MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT
from src.ui_theme import apply_chart_theme, research_module_header_html

EVAL_DIR = OUTPUTS_DIR / "evaluation"
LAB_STRENGTH_DIR = OUTPUTS_DIR / "lab_strength"

OFFLINE_DISCLAIMER = (
    "以下为**历史离线指标**（hold-out 或交叉验证汇总），"
    "**不代表当前输入样本的实时误差**；"
    "与侧栏单次预测无直接对应关系。"
)

TRUE_PRED_HINT = (
    "当前缺少 y_true/y_pred 明细文件，需后续由 evaluate 脚本导出。"
    "建议路径：`outputs/evaluation/crack_width_true_pred.csv`、"
    "`outputs/evaluation/crack_density_true_pred.csv`。"
)

CONFUSION_HINT = "当前缺少分类预测明细，暂不展示混淆矩阵。"

METRIC_SOURCES = [
    ("Hold-out 测试集", PROJECT_ROOT / "models" / "training_metrics.json"),
    ("Hold-out 副本", PROJECT_ROOT / "outputs" / "training_metrics.json"),
    ("K 折交叉验证", PROJECT_ROOT / "models" / "cv_metrics.json"),
]


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _read_true_pred_csv(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None
    if df.empty:
        return None
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    y_true_aliases = (
        "y_true",
        "true",
        "actual",
        "y",
        "实测值_mpa",
        "实测值",
    )
    y_pred_aliases = (
        "y_pred",
        "pred",
        "predicted",
        "yhat",
        "预测值_mpa",
        "预测值",
    )
    y_true_key = next(
        (cols_lower[k] for k in y_true_aliases if k in cols_lower),
        None,
    )
    y_pred_key = next(
        (cols_lower[k] for k in y_pred_aliases if k in cols_lower),
        None,
    )
    if y_true_key is None or y_pred_key is None:
        numeric_cols = [
            c
            for c in df.columns
            if pd.to_numeric(df[c], errors="coerce").notna().any()
        ]
        if len(numeric_cols) >= 2:
            y_true_key = numeric_cols[0]
            y_pred_key = numeric_cols[1]
        elif df.shape[1] >= 2:
            y_true_key = df.columns[0]
            y_pred_key = df.columns[1]
        else:
            return None
    yt_s = pd.to_numeric(df[y_true_key], errors="coerce")
    yp_s = pd.to_numeric(df[y_pred_key], errors="coerce")
    mask = yt_s.notna() & yp_s.notna()
    if int(mask.sum()) < 2:
        return None
    return yt_s[mask].to_numpy(dtype=float), yp_s[mask].to_numpy(dtype=float)


def _read_confusion_matrix_csv(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, index_col=0)
    except (OSError, pd.errors.ParserError):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            return None
    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all().all():
        return None
    mat = numeric.fillna(0).to_numpy(dtype=float)
    if mat.ndim != 2 or mat.shape[0] < 1 or mat.shape[1] < 1:
        return None
    return mat


def _build_capability_conclusions(tm: dict[str, Any] | None) -> list[str]:
    """根据 hold-out 指标生成克制结论（禁止夸大）。"""
    if not tm:
        return [
            "未找到 hold-out 指标文件，暂无法给出模型能力结论。",
            "请先完成训练或确认 `models/training_metrics.json` 存在。",
        ]

    lines: list[str] = []
    cw = tm.get("crack_width") if isinstance(tm.get("crack_width"), dict) else {}
    cd = tm.get("crack_density") if isinstance(tm.get("crack_density"), dict) else {}
    cr = tm.get("cracking_risk") if isinstance(tm.get("cracking_risk"), dict) else {}

    r2_w = _safe_float(cw.get("test_r2"))
    r2_d = _safe_float(cd.get("test_r2"))
    acc = _safe_float(cr.get("test_accuracy"))

    if r2_w is not None:
        if r2_w >= 0.35:
            lines.append(
                "**裂缝宽度**：离线 hold-out R² 处于中等区间，"
                "对量级判断具有**中等参考价值**；不宜单独作为验算依据。"
            )
        elif r2_w >= 0.0:
            lines.append(
                "**裂缝宽度**：离线拟合一般，仅宜作**粗量级对照**，"
                "需结合试验与规范限值。"
            )
        else:
            lines.append(
                "**裂缝宽度**：离线 R² 偏低或为负，"
                "当前对数值精度的**参考价值有限**。"
            )
    else:
        lines.append("**裂缝宽度**：缺少有效离线 R²，暂不评价。")

    if r2_d is not None:
        if r2_d < 0.0:
            lines.append(
                "**裂缝密度**：离线 R² 为负，**当前稳定性较弱**，"
                "不宜用于精细定量对比。"
            )
        elif r2_d < 0.2:
            lines.append(
                "**裂缝密度**：离线解释度偏低，**稳定性偏弱**，"
                "建议以趋势与区间为主。"
            )
        else:
            lines.append(
                "**裂缝密度**：离线指标波动仍较大，"
                "宜作**辅助参考**，不宜替代现场统计。"
            )
    else:
        lines.append("**裂缝密度**：缺少有效离线 R²，暂不评价。")

    if acc is not None:
        lines.append(
            "**开裂风险分类**：离线准确率约 "
            f"{acc:.0%}，**仅作趋势提示**；"
            "类别边界样本少时易误判，须结合工程判据。"
        )
    else:
        lines.append("**开裂风险**：缺少离线分类指标，暂不评价。")

    lines.append(
        "_以上结论仅依据历史 hold-out JSON，"
        "不承诺对当前输入的实时误差水平。_"
    )
    return lines


def _cv_summary_line(cv: dict[str, Any] | None, task_key: str, mode: str) -> str:
    if not cv:
        return "—"
    summary = cv.get("summary_mean_std") or {}
    block = summary.get(task_key) if isinstance(summary.get(task_key), dict) else {}
    if mode == "reg":
        r2m = _safe_float(block.get("r2_mean"))
        r2s = _safe_float(block.get("r2_std"))
        if r2m is None:
            return "—"
        std = f" ± {r2s:.3f}" if r2s is not None else ""
        return f"CV R² mean={r2m:.3f}{std}"
    accm = _safe_float(block.get("accuracy_mean"))
    accs = _safe_float(block.get("accuracy_std"))
    if accm is None:
        return "—"
    std = f" ± {accs:.3f}" if accs is not None else ""
    return f"CV Acc mean={accm:.3f}{std}"


def _fig_true_pred_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> go.Figure:
    m = _reg_metrics(y_true, y_pred)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    span = hi - lo
    pad = 0.05 * span if span > 0 else 0.05
    lim_lo, lim_hi = lo - pad, hi + pad

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="测试样本",
            marker=dict(size=9, opacity=0.78, color="#2563eb"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lim_lo, lim_hi],
            y=[lim_lo, lim_hi],
            mode="lines",
            name="y = x",
            line=dict(color="#64748b", dash="dash", width=1.5),
        )
    )
    ann = (
        f"R² = {m['r2']:.4f}<br>"
        f"RMSE = {m['rmse']:.4f}<br>"
        f"MAE = {m['mae']:.4f}"
    )
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=ann,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        font=dict(size=11),
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(range=[lim_lo, lim_hi])
    fig.update_yaxes(range=[lim_lo, lim_hi])
    apply_chart_theme(fig, height=380)
    return fig


def _fig_residual_hist(residuals: np.ndarray, *, title: str) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=residuals,
            nbinsx=min(20, max(8, len(residuals) // 3)),
            marker_color="#0ea5e9",
            opacity=0.85,
        )
    )
    fig.update_layout(title=title, xaxis_title="残差 (预测 − 实测)", yaxis_title="频数")
    apply_chart_theme(fig, height=320)
    return fig


def _fig_residual_vs_pred(
    y_pred: np.ndarray, residuals: np.ndarray, *, title: str
) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(size=9, opacity=0.78, color="#7c3aed"),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1.2)
    fig.update_layout(
        title=title,
        xaxis_title="预测值",
        yaxis_title="残差 (预测 − 实测)",
    )
    apply_chart_theme(fig, height=320)
    return fig


def _fig_confusion_heatmap(mat: np.ndarray, labels: list[str] | None) -> go.Figure:
    n, m = mat.shape
    if labels is None or len(labels) != n:
        labels = [str(i) for i in range(n)]
    col_labels = labels if m == n else [str(j) for j in range(m)]
    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=col_labels,
            y=labels,
            colorscale="Blues",
            text=np.round(mat, 0).astype(int),
            texttemplate="%{text}",
            textfont={"size": 11},
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="开裂风险 · 混淆矩阵（离线导出）",
        xaxis_title="预测类别",
        yaxis_title="实测类别",
    )
    apply_chart_theme(fig, height=400)
    return fig


def _render_metric_cards(tm: dict[str, Any] | None, cv: dict[str, Any] | None) -> None:
    st.markdown("##### 性能指标卡片（Hold-out 测试集）")
    if not tm:
        st.warning(
            "未找到 `models/training_metrics.json`。"
            "请先运行训练脚本生成历史指标；**此处不展示编造数值**。"
        )
        return

    n_train = tm.get("n_train")
    n_test = tm.get("n_test")
    if n_train is not None and n_test is not None:
        st.caption(f"划分规模：n_train={n_train}，n_test={n_test}（与当前单次预测无关）")

    c1, c2, c3 = st.columns(3)
    tasks = (
        ("裂缝宽度", "crack_width", "reg", c1),
        ("裂缝密度", "crack_density", "reg", c2),
        ("开裂风险分类", "cracking_risk", "cls", c3),
    )
    for title, key, mode, col in tasks:
        block = tm.get(key) if isinstance(tm.get(key), dict) else {}
        with col:
            st.markdown(f"**{title}**")
            if mode == "reg":
                r2 = _safe_float(block.get("test_r2"))
                rmse = _safe_float(block.get("test_rmse"))
                mae = _safe_float(block.get("test_mae"))
                st.metric("R²", f"{r2:.4f}" if r2 is not None else "—")
                st.metric("RMSE", f"{rmse:.4f}" if rmse is not None else "—")
                st.metric("MAE", f"{mae:.4f}" if mae is not None else "—")
            else:
                acc = _safe_float(block.get("test_accuracy"))
                mf1 = _safe_float(block.get("test_macro_f1"))
                wf1 = _safe_float(block.get("test_weighted_f1"))
                st.metric("Accuracy", f"{acc:.4f}" if acc is not None else "—")
                st.metric("Macro F1", f"{mf1:.4f}" if mf1 is not None else "—")
                st.metric("Weighted F1", f"{wf1:.4f}" if wf1 is not None else "—")
            st.caption(_cv_summary_line(cv, key, mode))


def _render_regression_block(
    task_label: str,
    csv_path: Path,
    *,
    x_label: str,
    y_label: str,
    plot_fn: Callable[[go.Figure, str], None],
    key_prefix: str,
) -> None:
    st.markdown(f"##### {task_label}")
    pair = _read_true_pred_csv(csv_path)
    if pair is None:
        st.info(TRUE_PRED_HINT)
        st.caption(f"期望文件：`{csv_path}`")
        return
    y_true, y_pred = pair
    plot_fn(
        _fig_true_pred_scatter(
            y_true,
            y_pred,
            title=f"{task_label} · 实测 vs 预测",
            x_label=x_label,
            y_label=y_label,
        ),
        f"{key_prefix}_scatter",
    )
    residuals = y_pred - y_true
    c1, c2 = st.columns(2)
    with c1:
        plot_fn(
            _fig_residual_hist(residuals, title=f"{task_label} · 残差分布"),
            f"{key_prefix}_res_hist",
        )
    with c2:
        plot_fn(
            _fig_residual_vs_pred(
                y_pred, residuals, title=f"{task_label} · 残差 vs 预测值"
            ),
            f"{key_prefix}_res_scatter",
        )


def _render_residual_pair(
    task_label: str,
    csv_path: Path,
    *,
    plot_fn: Callable[[go.Figure, str], None],
    key_prefix: str,
) -> bool:
    """仅残差图（用于「3 · 残差分析」分块）。返回是否成功绘图。"""
    pair = _read_true_pred_csv(csv_path)
    if pair is None:
        return False
    y_true, y_pred = pair
    residuals = y_pred - y_true
    st.markdown(f"**{task_label}**")
    c1, c2 = st.columns(2)
    with c1:
        plot_fn(
            _fig_residual_hist(residuals, title=f"{task_label} · 残差分布"),
            f"{key_prefix}_res_hist",
        )
    with c2:
        plot_fn(
            _fig_residual_vs_pred(
                y_pred, residuals, title=f"{task_label} · 残差 vs 预测值"
            ),
            f"{key_prefix}_res_scatter",
        )
    return True


def render_model_performance_evaluation(
    plot_fn: Callable[[go.Figure, str], None] | None = None,
) -> None:
    """
    在 Streamlit 中渲染「模型性能评价」模块。

    plot_fn: (figure, streamlit_key) -> None；须为每个图表传入唯一 key，避免残差图被覆盖。
    """
    if plot_fn is None:
        from src.ui_theme import PLOTLY_CONFIG

        def _default_plot(fig: go.Figure, key: str) -> None:
            st.plotly_chart(
                fig, use_container_width=True, config=PLOTLY_CONFIG, key=key
            )

        plot_fn = _default_plot

    st.markdown(
        research_module_header_html(
            "模型性能评价",
            "历史训练效果与离线预测误差展示",
            badge="Metrics",
            tone="info",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sfc-rd-note">'
        "<strong>只读展示层</strong> · 读取既有 JSON/CSV，"
        "不重训、不改推理权重、不生成合成数据。</p>",
        unsafe_allow_html=True,
    )
    st.caption(OFFLINE_DISCLAIMER)

    tm_models = load_training_metrics_json(MODELS_DIR)
    tm_outputs = _load_json(OUTPUTS_DIR / "training_metrics.json")
    cv = load_cv_metrics_json(MODELS_DIR)
    tm = tm_models or tm_outputs

    with st.expander("1 · 性能指标卡片", expanded=True):
        if tm_models and tm_outputs and tm_models != tm_outputs:
            st.caption(
                "注意：`models/` 与 `outputs/` 下 training_metrics 内容不一致，"
                "卡片以 `models/training_metrics.json` 为准。"
            )
        _render_metric_cards(tm, cv)

    with st.expander("2 · 真实值-预测值散点图", expanded=False):
        _render_regression_block(
            "裂缝宽度",
            EVAL_DIR / "crack_width_true_pred.csv",
            x_label="实测裂缝宽度 (mm)",
            y_label="预测裂缝宽度 (mm)",
            plot_fn=plot_fn,
            key_prefix="sfc_perf_width",
        )
        st.markdown("---")
        _render_regression_block(
            "裂缝密度",
            EVAL_DIR / "crack_density_true_pred.csv",
            x_label="实测裂缝密度 (条/m²)",
            y_label="预测裂缝密度 (条/m²)",
            plot_fn=plot_fn,
            key_prefix="sfc_perf_density",
        )
        st.markdown("---")
        _render_regression_block(
            "抗压强度（OOF）",
            LAB_STRENGTH_DIR / "lab_strength_compressive_true_pred.csv",
            x_label="实测抗压强度 (MPa)",
            y_label="预测抗压强度 (MPa)",
            plot_fn=plot_fn,
            key_prefix="sfc_perf_comp",
        )
        st.markdown("---")
        _render_regression_block(
            "抗折强度（OOF）",
            LAB_STRENGTH_DIR / "lab_strength_flexural_true_pred.csv",
            x_label="实测抗折强度 (MPa)",
            y_label="预测抗折强度 (MPa)",
            plot_fn=plot_fn,
            key_prefix="sfc_perf_flex",
        )

    with st.expander("3 · 残差分析", expanded=False):
        st.caption(
            "主开裂任务来自 `outputs/evaluation/`（evaluate 导出）；"
            "力学强度来自 `outputs/lab_strength/`（export_lab_strength_true_pred 导出）。"
        )
        any_residual = False
        if _render_residual_pair(
            "裂缝宽度",
            EVAL_DIR / "crack_width_true_pred.csv",
            plot_fn=plot_fn,
            key_prefix="sfc_perf_res_width",
        ):
            any_residual = True
        if _render_residual_pair(
            "裂缝密度",
            EVAL_DIR / "crack_density_true_pred.csv",
            plot_fn=plot_fn,
            key_prefix="sfc_perf_res_density",
        ):
            any_residual = True
        if _render_residual_pair(
            "抗压强度（OOF）",
            LAB_STRENGTH_DIR / "lab_strength_compressive_true_pred.csv",
            plot_fn=plot_fn,
            key_prefix="sfc_perf_res_comp",
        ):
            any_residual = True
        if _render_residual_pair(
            "抗折强度（OOF）",
            LAB_STRENGTH_DIR / "lab_strength_flexural_true_pred.csv",
            plot_fn=plot_fn,
            key_prefix="sfc_perf_res_flex",
        ):
            any_residual = True
            st.caption(
                "抗折默认路径为仅公式基线时，残差修正恒为 0，残差分布将集中在 0 附近。"
            )
        if not any_residual:
            st.info(
                "当前缺少 y_true/y_pred 明细，无法绘制残差图。"
                "主开裂：运行 `py evaluate.py --csv data/training_data.csv`；"
                "力学：运行 `py scripts/export_lab_strength_true_pred.py`。"
            )

    with st.expander("4 · 分类混淆矩阵", expanded=False):
        cm_path = EVAL_DIR / "cracking_risk_confusion_matrix.csv"
        mat = _read_confusion_matrix_csv(cm_path)
        if mat is None:
            st.info(CONFUSION_HINT)
            st.caption(f"期望文件：`{cm_path}`")
        else:
            try:
                df_cm = pd.read_csv(cm_path, index_col=0)
                row_labels = [str(x) for x in df_cm.index.tolist()]
            except (OSError, pd.errors.ParserError):
                row_labels = None
            plot_fn(_fig_confusion_heatmap(mat, row_labels), "sfc_perf_confusion")

    with st.expander("5 · 模型能力结论", expanded=True):
        for line in _build_capability_conclusions(tm):
            st.markdown(line)

    with st.expander("6 · 高级说明", expanded=False):
        st.markdown("**指标来源说明**")
        st.markdown(
            "- **Hold-out**：`models/training_metrics.json`（训练脚本写入；"
            "`outputs/training_metrics.json` 为同步副本）。\n"
            "- **交叉验证**：`models/cv_metrics.json`（`py -m src.cross_validate` 生成；"
            "可能与当前 pkl 权重**不同批次**，仅作稳定性参考）。\n"
            "- **散点/残差/混淆矩阵**：`outputs/evaluation/` 下 CSV（需 evaluate 导出；"
            "本模块不触发训练或推理）。"
        )
        st.markdown("**文件路径**")
        for label, p in METRIC_SOURCES:
            status = "存在" if p.is_file() else "缺失"
            st.markdown(f"- {label}：`{p}`（{status}）")
        for name in (
            "crack_width_true_pred.csv",
            "crack_density_true_pred.csv",
            "cracking_risk_confusion_matrix.csv",
        ):
            p = EVAL_DIR / name
            status = "存在" if p.is_file() else "缺失"
            st.markdown(f"- 评估明细：`{p}`（{status}）")
        for name in (
            "lab_strength_compressive_true_pred.csv",
            "lab_strength_flexural_true_pred.csv",
        ):
            p = LAB_STRENGTH_DIR / name
            status = "存在" if p.is_file() else "缺失"
            st.markdown(f"- 力学 OOF 明细：`{p}`（{status}）")

        st.markdown("**JSON 原始指标**")
        st.markdown("*Hold-out · models/training_metrics.json*")
        if tm_models:
            st.json(tm_models)
        else:
            st.caption("未找到 models/training_metrics.json")
        st.markdown("*Hold-out 副本 · outputs/training_metrics.json*")
        if tm_outputs:
            st.json(tm_outputs)
        else:
            st.caption("未找到 outputs/training_metrics.json")
        st.markdown("*交叉验证 · models/cv_metrics.json*")
        if cv:
            st.json(cv)
        else:
            st.caption("未找到 models/cv_metrics.json")
