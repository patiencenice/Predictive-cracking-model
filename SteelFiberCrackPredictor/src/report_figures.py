"""
Word 报告用静态图：Matplotlib 导出 PNG（不依赖 kaleido），附自动生成的简要分析语句。
"""

from __future__ import annotations

import io
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.ui_theme import COLORS


def _cn_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def importance_bar_png_bytes(
    labels_cn: Sequence[str],
    values: Sequence[float],
    title: str,
    *,
    figsize: tuple[float, float] = (7.2, 4.2),
    dpi: int = 140,
) -> io.BytesIO:
    """横向条形图：特征重要性（已归一化）。"""
    _cn_font()
    vals = [float(v) for v in values]
    labs = list(labels_cn)
    n = len(labs)
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n)
    ax.barh(y, vals, color=COLORS["primary_light"], height=0.62, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("相对重要性（归一化）", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", color=COLORS["primary"])
    ax.grid(axis="x", alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    fig.patch.set_facecolor(COLORS["plot"])
    ax.set_facecolor(COLORS["paper"])
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def shap_bar_png_bytes(
    labels_cn: Sequence[str],
    shap_values: Sequence[float],
    title: str,
    *,
    figsize: tuple[float, float] = (7.2, 4.5),
    dpi: int = 140,
) -> io.BytesIO:
    """SHAP 水平条形图（红正蓝负）。"""
    _cn_font()
    labs = list(labels_cn)
    vals = np.array([float(v) for v in shap_values], dtype=np.float64)
    colors = np.where(vals >= 0, COLORS["danger"], COLORS["primary_light"])
    n = len(labs)
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n)
    ax.barh(y, vals, color=colors, height=0.58, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color=COLORS["text_muted"], linewidth=0.8)
    ax.set_xlabel("SHAP 值", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", color=COLORS["primary"])
    ax.grid(axis="x", alpha=0.35, linestyle="--")
    fig.patch.set_facecolor(COLORS["plot"])
    ax.set_facecolor(COLORS["paper"])
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def pdp_line_png_bytes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (6.8, 3.8),
    dpi: int = 140,
) -> io.BytesIO:
    """部分依赖曲线。"""
    _cn_font()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, color=COLORS["primary"], linewidth=2.2, marker="o", markersize=3)
    ax.fill_between(x, y, alpha=0.12, color=COLORS["primary"])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", color=COLORS["primary"])
    ax.grid(alpha=0.35, linestyle="--")
    fig.patch.set_facecolor(COLORS["plot"])
    ax.set_facecolor(COLORS["paper"])
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def ice_line_png_bytes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (6.8, 3.8),
    dpi: int = 140,
) -> io.BytesIO:
    """局部响应（ICE 风格）曲线。"""
    _cn_font()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, color=COLORS["success"], linewidth=2.2, marker="o", markersize=3)
    ax.fill_between(x, y, alpha=0.14, color=COLORS["success"])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", color=COLORS["primary"])
    ax.grid(alpha=0.35, linestyle="--")
    fig.patch.set_facecolor(COLORS["plot"])
    ax.set_facecolor(COLORS["paper"])
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def text_analysis_importance(
    task_cn: str, top_names: Sequence[str], top_vals: Sequence[float]
) -> str:
    """依据条形图上的归一化长度：集中度、头名与次名倍数、前三累计占比。"""
    if not top_names:
        return f"【{task_cn}】无有效重要性数据。"
    vals = np.array([float(v) for v in top_vals], dtype=np.float64)
    names = list(top_names)
    s = float(vals.sum()) or 1.0
    p = vals / s
    k = min(3, len(names))
    parts = [f"{names[i]}（{p[i]:.3f}）" for i in range(k)]
    v1 = float(p[0])
    v2 = float(p[1]) if len(p) > 1 else 0.0
    top3_share = float(p[: min(3, len(p))].sum())
    ratio12 = v1 / (v2 + 1e-12) if len(p) > 1 else 1.0

    if v1 >= 0.40:
        conc = "条形长度显示：重要性明显向首项集中，单一特征在图中占主导。"
    elif v1 >= 0.22:
        conc = "首项仍领先，但其余条形仍具可观长度，属「首项略突出、多因素并存」形态。"
    else:
        conc = "各条形长度差距相对较小，模型更依赖多特征共同解释，不宜只盯单一变量。"

    cmp12 = ""
    if len(names) > 1 and v2 > 1e-12:
        cmp12 = f"与上图一致：第一名约为第二名的 {ratio12:.2f} 倍（归一化份额 {v1:.3f} 对 {v2:.3f}）。"

    return (
        f"【{task_cn}·对照条形图】图中横轴为相对重要性（展示段内已归一化）。"
        f"最长的前三项为：{'、'.join(parts)}，三者合计约占本图展示项的 {top3_share*100:.1f}%。"
        f"{conc} {cmp12}"
        "解读须结合训练数据分布：条形只反映增益统计排序，工程上仍应对照配合比设计与试验验证。"
    )


def text_analysis_shap(
    pred_class: int,
    names_cn: Sequence[str],
    values: Sequence[float],
) -> str:
    """对照 SHAP 条形图：条长、红蓝侧累计、最大 |SHAP| 占比。"""
    vals = np.array([float(v) for v in values], dtype=np.float64)
    labs = list(names_cn)
    if vals.size == 0:
        return f"【开裂风险 SHAP】类别 {pred_class} 无有效 SHAP 向量。"

    abs_v = np.abs(vals)
    tot = float(abs_v.sum()) or 1.0
    top1_share = float(abs_v[0] / tot)
    pos_mask = vals > 0
    neg_mask = vals < 0
    pos_sum = float(vals[pos_mask].sum()) if np.any(pos_mask) else 0.0
    neg_sum = float(vals[neg_mask].sum()) if np.any(neg_mask) else 0.0

    pos = [(labs[i], float(vals[i])) for i in range(len(vals)) if vals[i] > 0.0]
    neg = [(labs[i], float(vals[i])) for i in range(len(vals)) if vals[i] < 0.0]
    pos.sort(key=lambda t: -t[1])
    neg.sort(key=lambda t: t[1])
    pos_s = "、".join(f"{a}（+{b:.4f}）" for a, b in pos[:3]) or "—"
    neg_s = "、".join(f"{a}（{b:.4f}）" for a, b in neg[:3]) or "—"

    balance = ""
    if pos_sum > 0 and abs(neg_sum) > 0:
        balance = (
            f"将红条（正）与蓝条（负）的数值分别累加，约得正向合计 {pos_sum:.4f}、"
            f"负向合计 {neg_sum:.4f}，与图中两侧条带总长度对比一致。"
        )
    dom = (
        f"图中条形按 |SHAP| 排序：最长条为「{labs[0]}」（|SHAP|={abs_v[0]:.4f}），"
        f"约占本图展示项 |SHAP| 总和的 {top1_share*100:.1f}%。"
    )

    return (
        f"【开裂风险 SHAP·对照条形图】当前预测类别 {pred_class}：{dom}"
        f"SHAP>0（红）推高该类别概率，典型项：{pos_s}；"
        f"SHAP<0（蓝）拉低该类别概率，典型项：{neg_s}。"
        f"{balance}"
        "条形长度为贡献量级；解释限于本模型与样本分布，不等同于因果。"
    )


def text_analysis_pdp(
    feat_label_cn: str,
    y_label: str,
    gy: np.ndarray,
    gx: np.ndarray | None = None,
) -> str:
    """依据 PDP 折线：端点差、峰谷、单调性与图中 y 轴量级。"""
    gy = np.asarray(gy, dtype=np.float64).ravel()
    if gy.size < 2:
        return f"【PDP】特征「{feat_label_cn}」有效网格点不足，无法判断边际趋势。"
    y_min = float(np.nanmin(gy))
    y_max = float(np.nanmax(gy))
    y_rng = y_max - y_min
    delta = float(gy[-1] - gy[0])
    d1 = np.diff(gy)
    if d1.size == 0:
        shape = "单点，无法描述线形。"
    elif np.all(np.abs(d1) <= 1e-10):
        shape = "平均线近似水平，与图中近常数折线一致。"
    elif np.all(d1 >= -1e-10):
        shape = "平均线阶梯式非降，与图中整体向上或持平折线一致。"
    elif np.all(d1 <= 1e-10):
        shape = "平均线阶梯式非增，与图中整体向下或持平折线一致。"
    else:
        sign_flip = int(np.sum(d1[1:] * d1[:-1] < 0)) if d1.size > 1 else 0
        shape = (
            f"平均线存在升降转折（斜率变号约 {sign_flip} 处），与图中起伏折线一致。"
        )

    span = y_rng + 1e-12
    rel = abs(delta) / span
    if y_rng < 1e-12 * (abs(y_max) + 1.0):
        trend_note = "图中纵轴变化极小，平均线近乎水平，边际敏感性弱。"
    elif rel < 0.08:
        trend_note = "首尾纵轴值接近，整体起伏有限；若中部有峰/谷，请以图中拐点为准。"
    elif delta > 0:
        trend_note = "横轴自左至右增大时，平均线终点高于起点，与上升趋势一致。"
    else:
        trend_note = "横轴自左至右增大时，平均线终点低于起点，与下降趋势一致。"

    gx_note = ""
    if gx is not None:
        gx = np.asarray(gx, dtype=np.float64).ravel()
        if gx.size == gy.size:
            gx_note = f"图中横轴约从 {gx[0]:.3f} 到 {gx[-1]:.3f}（标准化空间）。"

    return (
        f"【PDP·{y_label}·对照曲线图】特征「{feat_label_cn}」：平均线在纵轴上约处于 "
        f"{y_min:.4f}～{y_max:.4f} 区间，极差约 {y_rng:.4f}（与图中曲线高低范围一致）。{gx_note}"
        f"{shape}。{trend_note}"
        "PDP 为背景样本上的平均边际效应，不表示单一样本上的因果链。"
    )


def text_analysis_ice(
    feat_label_cn: str,
    y_label: str,
    yv: np.ndarray,
    xv: np.ndarray | None = None,
) -> str:
    """依据 ICE 风格曲线：预测值范围、斜率方向、与图中绿色折线一致。"""
    yv = np.asarray(yv, dtype=np.float64).ravel()
    if yv.size < 2:
        return f"【局部曲线】特征「{feat_label_cn}」扫描点不足。"
    y_min = float(np.nanmin(yv))
    y_max = float(np.nanmax(yv))
    delta = float(yv[-1] - yv[0])
    mid = float(np.median(yv))
    span_ref = max(abs(y_max), abs(y_min), 1e-9)
    rel_chg = abs(delta) / span_ref

    if abs(delta) < 1e-9 * span_ref:
        return (
            f"【局部曲线·{y_label}·对照图】固定其余为当前工况：图中纵轴预测值约在 "
            f"{y_min:.4f}～{y_max:.4f} 之间，首尾几乎重合，曲线呈平台状，敏感性弱。"
        )

    d1 = np.diff(yv)
    sign_flip = int(np.sum(d1[1:] * d1[:-1] < 0)) if d1.size > 1 else 0
    if sign_flip > 0:
        shape = f"折线在扫描区间内存在起伏（约 {sign_flip} 处转折），与图中非直线形态一致。"
    else:
        shape = "折线整体单向变化，无显著回折。"

    d = "增大" if delta > 0 else "减小"
    x_note = ""
    if xv is not None:
        xa = np.asarray(xv, dtype=np.float64).ravel()
        if xa.size == yv.size:
            x_note = f"横轴扫描约 [{xa[0]:.3f}, {xa[-1]:.3f}]（标准化）。"

    return (
        f"【局部曲线·{y_label}·对照图】当前工况下单变量扫描：{x_note}"
        f"纵轴预测约在 {y_min:.4f}～{y_max:.4f}，中位约 {mid:.4f}；"
        f"自左至右终点相对起点 {d}，变化量约 {delta:.6f}（相对量级约 {rel_chg*100:.1f}% 的参考尺度）。"
        f"{shape}"
        "此为局部敏感性示意，外推至未训练区间需谨慎。"
    )
