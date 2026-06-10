"""Rebuild corrupted Python tail of ui_theme.py; preserve STREAMLIT_CUSTOM_CSS block."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "ui_theme.py"

raw = SRC.read_bytes().decode("latin-1").replace("\xb7", "·")
marker = 'STREAMLIT_CUSTOM_CSS = """'
start = raw.index(marker)
css_end = raw.index('"""', start + len(marker)) + 3
css_block = raw[start:css_end]

HEADER = '''"""
界面与 Plotly 图表统一主题：配色、字体、Streamlit 自定义 CSS。
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# 全局配色（主色 + 风险/状态色）
COLORS = {
    "primary": "#1a5f7a",
    "primary_light": "#2d8bba",
    "accent": "#e8b923",
    "success": "#2d9d78",
    "warn": "#e6a23c",
    "danger": "#c45c5c",
    "text": "#1e293b",
    "text_muted": "#64748b",
    "paper": "#ffffff",
    "plot": "#f8fafc",
    "grid": "#e2e8f0",
    "border": "#cbd5e1",
    "sidebar_bg": "#f0f4f8",
}

CHART_FONT: dict[str, Any] = {
    "family": (
        "Microsoft YaHei UI, Microsoft YaHei, PingFang SC, "
        "Hiragino Sans GB, Noto Sans SC, sans-serif"
    ),
    "size": 13,
    "color": COLORS["text"],
}

PLOTLY_CONFIG: dict[str, Any] = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "chart",
        "scale": 2,
    },
}


def apply_chart_theme(
    fig: go.Figure,
    *,
    height: int | None = None,
    title: str | None = None,
    margin: dict[str, int] | None = None,
) -> go.Figure:
    """统一 Plotly 图表字体/背景/网格样式。"""
    m = margin or {"l": 64, "r": 28, "t": 56, "b": 52}
    fig.update_layout(
        font=CHART_FONT,
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["plot"],
        hoverlabel={
            "font": {"family": CHART_FONT["family"], "size": 12},
            "bgcolor": "white",
            "bordercolor": COLORS["border"],
        },
        margin=m,
    )
    if height is not None:
        fig.update_layout(height=height)
    if title:
        fig.update_layout(
            title={
                "text": title,
                "font": {
                    "size": 15,
                    "family": CHART_FONT["family"],
                    "color": COLORS["primary"],
                },
                "x": 0.02,
                "xanchor": "left",
            }
        )
    fig.update_xaxes(
        gridcolor=COLORS["grid"],
        zerolinecolor=COLORS["border"],
        linecolor=COLORS["border"],
        showline=True,
        mirror=False,
    )
    fig.update_yaxes(
        gridcolor=COLORS["grid"],
        zerolinecolor=COLORS["border"],
        linecolor=COLORS["border"],
        showline=True,
        mirror=False,
    )
    return fig


def style_gauge_indicator(fig: go.Figure, *, height: int = 320) -> go.Figure:
    """半圆仪表统一高度与背景。"""
    fig.update_layout(
        font=CHART_FONT,
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["plot"],
        height=height,
        margin=dict(l=24, r=24, t=48, b=8),
    )
    return fig


'''

TAIL = '''
def inject_streamlit_theme() -> None:
    """向 Streamlit 注入自定义 CSS。"""
    import streamlit as st

    st.markdown(STREAMLIT_CUSTOM_CSS, unsafe_allow_html=True)


def _esc_html(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def risk_tone_from_level(level: str) -> str:
    s = str(level or "")
    if "高" in s:
        return "high"
    if "中" in s:
        return "mid"
    if "低" in s:
        return "low"
    return "neutral"


def hero_banner_html(
    title: str,
    subtitle: str,
    *,
    risk_level: str | None = None,
    model_version: str = "v2.1",
    model_status: str = "工程辅助 · Research Assisted",
) -> str:
    """页头横幅：标题 + 风险/版本徽章。"""
    tone = risk_tone_from_level(risk_level or "")
    badge_cls = f"sfc-badge sfc-badge-risk-{tone}" if tone != "neutral" else "sfc-badge sfc-badge-neutral"
    risk_txt = _esc_html(risk_level or "待评估")
    return f"""
<div class="sfc-hero">
  <motion class="sfc-hero-grid">
    <motion class="sfc-hero-left">
      <div class="sfc-hero-title">{_esc_html(title)}</motion>
      <div class="sfc-hero-en">Steel Fiber Concrete Crack &amp; Risk Assessment Platform</div>
      <p class="sfc-hero-sub">{_esc_html(subtitle)}</p>
    </div>
    <div class="sfc-hero-right">
      <motion class="sfc-badge-row">
        <span class="{badge_cls}">当前风险 · {risk_txt}</span>
      </div>
      <div class="sfc-badge-row">
        <span class="sfc-badge sfc-badge-version">模型版本 · {_esc_html(model_version)}</span>
      </div>
      <div class="sfc-badge-row">
        <span class="sfc-badge sfc-badge-neutral">{_esc_html(model_status)}</span>
      </div>
    </div>
  </div>
</motion>
"""


def kpi_card_html(
    *,
    label: str,
    value: str,
    unit: str = "",
    hint: str = "",
    tier: str = "secondary",
    tone: str = "neutral",
    icon: str = "",
) -> str:
    tier_cls = "sfc-kpi-primary" if tier == "primary" else "sfc-kpi-secondary"
    tone_cls = (
        f"sfc-kpi-tone-{tone}"
        if tone in ("low", "mid", "high", "comp", "flex")
        else "sfc-kpi-tone-neutral"
    )
    ic = f'<span class="sfc-kpi-icon">{icon}</span>' if icon else ""
    u = f'<span class="sfc-kpi-unit">{_esc_html(unit)}</span>' if unit else ""
    h = f'<motion class="sfc-kpi-hint">{_esc_html(hint)}</div>' if hint else ""
    return f"""
<div class="sfc-kpi {tier_cls} {tone_cls}">
  <div class="sfc-kpi-label">{ic}{_esc_html(label)}</div>
  <div class="sfc-kpi-value">{_esc_html(value)}{u}</div>
  {h}
</div>
"""


def engineering_conclusion_html(
    lead: str,
    factors: list[str],
    advice: str,
) -> str:
    items = "".join(f"<li>{_esc_html(x)}</li>" for x in factors)
    return f"""
<div class="sfc-conclusion">
  <div class="sfc-conclusion-title">工程分析结论</div>
  <p class="sfc-conclusion-lead">{_esc_html(lead)}</p>
  {"<ul>" + items + "</ul>" if factors else ""}
  <p class="sfc-conclusion-advice"><strong>工程建议：</strong>{_esc_html(advice)}</p>
</div>
"""


def thermal_engineering_chain_html(*, dt_ok: bool, gradient_ok: bool, restraint_ok: bool) -> str:
    """温度应力工程流程链（四步）。"""
    return thermal_full_engineering_flow_html(
        dt_ok=dt_ok,
        strain_ok=dt_ok and gradient_ok,
        stress_ok=restraint_ok and dt_ok,
        criterion_ok=dt_ok and restraint_ok,
    )


def thermal_full_engineering_flow_html(
    *,
    dt_ok: bool,
    strain_ok: bool,
    stress_ok: bool,
    criterion_ok: bool,
) -> str:
    """温度变化 → 应变 → 约束应力 → 开裂判据。"""

    def step(text: str, active: bool) -> str:
        cls = (
            "sfc-thermal-step sfc-thermal-step-active"
            if active
            else "sfc-thermal-step sfc-thermal-step-muted"
        )
        return f'<span class="{cls}">{_esc_html(text)}</span>'

    parts = [
        step("有效温差 ΔT", dt_ok),
        '<span class="sfc-thermal-arrow">→</span>',
        step("温度应变 ε_T", strain_ok),
        '<span class="sfc-thermal-arrow">→</span>',
        step("约束应力 σ_T*", stress_ok),
        '<span class="sfc-thermal-arrow">→</span>',
        step("开裂判据 η", criterion_ok),
    ]
    return (
        '<div class="sfc-thermal-chain">'
        + "".join(parts)
        + "</div>"
        + '<p style="font-size:0.82rem;color:#64748b;margin:0 0 0.5rem 0;text-align:center;">'
        "温度变化 → 温度应变 → 约束温度应力 → 开裂风险解释（工程示意）"
        "</p>"
    )


def thermal_engineering_conclusion_banner_html(
    lead: str,
    meta: str,
    *,
    band: str | None = None,
) -> str:
    band_line = ""
    if band:
        band_line = (
            f'<p class="sfc-thermal-conclusion-meta">'
            f"<strong>风险区间：</strong>{_esc_html(band)}</p>"
        )
    return f"""
<div class="sfc-thermal-conclusion">
  <p class="sfc-thermal-conclusion-lead">{_esc_html(lead)}</p>
  {band_line}
  <p class="sfc-thermal-conclusion-meta">{_esc_html(meta)}</p>
</div>
"""


def thermal_formula_step_html(
    step_no: int,
    title: str,
    formulas: list[str],
    description: str,
    kv: list[tuple[str, str]],
    *,
    note: str = "",
) -> str:
    eqs = "".join(
        f'<div class="sfc-thermal-formula-eq">{_esc_html(eq)}</motion>' for eq in formulas
    )
    kv_html = ""
    if kv:
        items = "".join(
            f"<dt>{_esc_html(k)}</dt><dd>{_esc_html(v)}</dd>" for k, v in kv
        )
        kv_html = f'<dl class="sfc-thermal-formula-kv">{items}</dl>'
    note_html = (
        f'<p class="sfc-thermal-formula-desc" style="margin-top:0.35rem;">'
        f"<em>{_esc_html(note)}</em></p>"
        if note
        else ""
    )
    return f"""
<div class="sfc-thermal-formula-block">
  <h5>Step {step_no} · {_esc_html(title)}</h5>
  {eqs}
  <p class="sfc-thermal-formula-desc">{_esc_html(description)}</p>
  {kv_html}
  {note_html}
</div>
"""


def thermal_variable_cards_html(cards: list[dict[str, str]]) -> str:
    cells = []
    for c in cards:
        cells.append(
            f'<div class="sfc-thermal-var-card">'
            f'<h5>{_esc_html(c["title"])}</h5>'
            f'<p><strong>含义：</strong>{_esc_html(c["meaning"])}</p>'
            f'<p><strong>单位：</strong>{_esc_html(c["unit"])}</p>'
            f'<p class="sfc-thermal-var-dir"><strong>影响方向：</strong>{_esc_html(c["direction"])}</p>'
            f"</div>"
        )
    return '<div class="sfc-thermal-var-grid">' + "".join(cells) + "</motion>"


def thermal_main_model_relation_html(
    crack_width: str,
    risk_p: str,
    alert: str,
    tsi_line: str,
) -> str:
    return f"""
<div class="sfc-info-note" style="margin:0.5rem 0 1rem 0;">
  <p style="margin:0 0 0.4rem 0;"><strong>与主开裂模型关系</strong></p>
  <ul style="margin:0;padding-left:1.2rem;font-size:0.88rem;line-height:1.55;">
    <li>预测缝宽 <strong>{_esc_html(crack_width)}</strong>（单位 mm）</li>
    <li>开裂概率 <strong>{_esc_html(risk_p)}</strong> · 预警 <strong>{_esc_html(alert)}</strong></li>
    <li>温度应力指数：{tsi_line}</li>
  </ul>
  <p style="margin:0.5rem 0 0 0;font-size:0.82rem;color:#64748b;">
    温度应力为解释层指标，不直接进入主模型 FEATURE_COLUMNS；工程结论请结合机理 Tab 综合研判。
  </p>
</div>
"""


def mechanical_summary_html(bullets: list[str], footer: str = "") -> str:
    items = "".join(f"<li>{_esc_html(x)}</li>" for x in bullets)
    foot = (
        f'<p class="sfc-conclusion-advice">{_esc_html(footer)}</p>'
        if footer
        else ""
    )
    return f"""
<div class="sfc-conclusion">
  <div class="sfc-conclusion-title">力学性能结论</div>
  <p class="sfc-conclusion-lead">综合研判如下：</p>
  <ul>{items}</ul>
  {foot}
</div>
"""


_CHECK_EMOJI = {"low": "✓", "mid": "!", "high": "✗", "muted": "—"}


def trust_conclusion_banner_html(
    level_label: str,
    tone: str,
    intro: str,
    positives: list[str],
    caveats: list[str],
) -> str:
    pos_ul = "".join(f"<li>{_esc_html(x)}</li>" for x in positives) if positives else ""
    cav = "".join(f"<li>{_esc_html(x)}</li>" for x in caveats) if caveats else ""
    caveats_block = (
        f'<p class="sfc-trust-banner-intro" style="margin-top:0.5rem;"><strong>注意</strong></p>'
        f"<ul>{cav}</ul>"
        if caveats
        else ""
    )
    return f"""
<div class="sfc-trust-banner sfc-trust-banner-tone-{tone}">
  <motion class="sfc-trust-banner-title">当前样本可信度 · {_esc_html(level_label)}</div>
  <p class="sfc-trust-banner-intro">{_esc_html(intro)}</p>
  {"<ul>" + pos_ul + "</ul>" if positives else ""}
  {caveats_block}
  <p class="sfc-trust-banner-caveat">以上为工程辅助评估，正式判定请以标准试验与规范为准。</p>
</div>
"""


def input_range_check_panel_html(
    title: str,
    rows: list[tuple[str, str, str]],
) -> str:
    """rows: (label, status_text, tone) tone: low|mid|high|muted"""
    lines = []
    for label, status, tone in rows:
        emoji = _CHECK_EMOJI.get(tone, "—")
        cls = f"sfc-check-v sfc-check-v-{tone}"
        lines.append(
            f'<div class="sfc-check-row">'
            f'<span class="sfc-check-k">{_esc_html(label)}</span>'
            f'<span class="{cls}">{emoji} {_esc_html(status)}</span>'
            f"</div>"
        )
    return f"""
<div class="sfc-check-panel">
  <div class="sfc-check-panel-title">{_esc_html(title)}</div>
  {"".join(lines)}
</div>
"""


def stability_status_panel_html(rows: list[tuple[str, str, str]]) -> str:
    """rows: (task_label, status_text, tone)"""
    lines = []
    for task, status, tone in rows:
        emoji = _CHECK_EMOJI.get(tone, "—")
        cls = f"sfc-check-v sfc-check-v-{tone}"
        lines.append(
            f'<div class="sfc-stability-row">'
            f'<span class="sfc-check-k">{_esc_html(task)}</span>'
            f'<span class="{cls}">{emoji} {_esc_html(status)}</span>'
            f"</motion>"
        )
    return f'<motion class="sfc-check-panel"><motion class="sfc-check-panel-title">稳定性任务状态</motion>{"".join(lines)}</div>'


def research_status_overview_html(
    title: str,
    cards: list[dict],
) -> str:
    """cards: label, value, tone in blue|yellow|red|cyan|green"""
    cells = []
    for c in cards:
        tone = c.get("tone", "blue")
        cells.append(
            f'<div class="sfc-rd-stat sfc-rd-stat-{tone}">'
            f'<p class="sfc-rd-stat-k">{_esc_html(str(c.get("label", "")))}</p>'
            f'<p class="sfc-rd-stat-v">{_esc_html(str(c.get("value", "")))}</p>'
            f"</div>"
        )
    return (
        f'<p class="sfc-section-title" style="margin-bottom:0.5rem;">{_esc_html(title)}</p>'
        f'<div class="sfc-rd-overview">{"".join(cells)}</div>'
    )


def research_module_header_html(
    title: str,
    subtitle: str,
    badge: str,
    *,
    tone: str = "info",
) -> str:
    badge_cls = f"sfc-rd-module-badge sfc-rd-module-badge-{tone}"
    return f"""
<div class="sfc-rd-module-header">
  <div class="sfc-rd-module-left">
    <p class="sfc-rd-module-title">{_esc_html(title)}</p>
    <p class="sfc-rd-module-sub">{_esc_html(subtitle)}</p>
  </div>
  <span class="{badge_cls}">{_esc_html(badge)}</span>
</div>
"""


def governance_tier_strip_html(
    tier_a: int,
    tier_b: int,
    hold_pending: int,
    tier_c: int,
) -> str:
    return f"""
<div class="sfc-rd-tier-row">
  <div class="sfc-rd-tier sfc-rd-tier-a">
    <p class="sfc-rd-tier-k">A 级</p>
    <p class="sfc-rd-tier-v">{tier_a}</p>
  </div>
  <div class="sfc-rd-tier sfc-rd-tier-b">
    <p class="sfc-rd-tier-k">B 级</p>
    <p class="sfc-rd-tier-v">{tier_b}</p>
  </div>
  <div class="sfc-rd-tier sfc-rd-tier-hold">
    <p class="sfc-rd-tier-k">待定</p>
    <p class="sfc-rd-tier-v">{hold_pending}</p>
  </div>
  <div class="sfc-rd-tier sfc-rd-tier-c">
    <p class="sfc-rd-tier-k">C 级 / 剔除</p>
    <p class="sfc-rd-tier-v">{tier_c}</p>
  </div>
</div>
"""


def standards_compact_html(bullets: list[str], footer: str) -> str:
    items = "".join(f"<li>{_esc_html(x)}</li>" for x in bullets)
    return f"""
<div class="sfc-standards-compact">
  <h4>规范依据</h4>
  <ul>{items}</ul>
  <p style="margin:0;">{_esc_html(footer)}</p>
</div>
"""


def trust_notice_card_html(title: str, bullets: list[str]) -> str:
    items = "".join(f"<li>{_esc_html(x)}</li>" for x in bullets)
    return f"""
<div class="sfc-trust-card">
  <div class="sfc-trust-card-title">{_esc_html(title)}</div>
  <ul>{items}</ul>
</div>
"""


def engineering_info_panel_html(
    title: str,
    rows: list[tuple[str, str]],
    note: str = "",
) -> str:
    cells = "".join(
        f'<div><p class="sfc-info-k">{_esc_html(k)}</p>'
        f'<p class="sfc-info-v">{_esc_html(v)}</p></div>'
        for k, v in rows
    )
    note_html = (
        f'<p class="sfc-info-note">{_esc_html(note)}</p>' if note else ""
    )
    return f"""
<div class="sfc-info-panel">
  <div class="sfc-info-panel-title">{_esc_html(title)}</div>
  <div class="sfc-info-grid">{cells}</div>
  {note_html}
</div>
"""


def mechanism_conclusion_html(
    drivers: list[str],
    synthesis: str,
    *,
    lead: str = "当前风险主要由：",
) -> str:
    items = "".join(
        f"<li>{i + 1}. {_esc_html(d)}</li>" for i, d in enumerate(drivers)
    )
    return f"""
<div class="sfc-conclusion" style="border-left-color:#e6a23c;">
  <div class="sfc-conclusion-title">当前样本开裂机理分析</div>
  <p class="sfc-conclusion-lead">{_esc_html(lead)}</p>
  <ul>{items}</ul>
  <p class="sfc-conclusion-advice"><strong>机理归纳：</strong>{_esc_html(synthesis)}</p>
</motion>
"""


def crack_formation_path_html(nodes: list[dict]) -> str:
    """nodes: icon, title, status, tone in low|mid|high|muted|info"""
    parts: list[str] = []
    for i, node in enumerate(nodes):
        tone = node.get("tone", "muted")
        parts.append(
            f'<div class="sfc-path-node sfc-path-node-tone-{tone}">'
            f'<span class="sfc-path-node-icon">{_esc_html(str(node.get("icon", "")))}</span>'
            f'<p class="sfc-path-node-title">{_esc_html(str(node.get("title", "")))}</p>'
            f'<p class="sfc-path-node-status">{_esc_html(str(node.get("status", "")))}</p>'
            f"</div>"
        )
        if i < len(nodes) - 1:
            parts.append('<span class="sfc-path-arrow">→</span>')
    return '<div class="sfc-crack-path">' + "".join(parts) + "</div>"


def thermal_status_chips_html(chips: list[tuple[str, str, str]]) -> str:
    """(label, value, tone) tone: muted|info|warn|low|mid|high"""
    cells = []
    for label, value, tone in chips:
        cls = f"sfc-thermal-chip sfc-thermal-chip-{tone}"
        cells.append(
            f'<motion class="{cls}"><p class="sfc-thermal-chip-k">{_esc_html(label)}</p>'
            f'<p class="sfc-thermal-chip-v">{_esc_html(value)}</p></div>'
        )
    return '<div class="sfc-thermal-status-row">' + "".join(cells) + "</div>"


def fiber_engineering_kpi_html(kpis: list[tuple[str, str, str]]) -> str:
    """(label, value, hint) 纤维 KPI 行。"""
    cells = []
    for label, value, hint in kpis:
        cells.append(
            f'<div class="sfc-fiber-kpi">'
            f'<p class="sfc-fiber-kpi-k">{_esc_html(label)}</p>'
            f'<p class="sfc-fiber-kpi-v">{_esc_html(value)}</p>'
            f'<p class="sfc-fiber-kpi-h">{_esc_html(hint)}</p>'
            f"</div>"
        )
    return '<div class="sfc-fiber-kpi-row">' + "".join(cells) + "</div>"


def fiber_bridge_cards_html(cards: list[dict[str, str]]) -> str:
    cells = []
    for c in cards:
        cells.append(
            f'<div class="sfc-fiber-bridge-card">'
            f'<h5>{_esc_html(c.get("title", ""))}</h5>'
            f'<p>{_esc_html(c.get("body", ""))}</p>'
            f'<p class="sfc-fiber-bridge-fx">{_esc_html(c.get("effects", ""))}</p>'
            f"</div>"
        )
    return '<div class="sfc-fiber-bridge-grid">' + "".join(cells) + "</div>"


def fiber_thermal_coordination_note_html(text: str) -> str:
    return f'<div class="sfc-fiber-thermal-note">{_esc_html(text)}</motion>'


def fiber_engineering_summary_panel_html(summary: dict[str, str]) -> str:
    rows = [
        ("体系类型", summary.get("system_type_zh", "—")),
        ("抗裂倾向", summary.get("crack_resistance_tendency_zh", "—")),
        ("温度约束能力", summary.get("thermal_constraint_capacity_zh", "—")),
        ("桥联预期", summary.get("expected_bridging_zh", "—")),
        ("分散风险", summary.get("dispersion_risk_zh", "—")),
    ]
    items = "".join(
        f"<li><strong>{_esc_html(k)}：</strong>{_esc_html(v)}</li>" for k, v in rows
    )
    return f'<div class="sfc-eng-explain-card"><h4>纤维工程体系摘要</h4><ul>{items}</ul></motion>'


def engineering_explain_card_html(title: str, bullets: list[str], footer: str = "") -> str:
    items = "".join(f"<li>{_esc_html(x)}</li>" for x in bullets)
    foot = (
        f'<p class="sfc-info-note" style="margin-top:0.5rem;border:none;padding:0;">'
        f"{_esc_html(footer)}</p>"
        if footer
        else ""
    )
    return f"""
<div class="sfc-eng-explain-card">
  <h4>{_esc_html(title)}</h4>
  <ul>{items}</ul>
  {foot}
</div>
"""


def shap_narrative_html(text: str) -> str:
    return f'<motion class="sfc-shap-narrative">{_esc_html(text)}</div>'


def mechanical_context_line_html(grade: str, fiber_pct: str) -> str:
    return (
        f'<div class="sfc-mech-context">'
        f"<strong>强度等级：</strong>{_esc_html(grade)}"
        f' &nbsp;·&nbsp; <strong>当前纤维掺量：</strong>{_esc_html(fiber_pct)}'
        f"</div>"
    )


def tab_heading_html(text: str) -> str:
    return f'<p class="sfc-tab-h1">{_esc_html(text)}</p>'


def section_title_html(text: str) -> str:
    """章节标题（允许少量 HTML，由调用方控制）。"""
    return f'<p class="sfc-section-title">{text}</p>'


def empty_state_hint_html(message: str) -> str:
    """空状态提示，使用 .sfc-empty-hint 样式。"""
    safe = (
        message.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<div class="sfc-empty-hint">{safe}</div>'


def sidebar_footer_html() -> str:
    """侧栏底部说明文字。"""
    return (
        '<p class="sfc-sidebar-foot">'
        "钢纤维混凝土抗裂 · 工程辅助评估 · XGBoost 集成 · 仅供科研与方案比选"
        "</p>"
    )


def mechanism_flow_html() -> str:
    """机理页顶部流程示意。"""
    return """
<div class="sfc-flow-wrap">
  <span class="sfc-flow-box">输入参数与材料特征</span>
  <span class="sfc-flow-arrow">→</span>
  <span class="sfc-flow-box">公式基线 + XGBoost</span>
  <span class="sfc-flow-arrow">→</span>
  <span class="sfc-flow-box">开裂概率</span>
  <span class="sfc-flow-arrow">→</span>
  <span class="sfc-flow-box">缝宽预测</span>
  <span class="sfc-flow-arrow">→</span>
  <span class="sfc-flow-box">机理解释</span>
</div>
"""
'''

# Fix accidental motion tags from copy-paste
TAIL = TAIL.replace('<motion ', '<div ').replace('</motion>', '</div>')

out = HEADER + css_block + "\n\n" + TAIL
SRC.write_text(out, encoding="utf-8")
compile(out, str(SRC), "exec")
print("rebuilt OK", SRC, "bytes", len(out.encode("utf-8")))
