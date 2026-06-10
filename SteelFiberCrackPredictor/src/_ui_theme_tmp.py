"""
??? Plotly ?????????????Streamlit ??? CSS?
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# ?? ????????????????? + ???????
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
    """?? Plotly ??/??/??????????????"""
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
    """??????????????"""
    fig.update_layout(
        font=CHART_FONT,
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["plot"],
        height=height,
        margin=dict(l=24, r=24, t=48, b=8),
    )
    return fig


STREAMLIT_CUSTOM_CSS = """
<style>
    /* ??????*/
    .block-container {
        padding-top: 1.1rem !important;
        padding-bottom: 2.75rem !important;
        max-width: 1320px !important;
    }
    .main {
        background: radial-gradient(ellipse 120% 80% at 50% -20%, rgba(26, 95, 122, 0.06), transparent 50%),
                    linear-gradient(180deg, #f4f7fb 0%, #eef2f7 100%);
    }
    h1 {
        font-family: "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
        color: #0f172a !important;
        border-bottom: none !important;
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }
    h2, h3 {
        font-family: "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", sans-serif !important;
        color: #1e293b !important;
    }
    /* ??????????*/
    .sfc-hero {
        margin: 0 0 0.85rem 0;
        padding: 1rem 1.25rem 1rem 1.25rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #ffffff 0%, #f0f7fb 48%, #e8f2f8 100%);
        border: 1px solid #c5dce8;
        box-shadow: 0 6px 24px rgba(15, 23, 42, 0.08);
    }
    .sfc-hero-grid {
        display: flex;
        flex-wrap: wrap;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem 1.5rem;
    }
    .sfc-hero-left { flex: 1 1 420px; min-width: 260px; }
    .sfc-hero-right {
        flex: 0 0 auto;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.45rem;
        min-width: 200px;
    }
    .sfc-hero-title {
        font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
        font-size: clamp(1.75rem, 2.4vw, 2.35rem);
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 0.2rem 0;
        letter-spacing: 0.02em;
        line-height: 1.2;
    }
    .sfc-hero-en {
        font-size: 0.8rem;
        color: #64748b;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0 0 0.45rem 0;
        font-weight: 600;
    }
    .sfc-hero-sub {
        font-size: 0.95rem;
        color: #475569;
        margin: 0;
        line-height: 1.55;
        max-width: 52rem;
    }
    .sfc-badge-row { display: flex; flex-wrap: wrap; gap: 0.4rem; justify-content: flex-end; }
    .sfc-badge {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid transparent;
        white-space: nowrap;
    }
    .sfc-badge-risk-low { background: #e6f6f1; color: #0d6b52; border-color: #9fd9c8; }
    .sfc-badge-risk-mid { background: #fff4e5; color: #9a5b0a; border-color: #f0d4a8; }
    .sfc-badge-risk-high { background: #fdecec; color: #9b2c2c; border-color: #f0b4b4; }
    .sfc-badge-neutral { background: #eef2f7; color: #334155; border-color: #cbd5e1; }
    .sfc-badge-version { background: #e8f2f8; color: #1a5f7a; border-color: #94c4d8; }
    /* Tab ???? */
    .sfc-tab-h1 {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        margin: 0 0 0.35rem 0 !important;
        line-height: 1.3 !important;
    }
    /* ?????? */
    .sfc-conclusion {
        margin: 0 0 1rem 0;
        padding: 1rem 1.15rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #f8fbfe 0%, #f0f7fb 100%);
        border: 1px solid #c5dce8;
        border-left: 5px solid #1a5f7a;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.05);
    }
    .sfc-conclusion-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1a5f7a;
        margin: 0 0 0.5rem 0;
    }
    .sfc-conclusion-lead {
        font-size: 0.98rem;
        color: #334155;
        margin: 0 0 0.55rem 0;
        line-height: 1.6;
    }
    .sfc-conclusion ul {
        margin: 0.35rem 0 0.5rem 1.1rem;
        padding: 0;
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .sfc-conclusion-advice {
        font-size: 0.92rem;
        color: #1e293b;
        margin: 0.5rem 0 0 0;
        padding-top: 0.5rem;
        border-top: 1px dashed #cbd5e1;
        line-height: 1.55;
    }
    /* KPI ?? */
    .sfc-kpi-grid-primary {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0 0 0.75rem 0;
    }
    .sfc-kpi-grid-secondary {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0 0 1rem 0;
    }
    @media (max-width: 900px) {
        .sfc-kpi-grid-primary, .sfc-kpi-grid-secondary { grid-template-columns: 1fr; }
    }
    .sfc-kpi {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.85rem 1rem 0.9rem 1rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
        border-left: 5px solid #94a3b8;
    }
    .sfc-kpi-primary {
        padding: 1.05rem 1.1rem 1.1rem 1.1rem;
        min-height: 7.5rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
    }
    .sfc-kpi-secondary {
        padding: 0.7rem 0.85rem;
        min-height: 5.5rem;
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    }
    .sfc-kpi-tone-low { border-left-color: #2d9d78 !important; background: linear-gradient(135deg, #f6fcfa 0%, #ffffff 70%); }
    .sfc-kpi-tone-mid { border-left-color: #e6a23c !important; background: linear-gradient(135deg, #fffaf3 0%, #ffffff 70%); }
    .sfc-kpi-tone-high { border-left-color: #c45c5c !important; background: linear-gradient(135deg, #fef7f7 0%, #ffffff 70%); }
    .sfc-kpi-tone-neutral { border-left-color: #64748b !important; }
    .sfc-kpi-tone-comp {
        border-left-color: #2563eb !important;
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 72%) !important;
    }
    .sfc-kpi-tone-flex {
        border-left-color: #0891b2 !important;
        background: linear-gradient(135deg, #ecfeff 0%, #ffffff 72%) !important;
    }
    .sfc-mech-context {
        margin: 0 0 1rem 0;
        padding: 0.65rem 1rem;
        border-radius: 10px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        font-size: 0.92rem;
        color: #475569;
        line-height: 1.55;
    }
    .sfc-mech-context strong { color: #1e293b; }
    .sfc-trust-card {
        margin: 0 0 1rem 0;
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #f1f5f9 0%, #e8f0f7 100%);
        border: 1px solid #cbd5e1;
        border-left: 5px solid #64748b;
    }
    .sfc-trust-card-title {
        font-size: 1rem;
        font-weight: 700;
        color: #334155;
        margin: 0 0 0.5rem 0;
    }
    .sfc-trust-card ul {
        margin: 0;
        padding-left: 1.1rem;
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .sfc-info-panel {
        margin: 0 0 1rem 0;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .sfc-info-panel-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1a5f7a;
        margin: 0 0 0.65rem 0;
    }
    .sfc-info-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.5rem 1.25rem;
    }
    @media (max-width: 720px) {
        .sfc-info-grid { grid-template-columns: 1fr; }
    }
    .sfc-info-k {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        margin: 0 0 0.15rem 0;
    }
    .sfc-info-v {
        font-size: 0.95rem;
        color: #0f172a;
        font-weight: 600;
        margin: 0;
    }
    .sfc-info-note {
        font-size: 0.82rem;
        color: #64748b;
        margin: 0.65rem 0 0 0;
        line-height: 1.5;
        border-top: 1px dashed #e2e8f0;
        padding-top: 0.55rem;
    }
    .sfc-mech-kpi-row {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0 0 0.75rem 0;
    }
    @media (max-width: 720px) {
        .sfc-mech-kpi-row { grid-template-columns: 1fr; }
    }
    /* ???????*/
    .sfc-crack-path {
        display: flex;
        flex-wrap: wrap;
        align-items: stretch;
        justify-content: center;
        gap: 0.35rem 0.4rem;
        padding: 1rem 0.85rem;
        margin: 0 0 1rem 0;
        border-radius: 14px;
        background: linear-gradient(90deg, #f8fafc 0%, #eef4f8 45%, #f8fafc 100%);
        border: 1px solid #cbd5e1;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.05);
    }
    .sfc-path-node {
        flex: 1 1 108px;
        max-width: 140px;
        min-width: 96px;
        padding: 0.55rem 0.5rem;
        border-radius: 10px;
        background: #fff;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 1px 6px rgba(15, 23, 42, 0.04);
    }
    .sfc-path-node-icon { font-size: 1.05rem; display: block; margin-bottom: 0.2rem; }
    .sfc-path-node-title {
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 600;
        margin: 0 0 0.25rem 0;
        line-height: 1.25;
    }
    .sfc-path-node-status {
        font-size: 0.8rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
        line-height: 1.3;
    }
    .sfc-path-node-tone-low { border-left: 3px solid #2d9d78; background: linear-gradient(135deg, #f6fcfa, #fff); }
    .sfc-path-node-tone-mid { border-left: 3px solid #e6a23c; background: linear-gradient(135deg, #fffaf3, #fff); }
    .sfc-path-node-tone-high { border-left: 3px solid #c45c5c; background: linear-gradient(135deg, #fef7f7, #fff); }
    .sfc-path-node-tone-muted { border-left: 3px solid #94a3b8; background: #f8fafc; color: #64748b; }
    .sfc-path-node-tone-info { border-left: 3px solid #2563eb; background: linear-gradient(135deg, #eff6ff, #fff); }
    .sfc-path-arrow {
        align-self: center;
        color: #e6a23c;
        font-weight: 700;
        font-size: 1rem;
        padding: 0 0.1rem;
    }
    .sfc-thermal-status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0 0 0.75rem 0;
    }
    .sfc-thermal-chip {
        flex: 1 1 140px;
        padding: 0.55rem 0.7rem;
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        background: #fff;
        font-size: 0.82rem;
    }
    .sfc-thermal-chip-k { color: #64748b; font-weight: 600; margin: 0 0 0.2rem 0; }
    .sfc-thermal-chip-v { font-weight: 700; margin: 0; }
    .sfc-thermal-chip-muted .sfc-thermal-chip-v { color: #94a3b8; }
    .sfc-thermal-chip-info .sfc-thermal-chip-v { color: #2563eb; }
    .sfc-thermal-chip-warn .sfc-thermal-chip-v { color: #b45309; }
    .sfc-eng-explain-card {
        margin: 0 0 0.75rem 0;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #e6a23c;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .sfc-eng-explain-card h4 {
        font-size: 0.95rem;
        color: #1a5f7a;
        margin: 0 0 0.45rem 0;
        font-weight: 700;
    }
    .sfc-eng-explain-card ul {
        margin: 0.35rem 0 0 1rem;
        padding: 0;
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.5;
    }
    .sfc-shap-narrative {
        font-size: 0.92rem;
        color: #334155;
        margin: 0 0 0.75rem 0;
        padding: 0.65rem 0.85rem;
        background: #f8fafc;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        line-height: 1.55;
    }
    /* ??????*/
    .sfc-trust-banner {
        margin: 0 0 1rem 0;
        padding: 1.1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    }
    .sfc-trust-banner-tone-low {
        background: linear-gradient(135deg, #f6fcfa 0%, #ffffff 72%);
        border-left: 5px solid #2d9d78;
    }
    .sfc-trust-banner-tone-mid {
        background: linear-gradient(135deg, #fffaf3 0%, #ffffff 72%);
        border-left: 5px solid #e6a23c;
    }
    .sfc-trust-banner-tone-high {
        background: linear-gradient(135deg, #fef7f7 0%, #ffffff 72%);
        border-left: 5px solid #c45c5c;
    }
    .sfc-trust-banner-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 0.45rem 0;
    }
    .sfc-trust-banner-intro {
        font-size: 0.95rem;
        color: #334155;
        margin: 0 0 0.5rem 0;
        line-height: 1.55;
    }
    .sfc-trust-banner ul {
        margin: 0.35rem 0 0 1.1rem;
        padding: 0;
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.5;
    }
    .sfc-trust-banner-caveat {
        margin: 0.55rem 0 0 0;
        padding-top: 0.5rem;
        border-top: 1px dashed #cbd5e1;
        font-size: 0.88rem;
        color: #64748b;
        line-height: 1.5;
    }
    .sfc-check-panel {
        margin: 0 0 1rem 0;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: #fff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .sfc-check-panel-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1a5f7a;
        margin: 0 0 0.65rem 0;
    }
    .sfc-check-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.9rem;
    }
    .sfc-check-row:last-child { border-bottom: none; }
    .sfc-check-k { color: #475569; font-weight: 600; }
    .sfc-check-v { font-weight: 700; }
    .sfc-check-v-low { color: #0d6b52; }
    .sfc-check-v-mid { color: #b45309; }
    .sfc-check-v-high { color: #9b2c2c; }
    .sfc-check-v-muted { color: #94a3b8; }
    .sfc-stability-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0.65rem;
        margin: 0 0 0.35rem 0;
        border-radius: 8px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        font-size: 0.9rem;
    }
    .sfc-stability-row:last-child { margin-bottom: 0; }
    .sfc-standards-compact {
        margin: 0 0 1rem 0;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #cbd5e1;
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.55;
    }
    .sfc-standards-compact h4 {
        margin: 0 0 0.4rem 0;
        font-size: 0.95rem;
        color: #334155;
    }
    .sfc-standards-compact ul { margin: 0 0 0.5rem 1.1rem; padding: 0; }
    /* ??????*/
    .sfc-rd-overview {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0 0 1.1rem 0;
    }
    @media (max-width: 900px) {
        .sfc-rd-overview { grid-template-columns: repeat(2, 1fr); }
    }
    .sfc-rd-stat {
        padding: 0.7rem 0.75rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background: #fff;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .sfc-rd-stat-k {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 600;
        margin: 0 0 0.25rem 0;
    }
    .sfc-rd-stat-v {
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0;
    }
    .sfc-rd-stat-blue { border-left: 4px solid #2563eb; background: linear-gradient(135deg, #eff6ff, #fff); }
    .sfc-rd-stat-blue .sfc-rd-stat-v { color: #1d4ed8; }
    .sfc-rd-stat-yellow { border-left: 4px solid #e6a23c; background: linear-gradient(135deg, #fffaf3, #fff); }
    .sfc-rd-stat-yellow .sfc-rd-stat-v { color: #b45309; }
    .sfc-rd-stat-red { border-left: 4px solid #c45c5c; background: linear-gradient(135deg, #fef7f7, #fff); }
    .sfc-rd-stat-red .sfc-rd-stat-v { color: #9b2c2c; }
    .sfc-rd-stat-cyan { border-left: 4px solid #0891b2; background: linear-gradient(135deg, #ecfeff, #fff); }
    .sfc-rd-stat-cyan .sfc-rd-stat-v { color: #0e7490; }
    .sfc-rd-stat-green { border-left: 4px solid #2d9d78; background: linear-gradient(135deg, #f6fcfa, #fff); }
    .sfc-rd-stat-green .sfc-rd-stat-v { color: #0d6b52; }
    .sfc-rd-module {
        margin: 0 0 0.85rem 0;
        padding: 0.85rem 1rem 0.35rem 1rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border: 1px solid #e2e8f0;
    }
    .sfc-rd-module-head {
        display: flex;
        flex-wrap: wrap;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .sfc-rd-module-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .sfc-rd-module-sub {
        font-size: 0.85rem;
        color: #64748b;
        margin: 0.2rem 0 0 0;
        line-height: 1.45;
    }
    .sfc-rd-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        background: #e8f2f8;
        color: #1a5f7a;
        border: 1px solid #94c4d8;
    }
    .sfc-rd-badge-exp {
        background: #f1f5f9;
        color: #64748b;
        border-color: #cbd5e1;
    }
    .sfc-rd-tier-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0 0 0.65rem 0;
    }
    .sfc-rd-tier {
        flex: 1 1 100px;
        padding: 0.55rem 0.65rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        text-align: center;
        background: #fff;
    }
    .sfc-rd-tier-k { font-size: 0.78rem; color: #64748b; margin: 0 0 0.15rem 0; }
    .sfc-rd-tier-v { font-size: 1.1rem; font-weight: 700; margin: 0; }
    .sfc-rd-tier-a { border-left: 4px solid #2d9d78; }
    .sfc-rd-tier-a .sfc-rd-tier-v { color: #0d6b52; }
    .sfc-rd-tier-b { border-left: 4px solid #2563eb; }
    .sfc-rd-tier-b .sfc-rd-tier-v { color: #1d4ed8; }
    .sfc-rd-tier-hold { border-left: 4px solid #e6a23c; }
    .sfc-rd-tier-hold .sfc-rd-tier-v { color: #b45309; }
    .sfc-rd-tier-c { border-left: 4px solid #c45c5c; }
    .sfc-rd-tier-c .sfc-rd-tier-v { color: #9b2c2c; }
    .sfc-rd-note {
        font-size: 0.86rem;
        color: #64748b;
        margin: 0 0 0.65rem 0;
        line-height: 1.5;
    }
    .sfc-kpi-icon { font-size: 1.1rem; margin-right: 0.25rem; vertical-align: middle; }
    .sfc-kpi-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 600;
        margin: 0 0 0.35rem 0;
        letter-spacing: 0.02em;
    }
    .sfc-kpi-value {
        font-size: clamp(1.75rem, 3vw, 2.65rem);
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .sfc-kpi-primary .sfc-kpi-value { font-size: clamp(2rem, 3.5vw, 2.85rem); }
    .sfc-kpi-secondary .sfc-kpi-value { font-size: clamp(1.35rem, 2.2vw, 1.85rem); color: #1a5f7a; }
    .sfc-kpi-unit { font-size: 0.95rem; font-weight: 600; color: #64748b; margin-left: 0.15rem; }
    .sfc-kpi-hint { font-size: 0.78rem; color: #94a3b8; margin: 0.35rem 0 0 0; line-height: 1.4; }
    /* ??????*/
    .sfc-thermal-chain {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: center;
        gap: 0.35rem 0.5rem;
        padding: 0.85rem 1rem;
        margin: 0.65rem 0 1rem 0;
        background: linear-gradient(90deg, #f0f9ff 0%, #f8fafc 50%, #fff7ed 100%);
        border: 1px solid #bae0ef;
        border-radius: 12px;
        font-size: 0.88rem;
    }
    .sfc-thermal-step {
        background: #fff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.4rem 0.65rem;
        font-weight: 600;
        color: #334155;
    }
    .sfc-thermal-step-active {
        border-color: #1a5f7a;
        background: #e8f4fa;
        color: #1a5f7a;
    }
    .sfc-thermal-step-muted { opacity: 0.55; }
    .sfc-thermal-arrow { color: #2d8bba; font-weight: 700; font-size: 1rem; }
    .sfc-thermal-conclusion {
        background: linear-gradient(135deg, #f0f9ff 0%, #f8fafc 100%);
        border: 1px solid #7dd3fc;
        border-left: 4px solid #1a5f7a;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin: 0 0 1rem 0;
    }
    .sfc-thermal-conclusion-lead {
        font-size: 1.02rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0 0 0.45rem 0;
        line-height: 1.55;
    }
    .sfc-thermal-conclusion-meta {
        font-size: 0.82rem;
        color: #64748b;
        margin: 0;
    }
    .sfc-thermal-formula-block {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin: 0 0 0.65rem 0;
        background: #fff;
    }
    .sfc-thermal-formula-block h5 {
        margin: 0 0 0.35rem 0;
        font-size: 0.92rem;
        color: #1a5f7a;
    }
    .sfc-thermal-formula-eq {
        font-family: ui-monospace, Consolas, monospace;
        font-size: 0.95rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0.25rem 0;
        padding: 0.35rem 0.5rem;
        background: #f1f5f9;
        border-radius: 6px;
        display: inline-block;
    }
    .sfc-thermal-formula-desc {
        font-size: 0.84rem;
        color: #475569;
        margin: 0.35rem 0 0.5rem 0;
        line-height: 1.5;
    }
    .sfc-thermal-formula-kv {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
        gap: 0.35rem 0.75rem;
        font-size: 0.82rem;
        margin: 0;
    }
    .sfc-thermal-formula-kv dt { color: #64748b; font-weight: 600; margin: 0; }
    .sfc-thermal-formula-kv dd { color: #0f172a; font-weight: 700; margin: 0; }
    .sfc-thermal-var-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 0.65rem;
        margin: 0.5rem 0 1rem 0;
    }
    .sfc-thermal-var-card {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 0.85rem;
        background: #fafbfc;
    }
    .sfc-thermal-var-card h5 {
        margin: 0 0 0.35rem 0;
        font-size: 0.9rem;
        color: #1e3a5f;
    }
    .sfc-thermal-var-card p {
        margin: 0.2rem 0;
        font-size: 0.8rem;
        color: #475569;
        line-height: 1.45;
    }
    .sfc-thermal-var-dir {
        color: #1a5f7a;
        font-weight: 600;
    }
    .sfc-thermal-footnote {
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.55;
        padding: 0.75rem 1rem;
        background: #f8fafc;
        border-radius: 8px;
        border: 1px dashed #cbd5e1;
        margin: 0.75rem 0 0 0;
    }
    /* ???? KPI????????*/
    .sfc-fiber-kpi-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.65rem;
        margin: 0.65rem 0 1rem 0;
    }
    .sfc-fiber-kpi {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-top: 3px solid #1a5f7a;
        border-radius: 8px;
        padding: 0.65rem 0.75rem;
    }
    .sfc-fiber-kpi-k {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 0 0 0.25rem 0;
    }
    .sfc-fiber-kpi-v {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        line-height: 1.25;
    }
    .sfc-fiber-kpi-h {
        font-size: 0.72rem;
        color: #94a3b8;
        margin: 0.3rem 0 0 0;
        line-height: 1.35;
    }
    .sfc-fiber-bridge-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.65rem;
        margin: 0.5rem 0 1rem 0;
    }
    .sfc-fiber-bridge-card {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #2d8bba;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
    }
    .sfc-fiber-bridge-card h5 {
        font-size: 0.88rem;
        font-weight: 700;
        color: #1a5f7a;
        margin: 0 0 0.4rem 0;
    }
    .sfc-fiber-bridge-card p {
        font-size: 0.8rem;
        color: #475569;
        line-height: 1.5;
        margin: 0 0 0.35rem 0;
    }
    .sfc-fiber-bridge-fx {
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 600;
    }
    .sfc-fiber-thermal-note {
        font-size: 0.84rem;
        color: #334155;
        line-height: 1.55;
        padding: 0.7rem 0.9rem;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-left: 3px solid #1a5f7a;
        border-radius: 8px;
        margin: 0.5rem 0 1rem 0;
    }
    .sfc-zone-tight { margin-bottom: 0.65rem !important; }
    .sfc-zone-loose { margin-bottom: 1.35rem !important; }
    /* ????*/
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eef3f8 0%, #e4edf5 100%);
        border-right: 1px solid #cbd5e1;
        min-width: 17.5rem !important;
        max-width: 18.5rem !important;
        width: 17.5rem !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 17.5rem !important;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #1a5f7a !important;
        font-size: 1.1rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.72);
        border-radius: 12px !important;
        border: 1px solid #c5d5e3 !important;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #1a5f7a !important;
    }
    /* Tab ?? */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #e8eef5;
        padding: 8px 10px 0 10px;
        border-radius: 12px 12px 0 0;
        border: 1px solid #cbd5e1;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
        font-weight: 500;
        font-size: 0.92rem !important;
        padding: 0.45rem 0.75rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #1a5f7a 0%, #164e63 100%) !important;
        color: #ffffff !important;
    }
    /* ?????????? */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.7rem 0.85rem !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a5f7a !important;
        font-weight: 600 !important;
    }
    /* ????????*/
    .sfc-section-title {
        font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a5f7a;
        margin: 1.1rem 0 0.65rem 0;
        padding: 0.35rem 0 0.35rem 0.65rem;
        border-left: 4px solid #2d8bba;
        background: linear-gradient(90deg, rgba(45,139,186,0.08) 0%, transparent 100%);
        border-radius: 0 6px 6px 0;
    }
    /* ????????*/
    .sfc-flow-wrap {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: center;
        gap: 0.35rem 0.5rem;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0 1.25rem 0;
        background: linear-gradient(135deg, #f0f7fb 0%, #e8f4f8 100%);
        border: 1px solid #bae0ef;
        border-radius: 12px;
        font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
        font-size: 0.92rem;
        color: #334155;
    }
    .sfc-flow-box {
        background: #fff;
        border: 1px solid #94c4d8;
        border-radius: 8px;
        padding: 0.45rem 0.75rem;
        font-weight: 500;
        color: #1a5f7a;
    }
    .sfc-flow-arrow {
        color: #2d8bba;
        font-weight: bold;
    }
    /* ????caption */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }
    /* ??????*/
    [data-testid="stAlert"] {
        border-radius: 10px !important;
    }
    /* ????*/
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
    /* ??????*/
    .stButton button[kind="primary"] {
        background: linear-gradient(180deg, #1a5f7a 0%, #155a72 100%) !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(26, 95, 122, 0.25) !important;
    }
    .stButton button[kind="primary"]:hover {
        filter: brightness(1.05);
    }
    /* ??border ?????Streamlit 1.29+??*/
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border-color: #c5dce8 !important;
        background: linear-gradient(165deg, rgba(255,255,255,0.97) 0%, rgba(248,250,252,0.95) 100%) !important;
        box-shadow: 0 2px 14px rgba(15, 23, 42, 0.06) !important;
        padding: 0.35rem 0.5rem 0.65rem 0.5rem !important;
    }
    /* Tab ??????????*/
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.85rem !important;
        padding-left: 0.15rem !important;
        padding-right: 0.15rem !important;
    }
    /* ??? Tab ?? */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(26, 95, 122, 0.08) !important;
    }
    /* ????*/
    hr {
        border: none !important;
        border-top: 1px solid #d1dbe8 !important;
        margin: 1rem 0 !important;
    }
    /* ?????? */
    [data-testid="stSidebar"] label {
        color: #334155 !important;
        font-weight: 500 !important;
    }
    /* ??????*/
    .main a {
        color: #1a5f7a !important;
        font-weight: 500 !important;
    }
    /* ?????? */
    .sfc-empty-hint {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #f0f7fb 0%, #e8f0f8 100%);
        border: 1px solid #bae0ef;
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    /* ???? */
    .sfc-sidebar-foot {
        font-size: 0.78rem !important;
        color: #64748b !important;
        margin-top: 1rem !important;
        padding-top: 0.75rem !important;
        border-top: 1px dashed #cbd5e1 !important;
    }
</style>
"""


def inject_streamlit_theme() -> None:
    """???????????????????"""
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
    if "?" in s:
        return "high"
    if "?" in s:
        return "mid"
    if "?" in s:
        return "low"
    return "neutral"


def hero_banner_html(
    title: str,
    subtitle: str,
    *,
    risk_level: str | None = None,
    model_version: str = "v2.1",
    model_status: str = "???? ? Research Assisted",
) -> str:
    """??????? + ??/?????"""
    tone = risk_tone_from_level(risk_level or "")
    badge_cls = f"sfc-badge sfc-badge-risk-{tone}" if tone != "neutral" else "sfc-badge sfc-badge-neutral"
    risk_txt = _esc_html(risk_level or "???")
    return f"""
<div class="sfc-hero">
  <div class="sfc-hero-grid">
    <div class="sfc-hero-left">
      <div class="sfc-hero-title">{_esc_html(title)}</div>
      <div class="sfc-hero-en">Steel Fiber Concrete Crack &amp; Risk Assessment Platform</div>
      <p class="sfc-hero-sub">{_esc_html(subtitle)}</p>
    </div>
    <div class="sfc-hero-right">
      <div class="sfc-badge-row">
        <span class="{badge_cls}">?? ?? ? {risk_txt}</span>
      </div>
      <div class="sfc-badge-row">
        <span class="sfc-badge sfc-badge-version">?? ?? · {_esc_html(model_version)}</span>
      </div>
      <div class="sfc-badge-row">
        <span class="sfc-badge sfc-badge-neutral">{_esc_html(model_status)}</span>
      </div>
    </div>
  </div>
</div>
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
    h = f'<div class="sfc-kpi-hint">{_esc_html(hint)}</div>' if hint else ""
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
  <div class="sfc-conclusion-title">?? ??????</div>
  <p class="sfc-conclusion-lead">{_esc_html(lead)}</p>
  {"<ul>" + items + "</ul>" if factors else ""}
  <p class="sfc-conclusion-advice"><strong>?????</strong>{_esc_html(advice)}</p>
</div>
"""


def thermal_engineering_chain_html(*, dt_ok: bool, gradient_ok: bool, restraint_ok: bool) -> str:
    """?????????????????????"""
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
    """???? ???? ?? ????????????"""

    def step(text: str, active: bool) -> str:
        cls = (
            "sfc-thermal-step sfc-thermal-step-active"
            if active
            else "sfc-thermal-step sfc-thermal-step-muted"
        )
        return f'<span class="{cls}">{_esc_html(text)}</span>'

    parts = [
        step("???? ?T", dt_ok),
        '<span class="sfc-thermal-arrow">??/span>',
        step("???? ?_T", strain_ok),
        '<span class="sfc-thermal-arrow">??/span>',
        step("???? ?_T*", stress_ok),
        '<span class="sfc-thermal-arrow">??/span>',
        step("???? ?", criterion_ok),
    ]
    return (
        '<div class="sfc-thermal-chain">'
        + "".join(parts)
        + "</div>"
        + '<p style="font-size:0.82rem;color:#64748b;margin:0 0 0.5rem 0;text-align:center;">'
        "???????? ???? ???? ????????????????
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
            f"<strong>??????????/strong>{_esc_html(band)}??</p>"
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
        f'<div class="sfc-thermal-formula-eq">{_esc_html(eq)}</div>' for eq in formulas
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
  <h5>Step {step_no}?{_esc_html(title)}</h5>
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
            f'<p><strong>?????</strong>{_esc_html(c["meaning"])}</p>'
            f'<p><strong>????/strong>{_esc_html(c["unit"])}</p>'
            f'<p class="sfc-thermal-var-dir"><strong>????</strong>{_esc_html(c["direction"])}</p>'
            f"</div>"
        )
    return '<div class="sfc-thermal-var-grid">' + "".join(cells) + "</div>"


def thermal_main_model_relation_html(
    crack_width: str,
    risk_p: str,
    alert: str,
    tsi_line: str,
) -> str:
    return f"""
<div class="sfc-info-note" style="margin:0.5rem 0 1rem 0;">
  <p style="margin:0 0 0.4rem 0;"><strong>???????????</strong></p>
  <ul style="margin:0;padding-left:1.2rem;font-size:0.88rem;line-height:1.55;">
    <li>????????<strong>{_esc_html(crack_width)}</strong>?????????? mm??/li>
    <li>????????<strong>{_esc_html(risk_p)}</strong> ? ????<strong>{_esc_html(alert)}</strong></li>
    <li>???????{tsi_line}</li>
  </ul>
  <p style="margin:0.5rem 0 0 0;font-size:0.82rem;color:#64748b;">
    ????????????????????????????????????????  </p>
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
  <div class="sfc-conclusion-title">?? ????????</div>
  <p class="sfc-conclusion-lead">???????</p>
  <ul>{items}</ul>
  {foot}
</div>
"""


_CHECK_EMOJI = {"low": "??", "mid": "??", "high": "??", "muted": "??}


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
        f'<p class="sfc-trust-banner-intro" style="margin-top:0.5rem;"><strong>??</strong></p>'
        f"<ul>{cav}</ul>"
        if caveats
        else ""
    )
    return f"""
<div class="sfc-trust-banner sfc-trust-banner-tone-{tone}">
  <div class="sfc-trust-banner-title">????????{_esc_html(level_label)}</div>
  <p class="sfc-trust-banner-intro">{_esc_html(intro)}</p>
  {"<ul>" + pos_ul + "</ul>" if positives else ""}
  {caveats_block}
  <p class="sfc-trust-banner-caveat">?????????????????????????????????</p>
</div>
"""


def input_range_check_panel_html(
    title: str,
    rows: list[tuple[str, str, str]],
) -> str:
    """rows: (label, status_text, tone) tone: low|mid|high|muted"""
    lines = []
    for label, status, tone in rows:
        emoji = _CHECK_EMOJI.get(tone, "??)
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
        emoji = _CHECK_EMOJI.get(tone, "??)
        cls = f"sfc-check-v sfc-check-v-{tone}"
        lines.append(
            f'<div class="sfc-stability-row">'
            f'<span class="sfc-check-k">{_esc_html(task)}</span>'
            f'<span class="{cls}">{emoji} {_esc_html(status)}</span>'
            f"</div>"
        )
    return f'<div class="sfc-check-panel"><div class="sfc-check-panel-title">??????????????/div>{"".join(lines)}</div>'


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
    *,
    badge: str = "Research",
    badge_class: str = "",
) -> str:
    badge_cls = f"sfc-rd-badge {badge_class}".strip()
    return f"""
<div class="sfc-rd-module">
  <div class="sfc-rd-module-head">
    <div>
      <p class="sfc-rd-module-title">{_esc_html(title)}</p>
      <p class="sfc-rd-module-sub">{_esc_html(subtitle)}</p>
    </div>
    <span class="{badge_cls}">{_esc_html(badge)}</span>
  </div>
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
    <p class="sfc-rd-tier-k">A ??/p>
    <p class="sfc-rd-tier-v">{tier_a}</p>
  </div>
  <div class="sfc-rd-tier sfc-rd-tier-b">
    <p class="sfc-rd-tier-k">B ??/p>
    <p class="sfc-rd-tier-v">{tier_b}</p>
  </div>
  <div class="sfc-rd-tier sfc-rd-tier-hold">
    <p class="sfc-rd-tier-k">??</p>
    <p class="sfc-rd-tier-v">{hold_pending}</p>
  </div>
  <div class="sfc-rd-tier sfc-rd-tier-c">
    <p class="sfc-rd-tier-k">C ??/ ??</p>
    <p class="sfc-rd-tier-v">{tier_c}</p>
  </div>
</div>
"""


def standards_compact_html(bullets: list[str], footer: str) -> str:
    items = "".join(f"<li>{_esc_html(x)}</li>" for x in bullets)
    return f"""
<div class="sfc-standards-compact">
  <h4>????</h4>
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
    lead: str = "????????",
) -> str:
    items = "".join(
        f"<li>{i + 1}. {_esc_html(d)}</li>" for i, d in enumerate(drivers)
    )
    return f"""
<div class="sfc-conclusion" style="border-left-color:#e6a23c;">
  <div class="sfc-conclusion-title">???????????/div>
  <p class="sfc-conclusion-lead">{_esc_html(lead)}</p>
  <ul>{items}</ul>
  <p class="sfc-conclusion-advice"><strong>?????</strong>{_esc_html(synthesis)}</p>
</div>
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
            parts.append('<span class="sfc-path-arrow">??/span>')
    return '<div class="sfc-crack-path">' + "".join(parts) + "</div>"


def thermal_status_chips_html(chips: list[tuple[str, str, str]]) -> str:
    """(label, value, tone) tone: muted|info|warn|low|mid|high"""
    cells = []
    for label, value, tone in chips:
        cls = f"sfc-thermal-chip sfc-thermal-chip-{tone}"
        cells.append(
            f'<div class="{cls}"><p class="sfc-thermal-chip-k">{_esc_html(label)}</p>'
            f'<p class="sfc-thermal-chip-v">{_esc_html(value)}</p></div>'
        )
    return '<div class="sfc-thermal-status-row">' + "".join(cells) + "</div>"


def fiber_engineering_kpi_html(kpis: list[tuple[str, str, str]]) -> str:
    """(label, value, hint) ?? KPI ???"""
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
    return f'<div class="sfc-fiber-thermal-note">{_esc_html(text)}</div>'


def fiber_engineering_summary_panel_html(summary: dict[str, str]) -> str:
    rows = [
        ("????", summary.get("system_type_zh", "?")),
        ("??????", summary.get("crack_resistance_tendency_zh", "?")),
        ("??????", summary.get("thermal_constraint_capacity_zh", "?")),
        ("??????", summary.get("expected_bridging_zh", "?")),
        ("??????", summary.get("dispersion_risk_zh", "?")),
    ]
    items = "".join(
        f"<li><strong>{_esc_html(k)}??/strong>{_esc_html(v)}</li>" for k, v in rows
    )
    return f'<div class="sfc-eng-explain-card"><h4>fiber_engineering_summary</h4><ul>{items}</ul></div>'


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
    return f'<div class="sfc-shap-narrative">{_esc_html(text)}</div>'


def mechanical_context_line_html(grade: str, fiber_pct: str) -> str:
    return (
        f'<div class="sfc-mech-context">'
        f"<strong>???????</strong>{_esc_html(grade)}??????"
        f' &nbsp;?&nbsp; <strong>???????</strong>{_esc_html(fiber_pct)}'
        f"</div>"
    )


def tab_heading_html(text: str) -> str:
    return f'<p class="sfc-tab-h1">{_esc_html(text)}</p>'


def section_title_html(text: str) -> str:
    """????????????? markdown unsafe_html ?????"""
    return f'<p class="sfc-section-title">{text}</p>'


def empty_state_hint_html(message: str) -> str:
    """??????????????? .sfc-empty-hint???"""
    safe = (
        message.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<div class="sfc-empty-hint">{safe}</div>'


def sidebar_footer_html() -> str:
    """????????????????"""
    return (
        '<p class="sfc-sidebar-foot">'
        "??????????? ???? XGBoost ? ???????
        "</p>"
    )


def mechanism_flow_html() -> str:
    """????????????????"""
    return """
<div class="sfc-flow-wrap">
  <span class="sfc-flow-box">????????????????/span>
  <span class="sfc-flow-arrow">??/span>
  <span class="sfc-flow-box">????+ XGBoost</span>
  <span class="sfc-flow-arrow">??/span>
  <span class="sfc-flow-box">????</span>
  <span class="sfc-flow-arrow">?</span>
  <span class="sfc-flow-box">????</span>
  <span class="sfc-flow-arrow">?</span>
  <span class="sfc-flow-box">?????/span>
</div>
"""
