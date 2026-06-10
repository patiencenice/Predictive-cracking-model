# -*- coding: utf-8 -*-
"""Rebuild ui_theme.py: preserve CSS + patched helpers, restore all Chinese strings."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "ui_theme.py"
REBUILD = ROOT / "scripts" / "_rebuild_ui_theme_tail.py"

raw = SRC.read_bytes().decode("utf-8", errors="replace")
raw = re.sub(r"\r+\n", "\n", raw).replace("\r", "\n")
marker = 'STREAMLIT_CUSTOM_CSS = """'
css_start = raw.index(marker)
css_end = raw.index('"""', css_start + len(marker)) + 3
css_block = raw[css_start:css_end]

rebuild_src = REBUILD.read_text(encoding="utf-8")
tail_start = rebuild_src.index("TAIL = '''") + len("TAIL = '''")
tail_end = rebuild_src.index("'''", tail_start)
TAIL = rebuild_src[tail_start:tail_end].replace("<motion ", "<div ").replace("</motion>", "</div>")

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

def _read_patch(path: Path, start: str, end: str = "'''\n") -> str:
    chunk = path.read_text(encoding="utf-8").split(start, 1)[1].split(end, 1)[0]
    chunk = chunk.replace("<motion ", "<div ").replace("</motion>", "</div>")
    chunk = chunk.replace("f\\'", "f'").replace("\\'", "'")
    return chunk


PATCHES = {
    "hero_banner_html": _read_patch(
        ROOT / "scripts" / "_patch_home_prediction_py.py", "HERO_NEW = '''"
    ),
    "home_kpi_dashboard_html": _read_patch(
        ROOT / "scripts" / "_patch_mech_labels.py", "FUNC = '''"
    ),
    "kpi_card_html": _read_patch(
        ROOT / "scripts" / "_patch_home_prediction_py.py", "KPI_NEW = '''"
    ),
    "engineering_conclusion_compact_html": _read_patch(
        ROOT / "scripts" / "_patch_home_prediction_py.py", "CONCLUSION_INSERT = '''"
    ).strip()
    + "\n",
}

ENV_BLOCK = (
    _read_patch(ROOT / "scripts" / "_patch_ui_theme_env.py", 'ENV_FUNCS = """', '"""')
    + "\n"
)


def _replace_function(src: str, name: str, new_body: str) -> str:
    start = src.find(f"def {name}(")
    if start == -1:
        raise SystemExit(f"{name} not found")
    nxt = src.find("\ndef ", start + 4)
    if nxt == -1:
        nxt = len(src)
    return src[:start] + new_body.rstrip() + "\n\n" + src[nxt + 1 :]


def _insert_before_function(src: str, anchor: str, block: str) -> str:
    pos = src.find(f"def {anchor}(")
    if pos == -1:
        raise SystemExit(f"anchor {anchor} not found")
    return src[:pos] + block.rstrip() + "\n\n" + src[pos:]


text = HEADER + css_block + "\n\n" + TAIL.strip() + "\n"

for name, body in PATCHES.items():
    if f"def {name}(" in text:
        text = _replace_function(text, name, body)
    elif name == "home_kpi_dashboard_html":
        text = _insert_before_function(text, "kpi_card_html", body)
    elif name == "engineering_conclusion_compact_html":
        text = _insert_before_function(text, "engineering_conclusion_html", body)

for fn in (
    "environment_engineering_kpi_html",
    "environment_engineering_summary_panel_html",
    "environment_driver_analysis_html",
):
    part = "def " + fn + ENV_BLOCK.split(f"def {fn}", 1)[1]
    part = part.split("\n\n", 1)[0].strip() + "\n"
    if f"def {fn}(" in text:
        text = _replace_function(text, fn, part)
    else:
        text = _insert_before_function(text, "engineering_explain_card_html", part)

text = re.sub(r"\n{3,}", "\n\n", text)
SRC.write_text(text, encoding="utf-8", newline="\n")
compile(text, str(SRC), "exec")

sys.path.insert(0, str(ROOT))
from src.ui_theme import (
    hero_banner_html,
    mechanical_context_line_html,
    mechanical_summary_html,
    risk_tone_from_level,
)

assert risk_tone_from_level("\u9ad8\u98ce\u9669") == "high"
s = mechanical_summary_html(["\u6297\u538b"], "\u811a\u6ce8")
assert "\u529b\u5b66\u6027\u80fd\u7ed3\u8bba" in s
assert "\u5f3a\u5ea6\u7b49\u7ea7" in mechanical_context_line_html("C30", "1.0%")
h = hero_banner_html("t", "s", risk_level="\u9ad8\u98ce\u9669")
assert "\n\n\n" not in h
assert "\u5f53\u524d\u98ce\u9669" in h
print("rebuilt OK", SRC.stat().st_size)
