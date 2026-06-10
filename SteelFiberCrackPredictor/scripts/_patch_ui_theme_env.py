"""Patch ui_theme.py: environment KPI CSS + HTML helpers."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "src" / "ui_theme.py"

ENV_CSS = """
    .sfc-env-kpi-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.55rem;
        margin: 0.5rem 0 0.85rem 0;
    }
    .sfc-env-kpi {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-left: 3px solid #475569;
        border-radius: 8px;
        padding: 0.6rem 0.7rem;
    }
    .sfc-env-kpi-k {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        margin: 0 0 0.2rem 0;
    }
    .sfc-env-kpi-v {
        font-size: 0.98rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
        line-height: 1.25;
    }
    .sfc-env-kpi-h {
        font-size: 0.7rem;
        color: #94a3b8;
        margin: 0.25rem 0 0 0;
        line-height: 1.3;
    }
    .sfc-env-driver-card {
        margin: 0 0 1rem 0;
        padding: 0.85rem 1rem;
        border-radius: 10px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #475569;
    }
    .sfc-env-driver-card h4 {
        margin: 0 0 0.45rem 0;
        font-size: 0.98rem;
        color: #1a5f7a;
    }
    .sfc-env-driver-card ul {
        margin: 0;
        padding-left: 1.1rem;
        color: #475569;
        font-size: 0.88rem;
        line-height: 1.55;
    }
"""

ENV_FUNCS = """

def environment_engineering_kpi_html(kpis: list[tuple[str, str, str]]) -> str:
    cells = []
    for label, value, hint in kpis:
        cells.append(
            f'<div class="sfc-env-kpi">'
            f'<p class="sfc-env-kpi-k">{_esc_html(label)}</p>'
            f'<p class="sfc-env-kpi-v">{_esc_html(value)}</p>'
            f'<p class="sfc-env-kpi-h">{_esc_html(hint)}</p>'
            f"</div>"
        )
    return '<div class="sfc-env-kpi-row">' + "".join(cells) + "</div>"


def environment_engineering_summary_panel_html(summary: dict[str, str]) -> str:
    rows = [
        ("蒸发风险", summary.get("evaporation_risk_zh", "\u2014")),
        ("温差风险", summary.get("thermal_gradient_risk_zh", "\u2014")),
        ("养护充分性", summary.get("curing_adequacy_zh", "\u2014")),
        ("表层开裂倾向", summary.get("surface_crack_tendency_zh", "\u2014")),
        ("热裂缝倾向", summary.get("thermal_crack_tendency_zh", "\u2014")),
    ]
    items = "".join(
        f"<li><strong>{_esc_html(k)}\uff1a</strong>{_esc_html(v)}</li>" for k, v in rows
    )
    return (
        f'<div class="sfc-eng-explain-card">'
        f"<h4>environment_engineering_summary</h4><ul>{items}</ul></div>"
    )


def environment_driver_analysis_html(bullets: list[str]) -> str:
    items = "".join(f"<li>{_esc_html(x)}</li>" for x in bullets)
    return (
        f'<div class="sfc-env-driver-card">'
        f"<h4>\u73af\u5883\u9a71\u52a8\u5f00\u88c2\u5206\u6790</h4><ul>{items}</ul></div>"
    )
"""

text = P.read_text(encoding="utf-8")

if "def environment_engineering_kpi_html" not in text:
    anchor = "def engineering_explain_card_html"
    if anchor not in text:
        raise SystemExit("anchor not found for env funcs")
    text = text.replace(f"\n\n{anchor}", ENV_FUNCS + f"\n\n{anchor}", 1)

if ".sfc-env-kpi-row" not in text:
    marker = ".sfc-fiber-bridge-grid"
    if marker not in text:
        raise SystemExit("css marker not found")
    text = text.replace(marker, ENV_CSS + "\n    " + marker, 1)

P.write_text(text, encoding="utf-8")
compile(text, str(P), "exec")
print("patched", P)
