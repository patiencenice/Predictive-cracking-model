"""Homepage prediction tab visual polish — CSS only."""
from __future__ import annotations

from pathlib import Path

P = Path(__file__).resolve().parent.parent / "src" / "ui_theme.py"
MARKER = "/* sfc-home-prediction-polish */"
CSS = """
    /* sfc-home-prediction-polish */
    .sfc-kpi-grid-primary {
        gap: 1rem !important;
        margin: 0 0 1rem 0 !important;
    }
    .sfc-kpi-grid-secondary {
        gap: 0.75rem !important;
        margin: 0 0 1.35rem 0 !important;
    }
    .sfc-kpi {
        border-radius: 12px !important;
        box-shadow: 0 1px 8px rgba(15, 23, 42, 0.04) !important;
        padding: 1rem 1.05rem 1.05rem 1.05rem !important;
    }
    .sfc-kpi-primary {
        padding: 1.1rem 1.15rem 1.15rem 1.15rem !important;
        min-height: 7.75rem !important;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.06) !important;
    }
    .sfc-kpi-tone-high {
        border-left: 6px solid #b91c1c !important;
        background: #fef2f2 !important;
    }
    .sfc-kpi-tone-high .sfc-kpi-value { color: #991b1b !important; }
    .sfc-kpi-tone-mid {
        border-left: 6px solid #c2410c !important;
        background: #fff7ed !important;
    }
    .sfc-kpi-tone-mid .sfc-kpi-value { color: #9a3412 !important; }
    .sfc-kpi-tone-low {
        border-left: 6px solid #059669 !important;
        background: #ecfdf5 !important;
    }
    .sfc-kpi-tone-low .sfc-kpi-value { color: #047857 !important; }
    .sfc-kpi-variant-risk-hero.sfc-kpi-tone-high .sfc-kpi-label { color: #991b1b; font-weight: 700; }
    .sfc-kpi-variant-risk-hero .sfc-kpi-icon { font-size: 1.45rem !important; }
    .sfc-kpi-variant-risk-hero .sfc-kpi-icon { font-size: 1.45rem !important; }
    .sfc-kpi-tone-prob {
        border-left: 4px solid #cbd5e1 !important;
        background: #f8fafc !important;
        box-shadow: 0 1px 6px rgba(15, 23, 42, 0.03) !important;
    }
    .sfc-kpi-tone-prob .sfc-kpi-value {
        color: #475569 !important;
        font-size: clamp(1.55rem, 2.6vw, 2.2rem) !important;
    }
    .sfc-kpi-tone-prob .sfc-kpi-label { color: #94a3b8; }
    .sfc-kpi-variant-width {
        border-left: 5px solid #1a5f7a !important;
        background: #ffffff !important;
    }
    .sfc-kpi-variant-width .sfc-kpi-value {
        font-size: clamp(2.15rem, 3.9vw, 3.05rem) !important;
        color: #0f172a !important;
    }
    .sfc-kpi-variant-width .sfc-kpi-unit {
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        color: #64748b !important;
    }
    .sfc-kpi-tag {
        display: inline-block;
        margin-top: 0.35rem;
        padding: 0.12rem 0.45rem;
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: 600;
        color: #475569;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
    }
    .sfc-kpi-variant-width .sfc-kpi-tag {
        color: #1a5f7a;
        background: #eff6ff;
        border-color: #bfdbfe;
    }
    .sfc-kpi-variant-mech {
        background: #fafbfc !important;
        border-left: 3px solid #e2e8f0 !important;
        padding: 0.55rem 0.75rem !important;
        min-height: auto !important;
        box-shadow: none !important;
    }
    .sfc-kpi-variant-mech .sfc-kpi-label {
        font-size: 0.72rem !important;
        color: #94a3b8 !important;
    }
    .sfc-kpi-variant-mech .sfc-kpi-value {
        font-size: 1.12rem !important;
        color: #64748b !important;
        font-weight: 600 !important;
    }
    .sfc-kpi-variant-mech .sfc-kpi-tag {
        font-size: 0.62rem;
        color: #94a3b8;
        background: transparent;
        border: none;
        padding: 0;
        margin-top: 0.2rem;
    }
    .sfc-conclusion-compact {
        padding: 0.85rem 1rem !important;
        margin-bottom: 0.85rem !important;
        border-left-width: 4px !important;
    }
    .sfc-conclusion-compact-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem 0.5rem;
        margin: 0 0 0.45rem 0;
        font-size: 0.9rem;
        line-height: 1.55;
        color: #334155;
    }
    .sfc-conclusion-compact-row:last-child { margin-bottom: 0; }
    .sfc-cc-k {
        font-weight: 700;
        color: #1a5f7a;
        white-space: nowrap;
    }
    .sfc-cc-v { flex: 1 1 12rem; color: #475569; }
    .sfc-cc-judgment { font-weight: 700; color: #0f172a; }
    .sfc-hero-right-aligned {
        align-items: flex-end;
        justify-content: center;
    }
    .sfc-badge-row-primary {
        align-items: center;
        gap: 0.35rem !important;
    }
    .sfc-badge-risk-main { font-size: 0.82rem; padding: 0.32rem 0.75rem; }
    .sfc-badge-version-mini {
        font-size: 0.68rem !important;
        padding: 0.18rem 0.45rem !important;
        background: #f1f5f9 !important;
        color: #64748b !important;
        border-color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    .sfc-hero-status-muted {
        margin: 0.15rem 0 0 0;
        font-size: 0.68rem;
        color: #94a3b8;
        text-align: right;
    }
"""

text = P.read_text(encoding="utf-8")
if MARKER not in text:
    text = text.replace("</style>", CSS + "\n</style>", 1)
    P.write_text(text, encoding="utf-8")
    compile(text, str(P), "exec")
    print("css patched")
else:
    print("css already present")
