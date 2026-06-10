"""Homepage KPI layout: 1 hero + 2 side + 3 mini cards."""
from __future__ import annotations

from pathlib import Path

P = Path(__file__).resolve().parent.parent / "src" / "ui_theme.py"
MARKER = "/* sfc-home-kpi-layout */"

CSS = """
    /* sfc-home-kpi-layout */
    .sfc-home-kpi {
        margin: 0 0 1.75rem 0;
    }
    .sfc-home-kpi-row1 {
        display: grid;
        grid-template-columns: 45fr 55fr;
        gap: 1rem;
        margin-bottom: 1rem;
        align-items: stretch;
    }
    @media (max-width: 860px) {
        .sfc-home-kpi-row1 { grid-template-columns: 1fr; }
    }
    .sfc-home-kpi-row2 {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
    }
    @media (max-width: 720px) {
        .sfc-home-kpi-row2 { grid-template-columns: 1fr; }
    }
    .sfc-hkpi {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 8px rgba(15, 23, 42, 0.04);
        background: #ffffff;
    }
    .sfc-hkpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.02em;
        margin: 0 0 0.35rem 0;
    }
    .sfc-hkpi-value {
        font-weight: 700;
        color: #0f172a;
        line-height: 1.05;
        margin: 0;
    }
    .sfc-hkpi-unit {
        font-size: 0.72rem;
        font-weight: 500;
        color: #94a3b8;
        margin-left: 0.12rem;
    }
    .sfc-hkpi-badge {
        display: inline-block;
        padding: 0.14rem 0.5rem;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 600;
        border: 1px solid transparent;
    }
    .sfc-hkpi-badge-red {
        background: #fee2e2;
        color: #991b1b;
        border-color: #fecaca;
    }
    .sfc-hkpi-badge-blue {
        background: #eff6ff;
        color: #1d4ed8;
        border-color: #bfdbfe;
    }
    .sfc-hkpi-badge-gray {
        background: #f1f5f9;
        color: #475569;
        border-color: #e2e8f0;
    }
    .sfc-hkpi-sub {
        font-size: 0.82rem;
        color: #64748b;
        margin: 0.55rem 0 0 0;
        line-height: 1.5;
        max-width: 22rem;
    }
    .sfc-hkpi-hero {
        padding: 1.35rem 1.25rem 1.3rem 1.25rem;
        border-left: 6px solid #b91c1c;
        background: #fef2f2;
        min-height: 11.5rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .sfc-hkpi-hero-head {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.5rem;
    }
    .sfc-hkpi-hero-icon { font-size: 1.35rem; line-height: 1; }
    .sfc-hkpi-hero .sfc-hkpi-label { color: #991b1b; font-size: 0.8rem; }
    .sfc-hkpi-hero .sfc-hkpi-value {
        font-size: clamp(2.4rem, 4.5vw, 3.35rem);
        color: #991b1b;
    }
    .sfc-hkpi-hero-mid {
        border-left-color: #94a3b8;
        background: #f8fafc;
    }
    .sfc-hkpi-hero-mid .sfc-hkpi-label { color: #475569; }
    .sfc-hkpi-hero-mid .sfc-hkpi-value { color: #334155; }
    .sfc-hkpi-hero-low {
        border-left-color: #64748b;
        background: #f8fafc;
    }
    .sfc-hkpi-hero-low .sfc-hkpi-label { color: #64748b; }
    .sfc-hkpi-hero-low .sfc-hkpi-value { color: #1e293b; }
    .sfc-home-kpi-side {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        min-height: 11.5rem;
    }
    .sfc-hkpi-side {
        flex: 1;
        padding: 0.85rem 1rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .sfc-hkpi-prob {
        background: #f1f5f9;
        border-left: none;
    }
    .sfc-hkpi-prob .sfc-hkpi-value {
        font-size: clamp(1.55rem, 2.4vw, 2rem);
        color: #334155;
    }
    .sfc-hkpi-width {
        background: #eff6ff;
        border-left: none;
    }
    .sfc-hkpi-width .sfc-hkpi-value {
        font-size: clamp(1.85rem, 3vw, 2.45rem);
        color: #0f172a;
    }
    .sfc-hkpi-width .sfc-hkpi-unit { font-size: 0.65rem; }
    .sfc-hkpi-meta { margin-top: 0.35rem; }
    .sfc-hkpi-mini {
        padding: 0.65rem 0.8rem;
        background: #fafbfc;
        border: 1px solid #eef2f7;
        box-shadow: none;
    }
    .sfc-hkpi-mini .sfc-hkpi-label {
        font-size: 0.68rem;
        color: #94a3b8;
        margin-bottom: 0.2rem;
    }
    .sfc-hkpi-mini .sfc-hkpi-value {
        font-size: 1.05rem;
        font-weight: 600;
        color: #64748b;
    }
    .sfc-hkpi-mini .sfc-hkpi-unit { font-size: 0.62rem; }
    .sfc-hkpi-mini-tag {
        font-size: 0.6rem;
        color: #cbd5e1;
        margin-top: 0.15rem;
    }
    .sfc-gauge-compact { margin-top: 0.5rem; }
    .sfc-gauge-compact .sfc-section-title {
        font-size: 0.95rem !important;
        color: #64748b !important;
    }
"""

FUNC = '''
def home_kpi_dashboard_html(
    *,
    risk_level: str,
    risk_tone: str,
    risk_subtitle: str,
    prob_value: str,
    prob_band: str,
    width_value: str,
    width_tag: str,
    density_value: str,
    density_unit: str,
    compressive_value: str,
    flexural_value: str,
) -> str:
    """首页 KPI：1 主卡 + 2 辅卡 + 3 次级卡。"""
    hero_cls = (
        f"sfc-hkpi-hero sfc-hkpi-hero-{risk_tone}"
        if risk_tone in ("high", "mid", "low")
        else "sfc-hkpi-hero"
    )
    badge_cls = "sfc-hkpi-badge-red" if risk_tone == "high" else "sfc-hkpi-badge-gray"
    width_badge_cls = "sfc-hkpi-badge-blue" if width_tag == "安全范围" else "sfc-hkpi-badge-gray"
    prob_badge_cls = (
        "sfc-hkpi-badge-red"
        if prob_band == "高预警带"
        else "sfc-hkpi-badge-blue"
        if prob_band == "中预警带"
        else "sfc-hkpi-badge-gray"
    )
    width_tag_html = (
        f\'<span class="sfc-hkpi-badge {width_badge_cls}">{_esc_html(width_tag)}</span>\'
        if width_tag
        else ""
    )
    return f"""
<div class="sfc-home-kpi">
  <div class="sfc-home-kpi-row1">
    <motion class="sfc-hkpi {hero_cls}">
      <div class="sfc-hkpi-hero-head">
        <span class="sfc-hkpi-hero-icon">&#9888;</span>
        <span class="sfc-hkpi-badge {badge_cls}">{_esc_html(risk_level)}</span>
      </div>
      <p class="sfc-hkpi-label">开裂风险等级</p>
      <p class="sfc-hkpi-value">{_esc_html(risk_level)}</p>
      <p class="sfc-hkpi-sub">{_esc_html(risk_subtitle)}</p>
    </div>
    <div class="sfc-home-kpi-side">
      <div class="sfc-hkpi sfc-hkpi-side sfc-hkpi-prob">
        <p class="sfc-hkpi-label">风险概率 P</p>
        <p class="sfc-hkpi-value">{_esc_html(prob_value)}</p>
        <div class="sfc-hkpi-meta">
          <span class="sfc-hkpi-badge {prob_badge_cls}">{_esc_html(prob_band)}</span>
        </div>
      </div>
      <div class="sfc-hkpi sfc-hkpi-side sfc-hkpi-width">
        <p class="sfc-hkpi-label">裂缝宽度</p>
        <p class="sfc-hkpi-value">{_esc_html(width_value)}<span class="sfc-hkpi-unit">mm</span></p>
        <div class="sfc-hkpi-meta">{width_tag_html}</div>
      </div>
    </div>
  </div>
  <div class="sfc-home-kpi-row2">
    <div class="sfc-hkpi sfc-hkpi-mini">
      <p class="sfc-hkpi-label">裂缝密度</p>
      <p class="sfc-hkpi-value">{_esc_html(density_value)}<span class="sfc-hkpi-unit">{_esc_html(density_unit)}</span></p>
    </div>
    <motion class="sfc-hkpi sfc-hkpi-mini">
      <p class="sfc-hkpi-label">抗压强度</p>
      <p class="sfc-hkpi-value">{_esc_html(compressive_value)}<span class="sfc-hkpi-unit">MPa</span></p>
      <p class="sfc-hkpi-mini-tag">公式基线 · 非主模型</p>
    </div>
    <div class="sfc-hkpi sfc-hkpi-mini">
      <p class="sfc-hkpi-label">抗折强度</p>
      <p class="sfc-hkpi-value">{_esc_html(flexural_value)}<span class="sfc-hkpi-unit">MPa</span></p>
      <p class="sfc-hkpi-mini-tag">公式基线 · 非主模型</p>
    </div>
  </div>
</div>
"""
'''

FUNC = FUNC.replace("<motion ", "<div ").replace("</motion>", "</div>")

text = P.read_text(encoding="utf-8")
if MARKER not in text:
    text = text.replace("</style>", CSS + "\n</style>", 1)

if "def home_kpi_dashboard_html" not in text:
    anchor = "def kpi_card_html("
    text = text.replace(anchor, FUNC + "\n\n" + anchor, 1)

P.write_text(text, encoding="utf-8")
compile(text, str(P), "exec")
print("patched")
