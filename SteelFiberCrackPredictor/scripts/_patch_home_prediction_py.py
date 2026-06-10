"""Patch ui_theme.py Python helpers for homepage prediction polish."""
from __future__ import annotations

from pathlib import Path

P = Path(__file__).resolve().parent.parent / "src" / "ui_theme.py"
text = P.read_text(encoding="utf-8")

HERO_NEW = '''def hero_banner_html(
    title: str,
    subtitle: str,
    *,
    risk_level: str | None = None,
    model_version: str = "v2.1",
    model_status: str = "工程辅助评估",
) -> str:
    """页头横幅：标题 + 风险/版本徽章。"""
    tone = risk_tone_from_level(risk_level or "")
    badge_cls = f"sfc-badge sfc-badge-risk-{tone}" if tone != "neutral" else "sfc-badge sfc-badge-neutral"
    risk_txt = _esc_html(risk_level or "待评估")
    return f"""
<div class="sfc-hero">
  <div class="sfc-hero-grid">
    <div class="sfc-hero-left">
      <div class="sfc-hero-title">{_esc_html(title)}</div>
      <div class="sfc-hero-en">Steel Fiber Concrete Crack &amp; Risk Assessment Platform</div>
      <p class="sfc-hero-sub">{_esc_html(subtitle)}</p>
    </div>
    <div class="sfc-hero-right sfc-hero-right-aligned">
      <div class="sfc-badge-row sfc-badge-row-primary">
        <span class="{badge_cls} sfc-badge-risk-main">当前风险 · {risk_txt}</span>
        <span class="sfc-badge sfc-badge-version-mini">{_esc_html(model_version)}</span>
      </div>
      <p class="sfc-hero-status-muted">{_esc_html(model_status)}</p>
    </div>
  </div>
</div>
"""'''

KPI_NEW = '''def kpi_card_html(
    *,
    label: str,
    value: str,
    unit: str = "",
    hint: str = "",
    tier: str = "secondary",
    tone: str = "neutral",
    icon: str = "",
    tag: str = "",
    variant: str = "",
) -> str:
    tier_cls = "sfc-kpi-primary" if tier == "primary" else "sfc-kpi-secondary"
    tone_cls = (
        f"sfc-kpi-tone-{tone}"
        if tone in ("low", "mid", "high", "comp", "flex", "prob", "width")
        else "sfc-kpi-tone-neutral"
    )
    variant_cls = f" sfc-kpi-variant-{variant}" if variant else ""
    ic = f\'<span class="sfc-kpi-icon">{icon}</span>\' if icon else ""
    u = f\'<span class="sfc-kpi-unit">{_esc_html(unit)}</span>\' if unit else ""
    h = f\'<div class="sfc-kpi-hint">{_esc_html(hint)}</div>\' if hint else ""
    tag_html = f\'<span class="sfc-kpi-tag">{_esc_html(tag)}</span>\' if tag else ""
    return f"""
<div class="sfc-kpi {tier_cls} {tone_cls}{variant_cls}">
  <div class="sfc-kpi-label">{ic}{_esc_html(label)}</div>
  <div class="sfc-kpi-value">{_esc_html(value)}{u}</div>
  {tag_html}
  {h}
</div>
"""'''

CONCLUSION_INSERT = '''

def engineering_conclusion_compact_html(
    judgment: str,
    reasons: list[str],
    suggestions: list[str],
) -> str:
    """首页预测结果：三行压缩工程结论。"""
    reasons_txt = "\uff1b".join(reasons[:2]) + "\u3002" if reasons else "\u2014"
    sugg_txt = "\uff1b".join(suggestions[:3]) + "\u3002" if suggestions else "\u2014"
    return f"""
<div class="sfc-conclusion sfc-conclusion-compact">
  <div class="sfc-conclusion-compact-row">
    <span class="sfc-cc-k">\u3010\u5f53\u524d\u5224\u65ad\u3011</span>
    <span class="sfc-cc-v sfc-cc-judgment">{_esc_html(judgment)}</span>
  </div>
  <div class="sfc-conclusion-compact-row">
    <span class="sfc-cc-k">\u3010\u4e3b\u8981\u539f\u56e0\u3011</span>
    <span class="sfc-cc-v">{_esc_html(reasons_txt)}</span>
  </div>
  <div class="sfc-conclusion-compact-row">
    <span class="sfc-cc-k">\u3010\u5de5\u7a0b\u5efa\u8bae\u3011</span>
    <span class="sfc-cc-v">{_esc_html(sugg_txt)}</span>
  </div>
</div>
"""
'''

def _replace_function(src: str, name: str, new_body: str) -> str:
    start = src.find(f"def {name}(")
    if start == -1:
        raise SystemExit(f"{name} not found")
    nxt = src.find("\ndef ", start + 4)
    if nxt == -1:
        nxt = len(src)
    return src[:start] + new_body + src[nxt:]

text = _replace_function(text, "hero_banner_html", HERO_NEW)
text = _replace_function(text, "kpi_card_html", KPI_NEW)
text = text.replace("<motion ", "<div ").replace("</motion>", "</div>")

if "def engineering_conclusion_compact_html" not in text:
    anchor = "def engineering_conclusion_html("
    text = text.replace(anchor, CONCLUSION_INSERT.lstrip() + "\n\n" + anchor, 1)

P.write_text(text, encoding="utf-8")
compile(text, str(P), "exec")
print("ok")
