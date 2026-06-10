# -*- coding: utf-8 -*-
"""Replace home_kpi_dashboard_html with clean UTF-8 version."""
from __future__ import annotations

from pathlib import Path

P = Path(__file__).resolve().parent.parent / "src" / "ui_theme.py"
raw = P.read_bytes()
try:
    text = raw.decode("utf-8")
except UnicodeDecodeError:
    text = raw.decode("latin-1").replace("\xb7", "\u00b7")

start = text.find("def home_kpi_dashboard_html(")
end = text.find("\ndef kpi_card_html(", start)
if start == -1 or end == -1:
    raise SystemExit(f"anchors missing start={start} end={end}")

FUNC = '''def home_kpi_dashboard_html(
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
    """\u9996\u9875 KPI\uff1a1 \u4e3b\u5361 + 2 \u8f85\u5361 + 3 \u6b21\u7ea7\u5361\u3002"""
    hero_cls = (
        f"sfc-hkpi-hero sfc-hkpi-hero-{risk_tone}"
        if risk_tone in ("high", "mid", "low")
        else "sfc-hkpi-hero"
    )
    badge_cls = "sfc-hkpi-badge-red" if risk_tone == "high" else "sfc-hkpi-badge-gray"
    width_badge_cls = (
        "sfc-hkpi-badge-blue" if width_tag == "\u5b89\u5168\u8303\u56f4" else "sfc-hkpi-badge-gray"
    )
    prob_badge_cls = (
        "sfc-hkpi-badge-red"
        if prob_band == "\u9ad8\u9884\u8b66\u5e26"
        else "sfc-hkpi-badge-blue"
        if prob_band == "\u4e2d\u9884\u8b66\u5e26"
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
    <div class="sfc-hkpi {hero_cls}">
      <div class="sfc-hkpi-hero-head">
        <span class="sfc-hkpi-hero-icon">&#9888;</span>
        <span class="sfc-hkpi-badge {badge_cls}">{_esc_html(risk_level)}</span>
      </div>
      <p class="sfc-hkpi-label">\u5f00\u88c2\u98ce\u9669\u7b49\u7ea7</p>
      <p class="sfc-hkpi-value">{_esc_html(risk_level)}</p>
      <p class="sfc-hkpi-sub">{_esc_html(risk_subtitle)}</p>
    </div>
    <div class="sfc-home-kpi-side">
      <motion class="sfc-hkpi sfc-hkpi-side sfc-hkpi-prob">
        <p class="sfc-hkpi-label">\u98ce\u9669\u6982\u7387 P</p>
        <p class="sfc-hkpi-value">{_esc_html(prob_value)}</p>
        <div class="sfc-hkpi-meta">
          <span class="sfc-hkpi-badge {prob_badge_cls}">{_esc_html(prob_band)}</span>
        </div>
      </div>
      <div class="sfc-hkpi sfc-hkpi-side sfc-hkpi-width">
        <p class="sfc-hkpi-label">\u88c2\u7f1d\u5bbd\u5ea6</p>
        <p class="sfc-hkpi-value">{_esc_html(width_value)}<span class="sfc-hkpi-unit">mm</span></p>
        <div class="sfc-hkpi-meta">{width_tag_html}</div>
      </div>
    </div>
  </div>
  <div class="sfc-home-kpi-row2">
    <div class="sfc-hkpi sfc-hkpi-mini">
      <p class="sfc-hkpi-label">\u88c2\u7f1d\u5bc6\u5ea6</p>
      <p class="sfc-hkpi-value">{_esc_html(density_value)}<span class="sfc-hkpi-unit">{_esc_html(density_unit)}</span></p>
    </div>
    <div class="sfc-hkpi sfc-hkpi-mini">
      <p class="sfc-hkpi-label">\u6297\u538b\u5f3a\u5ea6\uff08\u529b\u5b66\u53c2\u8003\uff09</p>
      <p class="sfc-hkpi-value">{_esc_html(compressive_value)}<span class="sfc-hkpi-unit">MPa</span></p>
      <p class="sfc-hkpi-mini-tag">\u516c\u5f0f\u57fa\u7ebf \u00b7 \u975e\u4e3b\u6a21\u578b</p>
    </div>
    <div class="sfc-hkpi sfc-hkpi-mini">
      <p class="sfc-hkpi-label">\u6297\u6298\u5f3a\u5ea6\uff08\u529b\u5b66\u53c2\u8003\uff09</p>
      <p class="sfc-hkpi-value">{_esc_html(flexural_value)}<span class="sfc-hkpi-unit">MPa</span></p>
      <p class="sfc-hkpi-mini-tag">\u516c\u5f0f\u57fa\u7ebf \u00b7 \u975e\u4e3b\u6a21\u578b</p>
    </div>
  </div>
</div>
"""
'''

FUNC = FUNC.replace("<motion ", "<motion ").replace("</motion>", "</motion>").replace("<motion ", "<div ").replace("</motion>", "</div>")

text = text[:start] + FUNC + text[end:]
P.write_text(text, encoding="utf-8")
compile(text, str(P), "exec")

# verify labels
assert "\u6297\u538b\u5f3a\u5ea6\uff08\u529b\u5b66\u53c2\u8003\uff09" in text
assert "\u6297\u6298\u5f3a\u5ea6\uff08\u529b\u5b66\u53c2\u8003\uff09" in text
print("ok")
