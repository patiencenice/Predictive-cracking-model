# -*- coding: utf-8 -*-
"""Normalize corrupted line endings in ui_theme.py and restore key HTML helpers."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "src" / "ui_theme.py"

raw = P.read_bytes()
text = raw.decode("utf-8", errors="replace")
text = re.sub(r"\r+\n", "\n", text)
text = re.sub(r"\n{3,}", "\n\n", text)
text = text.replace("\r", "\n")

HERO = '''def hero_banner_html(
    title: str,
    subtitle: str,
    *,
    risk_level: str | None = None,
    model_version: str = "v2.1",
    model_status: str = "\u5de5\u7a0b\u8f85\u52a9\u8bc4\u4f30",
) -> str:
    """\u9875\u5934\u6a2a\u5e45\uff1a\u6807\u9898 + \u98ce\u9669/\u7248\u672c\u5fbd\u7ae0\u3002"""
    tone = risk_tone_from_level(risk_level or "")
    badge_cls = f"sfc-badge sfc-badge-risk-{tone}" if tone != "neutral" else "sfc-badge sfc-badge-neutral"
    risk_txt = _esc_html(risk_level or "\u5f85\u8bc4\u4f30")
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
        <span class="{badge_cls} sfc-badge-risk-main">\u5f53\u524d \u98ce\u9669 \u00b7 {risk_txt}</span>
        <span class="sfc-badge sfc-badge-version-mini">{_esc_html(model_version)}</span>
      </div>
      <p class="sfc-hero-status-muted">{_esc_html(model_status)}</p>
    </div>
  </div>
</div>
"""'''

COMPACT = '''def engineering_conclusion_compact_html(
    judgment: str,
    reasons: list[str],
    suggestions: list[str],
) -> str:
    """\u9996\u9875\u9884\u6d4b\u7ed3\u679c\uff1a\u4e09\u884c\u538b\u7f29\u5de5\u7a0b\u7ed3\u8bba\u3002"""
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
"""'''


def _replace_function(src: str, name: str, new_body: str) -> str:
    start = src.find(f"def {name}(")
    if start == -1:
        raise SystemExit(f"{name} not found")
    nxt = src.find("\ndef ", start + 4)
    if nxt == -1:
        nxt = len(src)
    return src[:start] + new_body + src[nxt:]


text = _replace_function(text, "hero_banner_html", HERO)
if "def engineering_conclusion_compact_html(" in text:
    text = _replace_function(text, "engineering_conclusion_compact_html", COMPACT)

P.write_text(text, encoding="utf-8", newline="\n")
compile(text, str(P), "exec")

sys.path.insert(0, str(ROOT))
from src.ui_theme import hero_banner_html

out = hero_banner_html("\u6807\u9898", "\u526f\u6807\u9898", risk_level="\u9ad8\u98ce\u9669")
if "\n\n\n" in out:
    raise SystemExit("hero HTML still has triple newlines")
if "\u5f53\u524d \u98ce\u9669" not in out:
    raise SystemExit("hero badge text missing")
print("ok", "bytes", P.stat().st_size, "hero_len", len(out))
