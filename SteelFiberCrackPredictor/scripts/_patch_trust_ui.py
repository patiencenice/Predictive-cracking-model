# -*- coding: utf-8 -*-
"""Fix trust UI strings in ui_theme.py."""
from __future__ import annotations

import re
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "src" / "ui_theme.py"
text = P.read_text(encoding="utf-8", errors="replace")
text = re.sub(r"\r+\n", "\n", text).replace("\r", "\n")

TRUST_BANNER = r'''def trust_conclusion_banner_html(
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
  <div class="sfc-trust-banner-title">当前样本可信度 · {_esc_html(level_label)}</div>
  <p class="sfc-trust-banner-intro">{_esc_html(intro)}</p>
  {"<ul>" + pos_ul + "</ul>" if positives else ""}
  {caveats_block}
  <p class="sfc-trust-banner-caveat">以上为工程辅助评估，正式判定请以标准试验与规范为准。</p>
</div>
"""'''

TRUST_EXTRA = r'''

def trust_score_chip_html(score: int) -> str:
    tone = "low" if score >= 70 else "mid" if score >= 50 else "high"
    return (
        f'<motion class="sfc-trust-score sfc-trust-score-{tone}">'
        f'<span class="sfc-trust-score-k">综合可信度指数</span>'
        f'<span class="sfc-trust-score-v">{int(score)}</span>'
        f'<span class="sfc-trust-score-h">/ 100 · 公式锚点 + 离线稳定性 + 数据治理</span>'
        f"</div>"
    )


def trust_pipeline_methodology_html(
    pipelines: list[dict[str, str]],
) -> str:
    """各预测路径的方法论与证据来源（公式 / 公式+残差 / ML）。"""
    rows_html = []
    for p in pipelines:
        tone = p.get("tone", "muted")
        cls = f"sfc-check-v sfc-check-v-{tone}"
        note = p.get("note") or ""
        note_html = (
            f'<p class="sfc-trust-pipe-note">{_esc_html(note)}</p>' if note else ""
        )
        rows_html.append(
            f'<div class="sfc-trust-pipe-row">'
            f'<p class="sfc-trust-pipe-task">{_esc_html(p.get("task", ""))}</p>'
            f'<p class="sfc-trust-pipe-method">{_esc_html(p.get("method", ""))}</p>'
            f'<p class="sfc-trust-pipe-evidence">{_esc_html(p.get("evidence", ""))}</p>'
            f'<span class="{cls}">{_esc_html(p.get("stability", ""))}</span>'
            f"{note_html}"
            f"</div>"
        )
    body = "".join(rows_html)
    return f"""
<div class="sfc-check-panel sfc-trust-pipeline-panel">
  <div class="sfc-check-panel-title">分路径方法论（世界数据库 + 公式/残差对照）</div>
  <p class="sfc-trust-pipe-lead">有国标或物理公式的路径优先采用公式基线；残差学习仅在 OOF 优于公式时推荐。</p>
  {body}
</div>
"""
'''

TRUST_EXTRA = TRUST_EXTRA.replace("<motion ", "<div ").replace("</motion>", "</div>")

CSS = """
    .sfc-trust-score {
        display: flex;
        align-items: baseline;
        flex-wrap: wrap;
        gap: 0.35rem 0.75rem;
        margin: 0 0 1rem 0;
        padding: 0.65rem 0.85rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
    }
    .sfc-trust-score-v {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
    }
    .sfc-trust-score-k, .sfc-trust-score-h {
        font-size: 0.82rem;
        color: #64748b;
    }
    .sfc-trust-pipeline-panel { margin-top: 0.25rem; }
    .sfc-trust-pipe-lead {
        font-size: 0.84rem;
        color: #64748b;
        margin: 0 0 0.65rem 0;
        line-height: 1.45;
    }
    .sfc-trust-pipe-row {
        padding: 0.55rem 0;
        border-top: 1px solid #eef2f7;
    }
    .sfc-trust-pipe-row:first-of-type { border-top: none; }
    .sfc-trust-pipe-task {
        margin: 0;
        font-weight: 700;
        font-size: 0.88rem;
        color: #1a5f7a;
    }
    .sfc-trust-pipe-method, .sfc-trust-pipe-evidence {
        margin: 0.15rem 0 0 0;
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.4;
    }
    .sfc-trust-pipe-note {
        margin: 0.25rem 0 0 0;
        font-size: 0.78rem;
        color: #94a3b8;
    }
"""


def _replace_function(src: str, name: str, body: str) -> str:
    start = src.find(f"def {name}(")
    if start == -1:
        raise SystemExit(f"{name} not found")
    nxt = src.find("\ndef ", start + 4)
    if nxt == -1:
        nxt = len(src)
    return src[:start] + body + src[nxt:]


text = _replace_function(text, "trust_conclusion_banner_html", TRUST_BANNER)

if "def trust_score_chip_html(" in text:
    start = text.find("def trust_score_chip_html(")
    end = text.find("\ndef input_range_check_panel_html(", start)
    text = text[:start] + TRUST_EXTRA.strip() + "\n\n" + text[end + 1 :]
else:
    pat = r'(def trust_conclusion_banner_html\([^)]*\)[\s\S]*?""")\s*(?=\ndef input_range_check_panel_html\()'
    m = re.search(pat, text)
    if not m:
        raise SystemExit("trust→input_range anchor not found")
    insert_at = m.end(1)
    text = text[:insert_at] + "\n\n" + TRUST_EXTRA.strip() + "\n\n" + text[insert_at:].lstrip("\n")

if ".sfc-trust-score" not in text:
    text = text.replace("</style>", CSS + "\n</style>", 1)

P.write_text(text, encoding="utf-8", newline="\n")
compile(text, str(P), "exec")
print("ok")
