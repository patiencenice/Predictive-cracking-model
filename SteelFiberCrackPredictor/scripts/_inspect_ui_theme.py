from pathlib import Path

p = Path("src/ui_theme.py")
raw = p.read_bytes()
print("bytes", len(raw))
print("newlines", raw.count(b"\n"))
print("triple_newline_runs", raw.count(b"\n\n\n"))
idx = raw.find(b"def hero_banner_html")
print("hero byte idx", idx)
i = raw.find(b'return f"""', idx)
print("return snippet:", repr(raw[i : i + 150]))

from src.ui_theme import hero_banner_html

s = hero_banner_html("t", "sub", risk_level="高风险")
print("output newlines per line break:", s.count("\n\n"))
