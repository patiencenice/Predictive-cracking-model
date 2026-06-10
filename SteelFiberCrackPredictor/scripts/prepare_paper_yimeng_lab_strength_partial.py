"""入口：易梦表部分列推导 + 缺口 JSON + gate（不训练）。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.lab_strength_residual.yimeng_partial_prepare import run_cli

if __name__ == "__main__":
    run_cli()
