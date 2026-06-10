"""CLI 入口：调用 src.lab_strength_residual.paper_yimeng_prepare_review.run_cli。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.lab_strength_residual.paper_yimeng_prepare_review import run_cli

if __name__ == "__main__":
    run_cli()
