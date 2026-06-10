"""C30 试验约束档位 → 公式链 restraint_factor_R。"""

from __future__ import annotations


def restraint_factor_from_percent(restraint_percent: float | int | None) -> float | None:
    """
    C30 仪器试验：R0/R50/R100 对应 0/50/100 % 约束 → R ∈ [0, 1]。
    与 Phase1 restraint_level 枚举（low/medium/high）不同，专用于 C30 残差管线。
    """
    if restraint_percent is None:
        return None
    try:
        p = float(restraint_percent)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= p <= 100.0):
        return None
    return p / 100.0
