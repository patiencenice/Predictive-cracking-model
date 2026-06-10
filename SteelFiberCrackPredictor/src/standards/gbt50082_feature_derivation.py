"""
基于 GB/T 50082-2024 规则框架的 **派生特征** 占位实现。

- 不读取训练标签；不调用主链路 features.py。
- 具体几何/时间归一公式须在 YAML 与条文补全后实现。
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def derive_placeholder_features(
    row: pd.Series,
    *,
    standard_method_id: str | None = None,
) -> dict[str, float]:
    """
    返回空 dict：占位 API，避免在未补全条文前静默产出“伪国标特征”。

    后续实现时，仅允许输出 L2_derived（见 docs/data_contract_crack_gbt50082.md）。
    """
    _ = (row, standard_method_id)
    return {}


def explain_derivation_scope() -> str:
    return (
        "派生特征模块已预留；待 config/gbt50082_methods.yaml 与标准条文补全后，"
        "在此实现量纲一致化、标距归一、观测窗对齐等，且不得写入监督标签。"
    )
