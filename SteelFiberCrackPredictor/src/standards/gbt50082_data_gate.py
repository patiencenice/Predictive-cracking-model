"""
基于国标骨架的 **数据闸门**（协议检查），与主训练入口解耦。

- 不修改 train_model / predictor / features。
- 当前仅检查：standard_method_id 是否注册、关键协议列是否存在（若 schema 要求）。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import yaml

from src.paths import PROJECT_ROOT
from src.standards.gbt50082_rules import get_method_spec, list_method_ids

_SCHEMA_PATH = PROJECT_ROOT / "config" / "crack_protocol_schema.yaml"


def _load_schema() -> dict[str, Any]:
    if not _SCHEMA_PATH.is_file():
        return {}
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def gate_row_protocol(
    row: pd.Series,
    *,
    strict_method: bool = False,
) -> list[str]:
    """
    返回问题列表；空列表表示在「当前骨架规则下」未发现硬冲突。

    strict_method=True 时：若缺少 standard_method_id 列或值不在注册表，记一条问题。
    """
    problems: list[str] = []
    # 预留：读取 _load_schema() 按 crack_protocol_schema 做列级检查

    if strict_method:
        mid = row.get("standard_method_id") if "standard_method_id" in row.index else None
        if mid is None or (isinstance(mid, float) and pd.isna(mid)) or str(mid).strip() == "":
            problems.append("缺少 standard_method_id，无法对齐 GB/T 50082 方法表。")
        elif str(mid) not in list_method_ids():
            problems.append(
                f"standard_method_id={mid!r} 未在 config/gbt50082_methods.yaml 注册。"
            )
        else:
            spec = get_method_spec(str(mid))
            if spec is None:
                problems.append(f"方法表缺失条目: {mid!r}")
            elif spec.get("结果计算公式") is None:
                problems.append(
                    f"方法 {mid} 的「结果计算公式」仍为 null — 条文未补全前，禁止用作数值基线。"
                )
    return problems
