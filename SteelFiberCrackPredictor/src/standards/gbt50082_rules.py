"""
GB/T 50082-2024 试验方法 — 结构化规则加载与策略常量。

- 不实现具体公式数值（须由 config/gbt50082_methods.yaml 人工补全后接入）。
- 不提供“把公式当标签”的接口。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.paths import PROJECT_ROOT

_CONFIG_PATH = PROJECT_ROOT / "config" / "gbt50082_methods.yaml"


class GBT50082ConfigError(FileNotFoundError, ValueError):
    pass


@lru_cache(maxsize=1)
def load_methods_config() -> dict[str, Any]:
    if not _CONFIG_PATH.is_file():
        raise GBT50082ConfigError(f"缺少配置文件: {_CONFIG_PATH}")
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise GBT50082ConfigError("gbt50082_methods.yaml 根节点须为 mapping")
    return data


def list_method_ids() -> tuple[str, ...]:
    cfg = load_methods_config()
    methods = cfg.get("methods")
    if not isinstance(methods, dict):
        return ()
    return tuple(sorted(methods.keys()))


def get_method_spec(method_id: str) -> dict[str, Any] | None:
    cfg = load_methods_config()
    methods = cfg.get("methods")
    if not isinstance(methods, dict):
        return None
    m = methods.get(method_id)
    return dict(m) if isinstance(m, dict) else None


def formula_usage_policy() -> dict[str, Any]:
    cfg = load_methods_config()
    pol = cfg.get("formula_usage_policy")
    return dict(pol) if isinstance(pol, dict) else {}


def is_formula_role_allowed(role: str) -> bool:
    pol = formula_usage_policy()
    allowed = pol.get("allowed_roles")
    if not isinstance(allowed, list):
        return False
    return role in allowed
