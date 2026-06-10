from __future__ import annotations

from typing import Any

import pandas as pd


def _cfg_block(config: dict[str, Any], key: str) -> dict[str, Any]:
    v = config.get(key) or {}
    if not isinstance(v, dict):
        raise ValueError(f"{key} must be a mapping in config.")
    return v


def _is_x_only_mode(config: dict[str, Any]) -> bool:
    gen = config.get("generation") or {}
    return str(gen.get("mode") or "").strip().lower() == "x_only"


def check_required_columns(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    pf = _cfg_block(config, "protocol_filter")
    required = pf.get("required_columns") or []
    if _is_x_only_mode(config):
        required = [c for c in required if c not in ("compressive_true", "flexural_true")]
    missing = [c for c in required if c not in df.columns]
    return [f"missing_required_column:{c}" for c in missing]


def check_non_null_columns(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    pf = _cfg_block(config, "protocol_filter")
    cols = pf.get("non_null_columns") or []
    if _is_x_only_mode(config):
        cols = [c for c in cols if c not in ("compressive_true", "flexural_true")]
    issues: list[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        n_bad = int(df[c].isna().sum())
        if n_bad > 0:
            issues.append(f"column_has_null:{c}:n={n_bad}")
    return issues


def check_enum_rules(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    pf = _cfg_block(config, "protocol_filter")
    enum_rules = pf.get("enum_rules") or {}
    issues: list[str] = []
    for col, allowed in enum_rules.items():
        if col not in df.columns:
            issues.append(f"missing_enum_column:{col}")
            continue
        allow = {str(x).strip() for x in (allowed or [])}
        vals = df[col].astype(str).str.strip()
        bad = sorted({x for x in vals.unique().tolist() if x not in allow})
        if bad:
            issues.append(f"enum_invalid:{col}:{bad}")
    return issues


def check_main_eval_columns_parseable(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    me = _cfg_block(config, "main_eval_set")
    issues: list[str] = []
    for col in ("eligible_for_main_oof", "needs_manual_review"):
        if col not in df.columns:
            issues.append(f"missing_main_eval_column:{col}")
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        bad = int(s.isna().sum())
        if bad > 0:
            issues.append(f"non_numeric_main_eval_column:{col}:n={bad}")
    for col in ("source_domain", "data_tier"):
        if col not in df.columns:
            issues.append(f"missing_main_eval_column:{col}")
    # 读取配置仅为锁定字段存在性；具体筛选由后续 runner 统一执行
    _ = me
    return issues


def run_protocol_checks(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """
    协议闸门仅检查，不改值：
    - 必要列
    - 关键列缺失
    - 枚举合法性
    - 主评估规则字段可解析性
    """
    groups = {
        "required_columns": check_required_columns(df, config),
        "non_null_columns": check_non_null_columns(df, config),
        "enum_rules": check_enum_rules(df, config),
        "main_eval_parseable": check_main_eval_columns_parseable(df, config),
    }
    all_issues: list[str] = []
    for _, items in groups.items():
        all_issues.extend(items)
    return {
        "ok": len(all_issues) == 0,
        "issues": all_issues,
        "issue_groups": groups,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
    }
