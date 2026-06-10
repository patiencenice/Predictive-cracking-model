from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")
    return raw


def resolve_input_csv(config: dict[str, Any], project_root: str | Path) -> Path:
    data_cfg = config.get("data") or {}
    rel = data_cfg.get("input_csv")
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("config.data.input_csv is required.")
    p = Path(project_root) / rel
    if not p.is_file():
        raise FileNotFoundError(f"Input CSV not found: {p}")
    return p


def load_real_dataframe(config: dict[str, Any], project_root: str | Path) -> pd.DataFrame:
    """
    只读取现有真实 CSV，不做任何伪造与补写。
    """
    csv_path = resolve_input_csv(config, project_root)
    encoding = str((config.get("data") or {}).get("encoding") or "utf-8-sig")
    df = pd.read_csv(csv_path, encoding=encoding)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {csv_path}")
    return df


def split_by_source_domain(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = config.get("data") or {}
    col = str(data_cfg.get("source_domain_column") or "source_domain")
    allowed = data_cfg.get("source_domain_allowed") or ["ordinary", "fiber"]
    allowed_set = {str(x).strip() for x in allowed}
    if col not in df.columns:
        raise ValueError(f"Missing source domain column: {col}")

    dom = df[col].astype(str).str.strip()
    bad = sorted({x for x in dom.unique().tolist() if x not in allowed_set})
    if bad:
        raise ValueError(f"Unexpected source_domain values: {bad}")

    ordinary = df[dom == "ordinary"].copy()
    fiber = df[dom == "fiber"].copy()
    return ordinary, fiber


def select_tiers_for_gan_training(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config.get("data") or {}
    col = str(data_cfg.get("data_tier_column") or "data_tier")
    if col not in df.columns:
        raise ValueError(f"Missing tier column: {col}")

    tier_cfg = data_cfg.get("tier_train_default") or {}
    include_a = bool(tier_cfg.get("include_A", True))
    include_b = bool(tier_cfg.get("include_B", False))
    include_c = bool(tier_cfg.get("include_C", False))
    allow = []
    if include_a:
        allow.append("A")
    if include_b:
        allow.append("B")
    if include_c:
        allow.append("C")

    # V1 硬约束：Tier C 不允许进入 GAN 训练
    if "C" in allow:
        raise ValueError("Tier C is forbidden for GAN training in V1.")
    if not allow:
        raise ValueError("At least one of Tier A/B must be enabled.")

    tiers = df[col].astype(str).str.strip()
    return df[tiers.isin(allow)].copy()
