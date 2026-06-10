from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .load_real_data import (
    load_experiment_config,
    load_real_dataframe,
    select_tiers_for_gan_training,
)
from .protocol_filter import run_protocol_checks


DEFAULT_CONDITION_COLUMNS = [
    "source_domain",
    "strength_grade_enc",
    "concrete_type_enc",
    "fiber_type_enc",
    "curing_days",
    "test_method",
]


@dataclass
class TrainedCtganBundle:
    model: Any
    backend: str
    train_df: pd.DataFrame
    train_columns: list[str]
    discrete_columns: list[str]
    training_rows: int
    condition_columns: list[str]


def _infer_discrete_columns(df: pd.DataFrame) -> list[str]:
    disc: list[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_bool_dtype(s):
            disc.append(c)
            continue
        if c.endswith("_enc") or c in ("source_domain", "data_tier", "test_method"):
            disc.append(c)
    return sorted(set(disc))


def _build_train_table(df: pd.DataFrame) -> pd.DataFrame:
    # source_group 仅作为分组/防泄漏键，明确不进 GAN 训练表
    drop_cols = {"source_group", "extraction_method", "eligible_for_aux_training"}
    # X-only: 不再生成 compressive_true / flexural_true
    preferred = [
        "source_domain",
        "strength_grade_enc",
        "concrete_type_enc",
        "fiber_type_enc",
        "curing_days",
        "fiber_content",
        "aspect_ratio",
        "w_b_ratio",
        "binder_content",
        "cement_content",
        "mixing_water",
        "fly_ash",
        "slag_powder",
        "data_tier",
    ]
    cols = [c for c in preferred if c in df.columns and c not in drop_cols]
    out = df[cols].copy()
    need = ["source_domain", "data_tier"]
    missing = [c for c in need if c not in out.columns]
    if missing:
        raise ValueError(f"missing required X-only fields for GAN training: {missing}")
    out = out.dropna(subset=need, how="any")
    return out


def _normalize_admin_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    仅补齐研究链路管理列（非标签、非物理特征），用于协议闭合检查。
    不修改 compressive_true / flexural_true 或配合比数值。
    """
    out = df.copy()
    me = config.get("main_eval_set") or {}
    default_domain = str(me.get("source_domain") or "fiber")
    default_tier = str(me.get("data_tier") or "A")
    if "source_domain" not in out.columns:
        out["source_domain"] = default_domain
    else:
        s = out["source_domain"].astype(str).str.strip()
        out["source_domain"] = s.mask(s == "", default_domain)
    if "eligible_for_main_oof" not in out.columns:
        out["eligible_for_main_oof"] = int(me.get("eligible_for_main_oof", 1))
    if "needs_manual_review" not in out.columns:
        out["needs_manual_review"] = 0
    if "data_tier" not in out.columns:
        out["data_tier"] = default_tier
    else:
        s = out["data_tier"].astype(str).str.strip()
        out["data_tier"] = s.mask((s == "") | (s.str.lower() == "nan"), default_tier)
    return out


def train_ctgan_model(
    config_path: str,
    project_root: str,
    *,
    random_state: int = 42,
    epochs: int = 200,
    batch_size: int = 128,
) -> TrainedCtganBundle:
    config = load_experiment_config(config_path)
    real_df = load_real_dataframe(config, project_root)
    real_df = _normalize_admin_columns(real_df, config)

    chk = run_protocol_checks(real_df, config)
    if not chk["ok"]:
        raise ValueError(f"protocol check failed on real data: {chk['issues']}")

    train_df = select_tiers_for_gan_training(real_df, config)
    train_df = _build_train_table(train_df).reset_index(drop=True)
    if len(train_df) < 20:
        raise ValueError(f"too few rows for GAN training: {len(train_df)}")

    discrete_cols = [c for c in _infer_discrete_columns(train_df) if c in train_df.columns]
    cond_cols = [c for c in DEFAULT_CONDITION_COLUMNS if c in train_df.columns and c != "source_group"]

    model = None
    backend = ""
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=train_df)
        model = CTGANSynthesizer(
            metadata,
            enforce_min_max_values=True,
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=False,
        )
        model.fit(train_df)
        backend = "sdv"
    except Exception:
        try:
            from ctgan import CTGAN

            model = CTGAN(
                epochs=int(epochs),
                batch_size=int(batch_size),
                pac=1,
                verbose=False,
            )
            model.fit(train_df, discrete_columns=discrete_cols)
            backend = "ctgan"
        except Exception as ex:  # noqa: BLE001
            raise RuntimeError(
                f"CTGAN training failed on both backends. Last error: {type(ex).__name__}: {ex}"
            ) from ex

    return TrainedCtganBundle(
        model=model,
        backend=backend,
        train_df=train_df,
        train_columns=list(train_df.columns),
        discrete_columns=discrete_cols,
        training_rows=int(len(train_df)),
        condition_columns=cond_cols,
    )
