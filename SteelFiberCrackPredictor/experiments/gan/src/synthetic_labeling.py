from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def apply_fixed_synthetic_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一写入 synthetic 固定四元组（研究链路强约束）。
    """
    out = df.copy()
    out["data_tier"] = "C"
    out["extraction_method"] = "synthetic_gan"
    out["eligible_for_main_oof"] = 0
    out["eligible_for_aux_training"] = 1
    if "needs_manual_review" not in out.columns:
        out["needs_manual_review"] = 0
    if "source_group" not in out.columns:
        out["source_group"] = "GAN_SYNTH"
    else:
        s = out["source_group"].astype(str).str.strip()
        out["source_group"] = s.mask(s == "", "GAN_SYNTH")
    out["synthetic_generated_at_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return out


def validate_fixed_synthetic_labels(df: pd.DataFrame) -> dict[str, object]:
    required = [
        "data_tier",
        "extraction_method",
        "eligible_for_main_oof",
        "eligible_for_aux_training",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"ok": False, "issues": [f"missing:{c}" for c in missing]}

    issues: list[str] = []
    if not (df["data_tier"].astype(str) == "C").all():
        issues.append("data_tier_not_all_C")
    if not (df["extraction_method"].astype(str) == "synthetic_gan").all():
        issues.append("extraction_method_not_all_synthetic_gan")
    if not (pd.to_numeric(df["eligible_for_main_oof"], errors="coerce") == 0).all():
        issues.append("eligible_for_main_oof_not_all_0")
    if not (
        pd.to_numeric(df["eligible_for_aux_training"], errors="coerce") == 1
    ).all():
        issues.append("eligible_for_aux_training_not_all_1")

    return {"ok": len(issues) == 0, "issues": issues}
