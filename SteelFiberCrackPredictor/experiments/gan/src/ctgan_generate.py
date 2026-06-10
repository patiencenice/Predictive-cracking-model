from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ctgan_train import TrainedCtganBundle, train_ctgan_model
from .load_real_data import load_experiment_config
from .memorization_check import run_memorization_check
from .physical_gate import run_physical_gate
from .protocol_filter import run_protocol_checks
from .synthetic_labeling import apply_fixed_synthetic_labels, validate_fixed_synthetic_labels


def _sample_once(bundle: TrainedCtganBundle, n_rows: int) -> pd.DataFrame:
    if bundle.backend == "sdv":
        # sdv: sample 按整表元数据返回
        return bundle.model.sample(num_rows=int(n_rows))
    if bundle.backend == "ctgan":
        return bundle.model.sample(int(n_rows))
    raise ValueError(f"unknown backend: {bundle.backend}")


def _simple_condition_combos(bundle: TrainedCtganBundle, config: dict[str, Any]) -> list[dict[str, Any]]:
    gen_cfg = config.get("generation") or {}
    active = [c for c in (gen_cfg.get("condition_columns_active") or []) if c in bundle.train_df.columns and c != "source_group"]
    if not active:
        return []
    max_combos = int(gen_cfg.get("max_condition_combos", 4))
    base = bundle.train_df[active].dropna().copy()
    if base.empty:
        return []
    combos = base.drop_duplicates().head(max_combos)
    out: list[dict[str, Any]] = []
    for _, row in combos.iterrows():
        out.append({k: row[k] for k in active})
    return out


def _sample_with_conditions(bundle: TrainedCtganBundle, config: dict[str, Any], n_rows: int) -> pd.DataFrame:
    combos = _simple_condition_combos(bundle, config)
    if not combos:
        return _sample_once(bundle, n_rows)
    gen_cfg = config.get("generation") or {}
    batch_size = int(gen_cfg.get("condition_sample_batch_size", max(64, n_rows)))
    max_iters = int(gen_cfg.get("condition_max_iters_per_combo", 12))
    n_combo = len(combos)
    per = max(1, n_rows // n_combo)
    remain = n_rows - per * n_combo
    parts: list[pd.DataFrame] = []
    rng = np.random.default_rng(42)

    for i, cond in enumerate(combos):
        target = per + (1 if i < remain else 0)
        got = pd.DataFrame(columns=bundle.train_columns)
        for _ in range(max_iters):
            need = target - len(got)
            if need <= 0:
                break
            cand = _sample_once(bundle, max(batch_size, need))
            m = pd.Series(True, index=cand.index)
            for k, v in cond.items():
                if k not in cand.columns:
                    m = m & False
                    continue
                # 条件列当前均按离散/编码匹配，避免 source_group
                m = m & (cand[k].astype(str) == str(v))
            pick = cand[m]
            if not pick.empty:
                got = pd.concat([got, pick], axis=0, ignore_index=True)
        if len(got) < target:
            # 回填：尽量贴近条件（只用于凑足条数，仍受后续闸门）
            fill = _sample_once(bundle, target - len(got))
            for k, v in cond.items():
                if k in fill.columns:
                    fill[k] = v
            got = pd.concat([got, fill], axis=0, ignore_index=True)
        parts.append(got.head(target))

    out = pd.concat(parts, axis=0, ignore_index=True)
    if len(out) > n_rows:
        out = out.sample(n=n_rows, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    return out


def _apply_post_clip(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    gen_cfg = config.get("generation") or {}
    clip_cfg = (gen_cfg.get("post_clip") or {})
    out = df.copy()
    for col, rg in clip_cfg.items():
        if col not in out.columns:
            continue
        lo = rg.get("min")
        hi = rg.get("max")
        s = pd.to_numeric(out[col], errors="coerce")
        if lo is not None and hi is not None:
            out[col] = s.clip(lower=float(lo), upper=float(hi))
        elif lo is not None:
            out[col] = s.clip(lower=float(lo))
        elif hi is not None:
            out[col] = s.clip(upper=float(hi))
    return out


def _fill_non_negative_missing(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    研究链路保守补齐：对物理闸门要求的非负列，缺失时填 0。
    不改变已有非空值。
    """
    out = df.copy()
    gate_cfg = config.get("physical_gate") or {}
    cols = list(gate_cfg.get("non_negative_columns") or [])
    for c in cols:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            out[c] = s.fillna(0.0)
    return out


def _col_stats(df: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in df.columns:
        return {"present": False}
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    if v.empty:
        return {"present": True, "n": 0}
    return {
        "present": True,
        "n": int(v.shape[0]),
        "min": float(v.min()),
        "q05": float(v.quantile(0.05)),
        "q50": float(v.quantile(0.50)),
        "q95": float(v.quantile(0.95)),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.shape[0] > 1 else 0.0,
    }


def _distribution_shift_summary(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> dict[str, Any]:
    cols = [
        "w_b_ratio",
        "fiber_content",
        "aspect_ratio",
        "binder_content",
        "cement_content",
        "mixing_water",
        "fly_ash",
        "slag_powder",
    ]
    out: dict[str, Any] = {}
    scores: list[tuple[str, float]] = []
    for c in cols:
        r = _col_stats(real_df, c)
        s = _col_stats(syn_df, c)
        out[c] = {"real": r, "synthetic": s}
        if r.get("present") and s.get("present") and r.get("n", 0) > 0 and s.get("n", 0) > 0:
            rstd = max(float(r.get("std", 0.0)), 1e-9)
            mean_shift = abs(float(s["mean"]) - float(r["mean"])) / rstd
            std_ratio = abs(float(s.get("std", 0.0)) - float(r.get("std", 0.0))) / rstd
            scores.append((c, float(mean_shift + std_ratio)))
    scores.sort(key=lambda x: x[1], reverse=True)
    top5 = [{"column": c, "shift_score": sc} for c, sc in scores[:5]]
    if top5 and top5[0]["shift_score"] >= 2.5:
        risk = "high"
    elif top5 and top5[0]["shift_score"] >= 1.0:
        risk = "medium"
    else:
        risk = "low"
    return {
        "columns": out,
        "top_shifted_columns": top5,
        "distribution_shift_risk": risk,
    }


def _label_feature_coupling_check(
    real_df: pd.DataFrame, syn_df: pd.DataFrame, condition_cols: list[str]
) -> dict[str, Any]:
    if "compressive_true" not in syn_df.columns or "flexural_true" not in syn_df.columns:
        return {
            "syn_flexural_ge_compressive_count": 0,
            "syn_flexural_ge_compressive_ratio": 0.0,
            "condition_columns_checked": [],
            "condition_label_mean_gap": [],
            "avg_normalized_gap_weighted_by_syn_count": 0.0,
            "label_feature_coupling_risk": "low",
            "note": "x_only mode: synthetic does not generate y labels.",
        }

    rc = pd.to_numeric(real_df.get("compressive_true"), errors="coerce")
    rf = pd.to_numeric(real_df.get("flexural_true"), errors="coerce")
    sc = pd.to_numeric(syn_df.get("compressive_true"), errors="coerce")
    sf = pd.to_numeric(syn_df.get("flexural_true"), errors="coerce")

    syn_bad_relation = int(((sf >= sc) & sf.notna() & sc.notna()).sum())
    syn_bad_ratio = float(syn_bad_relation / max(1, int(sf.notna().sum())))

    active = [c for c in condition_cols if c in real_df.columns and c in syn_df.columns]
    combo_rows: list[dict[str, Any]] = []
    weighted_gap = 0.0
    weighted_n = 0
    if active:
        rg = (
            real_df.groupby(active)[["compressive_true", "flexural_true"]]
            .mean(numeric_only=True)
            .rename(columns={"compressive_true": "real_comp_mean", "flexural_true": "real_flex_mean"})
        )
        sg = (
            syn_df.groupby(active)[["compressive_true", "flexural_true"]]
            .mean(numeric_only=True)
            .rename(columns={"compressive_true": "syn_comp_mean", "flexural_true": "syn_flex_mean"})
        )
        both = rg.join(sg, how="inner")
        base_comp_std = max(float(rc.std(ddof=1)) if rc.notna().sum() > 1 else 0.0, 1e-9)
        base_flex_std = max(float(rf.std(ddof=1)) if rf.notna().sum() > 1 else 0.0, 1e-9)
        for idx, row in both.iterrows():
            key = idx if isinstance(idx, tuple) else (idx,)
            cond = {active[i]: key[i] for i in range(len(active))}
            n_syn = int(
                syn_df.merge(pd.DataFrame([cond]), on=active, how="inner").shape[0]
            )
            comp_gap = abs(float(row["syn_comp_mean"]) - float(row["real_comp_mean"])) / base_comp_std
            flex_gap = abs(float(row["syn_flex_mean"]) - float(row["real_flex_mean"])) / base_flex_std
            gap = float(comp_gap + flex_gap)
            combo_rows.append(
                {
                    "condition": cond,
                    "n_synthetic": n_syn,
                    "real_comp_mean": float(row["real_comp_mean"]),
                    "syn_comp_mean": float(row["syn_comp_mean"]),
                    "real_flex_mean": float(row["real_flex_mean"]),
                    "syn_flex_mean": float(row["syn_flex_mean"]),
                    "normalized_gap": gap,
                }
            )
            weighted_gap += gap * max(n_syn, 1)
            weighted_n += max(n_syn, 1)
    avg_gap = float(weighted_gap / weighted_n) if weighted_n > 0 else 0.0

    if syn_bad_ratio > 0.02 or avg_gap >= 2.0:
        risk = "high"
    elif syn_bad_ratio > 0.0 or avg_gap >= 1.0:
        risk = "medium"
    else:
        risk = "low"

    return {
        "syn_flexural_ge_compressive_count": syn_bad_relation,
        "syn_flexural_ge_compressive_ratio": syn_bad_ratio,
        "condition_columns_checked": active,
        "condition_label_mean_gap": sorted(
            combo_rows, key=lambda x: x["normalized_gap"], reverse=True
        )[:20],
        "avg_normalized_gap_weighted_by_syn_count": avg_gap,
        "label_feature_coupling_risk": risk,
    }


def _quality_check_summary(
    bundle: TrainedCtganBundle, syn_filtered: pd.DataFrame
) -> dict[str, Any]:
    mem = run_memorization_check(bundle.train_df, syn_filtered)
    dist = _distribution_shift_summary(bundle.train_df, syn_filtered)
    cond_cols = [c for c in bundle.condition_columns if c != "source_group"]
    lfc = _label_feature_coupling_check(bundle.train_df, syn_filtered, cond_cols)
    risks = {
        "memorization_risk": mem["memorization_risk"],
        "distribution_shift_risk": dist["distribution_shift_risk"],
        "label_feature_coupling_risk": lfc["label_feature_coupling_risk"],
    }
    risk_rank = {"low": 0, "medium": 1, "high": 2}
    max_risk = max(risk_rank[risks[k]] for k in risks)
    recommend_stage3 = "yes" if max_risk <= 1 else "no"
    if recommend_stage3 == "no":
        advice = (
            "先收紧条件采样与列范围，并优先考虑只生成 X 不生成 y；完成记忆/分布复核后再进第三阶段。"
        )
    else:
        advice = "可谨慎进入第三阶段，但主指标仍必须只在真实 A 类纤维样本上计算。"
    return {
        "memorization": mem,
        "distribution_shift": dist,
        "label_feature_coupling": lfc,
        "summary": {
            **risks,
            "recommend_enter_stage3": recommend_stage3,
            "minimal_next_step_zh": advice,
        },
    }


def _coerce_to_train_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols].copy()


def generate_synthetic_dataset(
    config_path: str,
    project_root: str,
    *,
    n_rows: int = 500,
    out_dir: str | None = None,
    random_state: int = 42,
    epochs: int = 200,
    batch_size: int = 128,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    bundle = train_ctgan_model(
        config_path=config_path,
        project_root=project_root,
        random_state=random_state,
        epochs=epochs,
        batch_size=batch_size,
    )

    raw = _sample_with_conditions(bundle, config, n_rows=int(n_rows))
    raw = _coerce_to_train_columns(raw, bundle.train_columns)
    raw = _apply_post_clip(raw, config)
    raw = _fill_non_negative_missing(raw, config)
    raw = apply_fixed_synthetic_labels(raw)

    prot_raw = run_protocol_checks(raw, config)
    phys_raw = run_physical_gate(raw, config)
    pass_phys = phys_raw["pass_mask"]
    filt = raw[pass_phys].copy()

    prot_filt = run_protocol_checks(filt, config)
    label_chk = validate_fixed_synthetic_labels(filt)
    qc = _quality_check_summary(bundle, filt) if len(filt) > 0 else {
        "summary": {
            "memorization_risk": "high",
            "distribution_shift_risk": "high",
            "label_feature_coupling_risk": "high",
            "recommend_enter_stage3": "no",
            "minimal_next_step_zh": "当前 filtered 样本为空，先修复生成质量后再评估。",
        }
    }

    if out_dir is None:
        out_path = Path(project_root) / "experiments" / "gan" / "outputs"
    else:
        out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_csv = out_path / "synthetic_raw.csv"
    filt_csv = out_path / "synthetic_filtered.csv"
    summary_json = out_path / "synthetic_filter_summary.json"
    quality_json = out_path / "synthetic_quality_summary.json"
    raw.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    filt.to_csv(filt_csv, index=False, encoding="utf-8-sig")

    summary = {
        "config_path": str(Path(config_path).resolve()),
        "backend": bundle.backend,
        "training_rows_real_protocol_closed_tier_selected": bundle.training_rows,
        "condition_columns_used": bundle.condition_columns,
        "condition_combos_used": _simple_condition_combos(bundle, config),
        "forbidden_condition_columns": ["source_group"],
        "n_rows_synthetic_raw": int(len(raw)),
        "n_rows_after_physical_gate": int(len(filt)),
        "n_rows_dropped_by_physical_gate": int(len(raw) - len(filt)),
        "protocol_check_raw_ok": bool(prot_raw["ok"]),
        "protocol_check_filtered_ok": bool(prot_filt["ok"]),
        "protocol_issues_raw": prot_raw["issues"],
        "protocol_issues_filtered": prot_filt["issues"],
        "physical_failed_reason_counts": phys_raw["failed_reason_counts"],
        "fixed_quadruplet_check": label_chk,
        "quality_check_summary": qc.get("summary", {}),
        "outputs": {
            "synthetic_raw_csv": str(raw_csv.resolve()),
            "synthetic_filtered_csv": str(filt_csv.resolve()),
            "summary_json": str(summary_json.resolve()),
            "quality_summary_json": str(quality_json.resolve()),
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(quality_json, "w", encoding="utf-8") as f:
        json.dump(qc, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CTGAN minimal generation pipeline (phase 2)")
    parser.add_argument("--config", required=True, help="Path to gan_experiment.yaml")
    parser.add_argument("--project-root", required=True, help="Project root path")
    parser.add_argument("--n-rows", type=int, default=500, help="Synthetic rows to sample")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rep = generate_synthetic_dataset(
        config_path=args.config,
        project_root=args.project_root,
        n_rows=args.n_rows,
        out_dir=args.out_dir,
        random_state=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))
