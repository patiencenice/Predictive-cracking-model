# -*- coding: utf-8 -*-
"""模型可信度引擎：读取项目内「世界数据库」产物，按公式/公式+残差/纯 ML 分路径评估。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.paths import MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT


@dataclass
class PipelineTrustRow:
    task: str
    method: str
    evidence: str
    stability: str
    tone: str  # low | mid | high | muted
    note: str = ""


@dataclass
class TrustAssessment:
    level_label: str
    tone: str
    intro: str
    positives: list[str]
    caveats: list[str]
    pipelines: list[PipelineTrustRow] = field(default_factory=list)
    trust_score: int = 50


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


@lru_cache(maxsize=1)
def load_world_trust_bundle() -> dict[str, Any]:
    """聚合训练库、治理、力学/温度残差管线的离线报告。"""
    return {
        "governance": _load_json(
            OUTPUTS_DIR / "crack_governance" / "crack_training_governance.json"
        ),
        "lab_strength": _load_json(
            OUTPUTS_DIR / "lab_strength" / "lab_strength_residual_report.json"
        ),
        "thermal_stress": _load_json(
            OUTPUTS_DIR / "thermal_stress" / "residual_model" / "thermal_stress_residual_report.json"
        ),
        "crack_metrics_models": _load_json(MODELS_DIR / "training_metrics.json"),
        "crack_metrics_outputs": _load_json(OUTPUTS_DIR / "training_metrics.json"),
        "training_rows": _count_training_rows(),
    }


def _count_training_rows() -> int | None:
    csv_path = PROJECT_ROOT / "data" / "training_data.csv"
    if not csv_path.is_file():
        return None
    try:
        import pandas as pd

        return int(len(pd.read_csv(csv_path)))
    except Exception:
        return None


def _reg_stability(block: dict[str, Any] | None) -> tuple[str, str]:
    if not block:
        return "待评估", "muted"
    try:
        r2f = float(block.get("test_r2"))
    except (TypeError, ValueError):
        return "待评估", "muted"
    if not math.isfinite(r2f):
        return "待评估", "muted"
    if r2f >= 0.4:
        return "相对稳定", "low"
    if r2f >= 0.1:
        return "一般", "mid"
    return "波动较大", "high"


def _cls_stability(block: dict[str, Any] | None) -> tuple[str, str]:
    if not block:
        return "待评估", "muted"
    try:
        af = float(block.get("test_accuracy"))
    except (TypeError, ValueError):
        return "待评估", "muted"
    if not math.isfinite(af):
        return "待评估", "muted"
    if af >= 0.65:
        return "相对稳定", "low"
    if af >= 0.45:
        return "一般", "mid"
    return "波动较大", "high"


def _oof_r2(lab_rep: dict[str, Any] | None, task: str) -> float | None:
    if not lab_rep:
        return None
    diag = lab_rep.get("diagnostics") or {}
    ab = diag.get("source_domain_ablation") or {}
    primary = ab.get("primary_append_source_domain_true") or {}
    task_blk = primary.get(task) or {}
    oof = task_blk.get("oof_global_metrics") or {}
    fo = oof.get("formula_only") or {}
    try:
        r2 = float(fo.get("r2"))
        return r2 if math.isfinite(r2) else None
    except (TypeError, ValueError):
        return None


def _lab_method_label(lab_rep: dict[str, Any] | None, task: str) -> str:
    if not lab_rep:
        return "国标公式基线（报告未生成）"
    defaults = lab_rep.get("default_method_by_task") or {}
    blk = defaults.get(task) or {}
    strat = str(blk.get("strategy") or "formula_only")
    if lab_rep.get("residual_not_recommended"):
        best = (
            (lab_rep.get("diagnostics") or {})
            .get("source_domain_ablation", {})
            .get("primary_append_source_domain_true", {})
            .get(task, {})
            .get("effective_best_method_oof_mae")
        )
        if best == "formula_only":
            return "国标公式基线（离线 OOF 最优，残差不推荐）"
    mapping = {
        "formula_only": "国标公式基线",
        "formula_plus_ridge_residual": "国标公式基线 + Ridge 残差",
        "formula_plus_hgb_residual": "国标公式基线 + HGB 残差",
    }
    return mapping.get(strat, strat)


def _thermal_method_label(rep: dict[str, Any] | None) -> str:
    if not rep:
        return "物理公式 σ=R·E·α·ΔT（报告未生成）"
    dm = str(rep.get("default_method") or "formula_only")
    if dm == "formula_only":
        return "物理公式 σ=R·E·α·ΔT"
    return f"物理公式基线 + {dm.upper()} 残差学习"


def build_pipeline_rows(
    stability: list[tuple[str, str, str]],
) -> list[PipelineTrustRow]:
    bundle = load_world_trust_bundle()
    lab = bundle.get("lab_strength")
    thermal = bundle.get("thermal_stress")
    gov = bundle.get("governance")
    stab_map = {r[0]: (r[1], r[2]) for r in stability}

    rows: list[PipelineTrustRow] = []

    w_stab, w_tone = stab_map.get("裂缝宽度预测", ("待评估", "muted"))
    rows.append(
        PipelineTrustRow(
            task="主开裂 · 裂缝宽度",
            method="XGBoost 端到端（无独立公式锚点）",
            evidence="models/training_metrics.json",
            stability=w_stab,
            tone=w_tone,
            note="有公式规范参照（GB 50010 缝宽限值），但预测本身为数据驱动。",
        )
    )

    d_stab, d_tone = stab_map.get("裂缝密度预测", ("待评估", "muted"))
    rows.append(
        PipelineTrustRow(
            task="主开裂 · 裂缝密度",
            method="XGBoost 端到端（无独立公式锚点）",
            evidence="models/training_metrics.json",
            stability=d_stab,
            tone=d_tone,
            note="离线稳定性通常弱于缝宽；宜结合现场监测。",
        )
    )

    r_stab, r_tone = stab_map.get("开裂风险分类", ("待评估", "muted"))
    rows.append(
        PipelineTrustRow(
            task="主开裂 · 风险等级",
            method="XGBoost 三分类（无独立公式锚点）",
            evidence="models/training_metrics.json",
            stability=r_stab,
            tone=r_tone,
        )
    )

    comp_r2 = _oof_r2(lab, "compressive")
    comp_note = ""
    if comp_r2 is not None:
        comp_note = f"公式基线 OOF R²≈{comp_r2:.2f}"
    rows.append(
        PipelineTrustRow(
            task="力学 · 抗压",
            method=_lab_method_label(lab, "compressive"),
            evidence="outputs/lab_strength/lab_strength_residual_report.json",
            stability="公式链可追溯",
            tone="low" if comp_r2 is None or comp_r2 >= 0.85 else "mid",
            note=comp_note,
        )
    )

    flex_r2 = _oof_r2(lab, "flexural")
    flex_note = ""
    if flex_r2 is not None:
        flex_note = f"公式基线 OOF R²≈{flex_r2:.2f}"
    rows.append(
        PipelineTrustRow(
            task="力学 · 抗折",
            method=_lab_method_label(lab, "flexural"),
            evidence="outputs/lab_strength/lab_strength_residual_report.json",
            stability="公式链可追溯",
            tone="low" if flex_r2 is None or flex_r2 >= 0.3 else "mid",
            note=flex_note,
        )
    )

    rows.append(
        PipelineTrustRow(
            task="机理 · 温度应力",
            method=_thermal_method_label(thermal),
            evidence="outputs/thermal_stress/residual_model/thermal_stress_residual_report.json",
            stability="解释层（不进主模型 FEATURE_COLUMNS）",
            tone="muted",
            note="C30 世界数据库时序已接入残差训练；用于机理解释与可信度交叉核对。",
        )
    )

    if gov:
        tc = gov.get("tier_ABC_hold_counts") or {}
        tier_a = int(tc.get("tier_A_candidate", 0) or 0)
        hold = int(tc.get("hold_pending", 0) or 0)
        total = int(gov.get("row_count", 0) or 0)
        gov_stab = (
            f"A 类 {tier_a} 条 / 暂缓 {hold} 条"
            if total
            else "训练库行数未知"
        )
        rows.append(
            PipelineTrustRow(
                task="世界数据库 · 协议治理",
                method="data_tier / source_group / 文献溯源侧车",
                evidence="outputs/crack_governance/crack_training_governance.json",
                stability=gov_stab,
                tone="low" if tier_a >= max(5, total // 10) else "mid" if tier_a > 0 else "high",
                note="填写 A/B/C 分级后可显著提升主模型结论可辩护性。",
            )
        )

    return rows


def _downgrade(level: str, tone: str) -> tuple[str, str]:
    order = [("较高", "low"), ("中等", "mid"), ("偏低", "high")]
    for i, pair in enumerate(order):
        if pair[0] == level:
            return order[min(i + 1, len(order) - 1)]
    return level, tone


def assess_trust(
    *,
    checks: list[tuple[str, str, str]],
    stability: list[tuple[str, str, str]],
    result: dict[str, Any],
    user_inputs: dict[str, Any] | None = None,
) -> TrustAssessment:
    """综合输入区间、离线稳定性、世界数据库与公式/残差管线，生成可信度结论。"""
    _ = user_inputs  # 预留：后续可接邻域距离 / 文献质量分
    bundle = load_world_trust_bundle()
    gov = bundle.get("governance") or {}
    lab = bundle.get("lab_strength") or {}

    warn = sum(1 for _, _, t in checks if t in ("mid", "high"))
    muted = sum(1 for _, _, t in checks if t == "muted")
    stab_map = {r[0]: (r[1], r[2]) for r in stability}
    width_ok = stab_map.get("裂缝宽度预测", ("", ""))[1] == "low"
    dens_bad = stab_map.get("裂缝密度预测", ("", ""))[1] == "high"
    risk_mid = stab_map.get("开裂风险分类", ("", ""))[1] in ("mid", "high")

    if warn >= 2 or dens_bad:
        level, tone = "偏低", "high"
    elif warn >= 1 or risk_mid or muted >= 2:
        level, tone = "中等", "mid"
    else:
        level, tone = "较高", "low"

    tc = gov.get("tier_ABC_hold_counts") or {}
    total_rows = int(gov.get("row_count", 0) or 0)
    hold = int(tc.get("hold_pending", 0) or 0)
    tier_a = int(tc.get("tier_A_candidate", 0) or 0)
    gov_weak = total_rows > 0 and tier_a == 0 and hold >= int(total_rows * 0.8)

    if gov_weak and level == "较高":
        level, tone = _downgrade(level, tone)
    if gov_weak and hold == total_rows and level == "中等":
        level, tone = _downgrade(level, tone)

    in_range = warn == 0 and muted <= 1
    intro = (
        "当前输入大部分位于训练经验范围内；"
        "力学/温度路径优先采用国标或物理公式基线，主开裂为数据驱动模型。"
        if in_range
        else "当前输入部分参数偏离训练常见区间；"
        "宜优先参考有公式锚点的力学估算，主开裂结果作敏感性参考。"
    )

    positives: list[str] = []
    if width_ok:
        positives.append("裂缝宽度模型在历史 hold-out 中相对稳定")
    if stab_map.get("开裂风险分类", ("", ""))[1] == "low":
        positives.append("开裂风险分类在历史测试中判别尚可")
    if in_range:
        positives.append("关键材料与工艺参数未出现明显分布外告警")

    comp_r2 = _oof_r2(lab, "compressive")
    if comp_r2 is not None and comp_r2 >= 0.85:
        positives.append(
            f"抗压力学路径：国标公式基线离线 OOF R²≈{comp_r2:.2f}，"
            "优先采用公式而非未验证残差"
        )
    if lab.get("residual_not_recommended"):
        positives.append(
            "力学残差管线已做 OOF 对照：当前数据下公式基线不劣于残差，界面以公式为准"
        )
    if bundle.get("thermal_stress"):
        positives.append(
            "温度应力链已接入 C30 世界数据库并训练公式+残差模型，可用于机理解释交叉核对"
        )
    if tier_a > 0:
        positives.append(f"训练库已有 {tier_a} 条 A 类高可信协议样本可对照")
    if not positives:
        positives.append("系统已完成预测并给出工程化风险与裂缝指标")

    caveats: list[str] = []
    caveats.append(
        "主开裂（缝宽/密度/风险）为 XGBoost 端到端，缺乏与国标公式一一对应的独立锚点"
    )
    if dens_bad:
        caveats.append("裂缝密度模型离线稳定性较弱，密度数值建议结合现场监测判断")
    if risk_mid:
        caveats.append("开裂风险分类离线表现一般，宜与试验与规范验算交叉核对")
    if warn >= 1:
        caveats.append("部分输入偏离训练常见范围，外推风险需自行评估")
    if muted >= 1:
        caveats.append("温度等可选路径未闭合时，温度相关解释权重下降")
    if gov_weak:
        caveats.append(
            "世界数据库协议分层（data_tier/source_group）尚未填写，"
            "训练样本均处于暂缓状态，主模型结论可辩护性受限"
        )
    if lab.get("residual_not_recommended"):
        caveats.append(
            "力学抗压残差学习未优于公式基线，请勿将残差修正当作已验证增益"
        )

    mid = result.get("intermediate") or {}
    if mid.get("crack_density_source") == "fallback":
        caveats.append("本次裂缝密度采用经验回退，不宜作为精确监测值")

    thermal_missing = mid.get("thermal_stress_index_missing_flag")
    if thermal_missing is None and user_inputs:
        try:
            from src.input_defaults import normalize_prediction_inputs
            from src.thermal_stress_inputs import series_for_thermal_derive
            from experiments.thermal_stress.derive import derive_thermal_stress_features

            merged = normalize_prediction_inputs(user_inputs)
            feats = derive_thermal_stress_features(series_for_thermal_derive(merged))
            thermal_missing = feats.get("thermal_stress_index_missing_flag")
        except Exception:
            thermal_missing = None
    if thermal_missing == 1:
        caveats.append("温度应力解释指数未完整计算，不宜单独作为开裂判定依据")

    pipelines = build_pipeline_rows(stability)

    score = 72
    score -= warn * 8
    score -= muted * 3
    if dens_bad:
        score -= 12
    if risk_mid:
        score -= 8
    if width_ok:
        score += 5
    if comp_r2 is not None and comp_r2 >= 0.85:
        score += 6
    if gov_weak:
        score -= 15
    if tier_a > 0:
        score += min(10, tier_a)
    score = max(15, min(95, score))

    return TrustAssessment(
        level_label=level,
        tone=tone,
        intro=intro,
        positives=positives[:5],
        caveats=caveats[:6],
        pipelines=pipelines,
        trust_score=score,
    )
