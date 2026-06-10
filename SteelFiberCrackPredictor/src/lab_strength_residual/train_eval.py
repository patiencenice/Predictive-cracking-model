from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold

from src.lab_strength_residual.dataset import (
    build_xy_matrices,
    merge_metrics_unweighted_weighted,
    regression_metrics,
    regression_metrics_weighted,
)
from src.lab_strength_residual.provenance_columns import summarize_tracing_columns
from src.lab_strength_residual.training_data_gate import validate_for_lab_strength_training


def _choose_cv(
    n_samples: int, groups: np.ndarray | None, n_splits: int = 5
) -> tuple[Any, np.ndarray | None, str]:
    if groups is not None:
        n_grp = int(len(np.unique(groups)))
        if n_grp >= 2:
            ns = min(max(2, n_splits), n_grp)
            return GroupKFold(n_splits=ns), groups, "GroupKFold"
    ns = min(max(2, n_splits), max(2, n_samples // 2))
    return KFold(n_splits=ns, shuffle=True, random_state=42), None, "KFold"


def _residual_estimator_templates() -> dict[str, Any]:
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "hgb": HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.08,
            max_iter=200,
            random_state=42,
        ),
    }


def _mean_std(xs: list[float]) -> dict[str, float]:
    a = np.array(xs, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
    }


def cv_eval_one_task(
    df: pd.DataFrame,
    *,
    task: str,
    n_splits: int = 5,
    append_source_domain: bool = True,
    use_sample_weight: bool = True,
    weight_local: float = 1.0,
    weight_literature: float = 0.7,
    weight_fiber_missing_mult: float = 0.8,
    include_lab_water_reducer_features: bool = True,
) -> dict[str, Any]:
    (
        X,
        y_resid,
        y_true,
        formula,
        _feat_names,
        groups,
        row_indices,
        sample_weights,
        dataset_stats,
    ) = build_xy_matrices(
        df,
        task=task,
        append_source_domain=append_source_domain,
        weight_local=weight_local,
        weight_literature=weight_literature,
        weight_fiber_missing_mult=weight_fiber_missing_mult,
        include_lab_water_reducer_features=include_lab_water_reducer_features,
    )
    n = X.shape[0]
    if n < 4:
        raise ValueError(f"{task}: 有效样本过少（{n}），至少需要约 4 条以上记录。")

    cv, grp, cv_name = _choose_cv(n, groups, n_splits=n_splits)
    if cv_name == "GroupKFold" and grp is not None:
        split_iter = list(cv.split(X, y_resid, groups=grp))
    else:
        split_iter = list(cv.split(X, y_resid))

    names = ("formula_only", "linear", "ridge", "hgb")
    fold_metrics: dict[str, list[dict[str, float]]] = {k: [] for k in names}
    templates = _residual_estimator_templates()

    oof_lin = np.full(n, np.nan)
    oof_ridge = np.full(n, np.nan)
    oof_hgb = np.full(n, np.nan)
    fold_of = np.full(n, -1, dtype=np.int32)
    group_str = (
        np.asarray(groups, dtype=object)
        if groups is not None
        else np.array(["NA"] * n, dtype=object)
    )

    for fold_id, (train_idx, test_idx) in enumerate(split_iter):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_res_tr = y_resid[train_idx]
        y_true_te = y_true[test_idx]
        f_te = formula[test_idx]

        fold_metrics["formula_only"].append(regression_metrics(y_true_te, f_te))

        for key, oof_arr in (
            ("linear", oof_lin),
            ("ridge", oof_ridge),
            ("hgb", oof_hgb),
        ):
            est = clone(templates[key])
            if use_sample_weight:
                est.fit(X_tr, y_res_tr, sample_weight=sample_weights[train_idx])
            else:
                est.fit(X_tr, y_res_tr)
            res_te = est.predict(X_te)
            y_hat = f_te + res_te
            fold_metrics[key].append(regression_metrics(y_true_te, y_hat))
            oof_arr[test_idx] = res_te

        fold_of[test_idx] = int(fold_id)

    cv_summary: dict[str, Any] = {
        "task": task,
        "n_samples": int(n),
        "cv": cv_name,
        "n_folds": len(split_iter),
    }
    for k in names:
        mae_l = [m["mae"] for m in fold_metrics[k]]
        rmse_l = [m["rmse"] for m in fold_metrics[k]]
        r2_l = [m["r2"] for m in fold_metrics[k]]
        cv_summary[k] = {
            "mae": _mean_std(mae_l),
            "rmse": _mean_std(rmse_l),
            "r2": _mean_std(r2_l),
        }

    # 全样本 OOF：同时给出非加权与加权指标（样本权重不改变 formula_only 预测值）
    oof_global: dict[str, dict[str, float]] = {
        "formula_only": merge_metrics_unweighted_weighted(
            regression_metrics(y_true, formula),
            regression_metrics_weighted(y_true, formula, sample_weights),
        ),
    }
    for key, oof_arr in (
        ("linear", oof_lin),
        ("ridge", oof_ridge),
        ("hgb", oof_hgb),
    ):
        final = formula + oof_arr
        oof_global[key] = merge_metrics_unweighted_weighted(
            regression_metrics(y_true, final),
            regression_metrics_weighted(y_true, final, sample_weights),
        )

    mae_f = float(oof_global["formula_only"]["mae"])
    mae_residuals = {
        k: float(oof_global[k]["mae"]) for k in ("linear", "ridge", "hgb")
    }
    best_residual_learner = min(mae_residuals, key=mae_residuals.get)
    min_res_mae = mae_residuals[best_residual_learner]
    formula_beats_all_residuals_oof = mae_f < min_res_mae - 1e-9
    effective_best_method = (
        "formula_only"
        if formula_beats_all_residuals_oof
        else str(best_residual_learner)
    )

    oof_rows: list[dict[str, Any]] = []
    for i in range(n):
        gid = str(group_str[i]) if group_str is not None else "NA"
        rid = int(row_indices[i])
        fid = int(fold_of[i])
        # 公式基线：残差视为 0
        oof_rows.append(
            {
                "row_id": rid,
                "task": task,
                "residual_model": "formula_only",
                "y_true": float(y_true[i]),
                "formula_pred": float(formula[i]),
                "residual_pred": 0.0,
                "final_pred": float(formula[i]),
                "source_group": gid,
                "fold_id": fid,
            }
        )
        for key, oof_arr in (
            ("linear", oof_lin),
            ("ridge", oof_ridge),
            ("hgb", oof_hgb),
        ):
            rp = float(oof_arr[i])
            oof_rows.append(
                {
                    "row_id": rid,
                    "task": task,
                    "residual_model": key,
                    "y_true": float(y_true[i]),
                    "formula_pred": float(formula[i]),
                    "residual_pred": rp,
                    "final_pred": float(formula[i] + rp),
                    "source_group": gid,
                    "fold_id": fid,
                }
            )

    return {
        "cv_summary": cv_summary,
        "oof_global_metrics": oof_global,
        "best_residual_learner_by_oof_mae": best_residual_learner,
        "formula_beats_all_residual_learners_oof_mae": formula_beats_all_residuals_oof,
        "effective_best_method_oof_mae": effective_best_method,
        "oof_rows": oof_rows,
        "dataset_stats": dataset_stats,
        "training_options": {
            "append_source_domain": append_source_domain,
            "use_sample_weight": use_sample_weight,
            "weight_local": weight_local,
            "weight_literature": weight_literature,
            "weight_fiber_missing_mult": weight_fiber_missing_mult,
            "include_lab_water_reducer_features": include_lab_water_reducer_features,
        },
    }


def _oof_metric_triplet(oof_global: dict[str, Any], name: str) -> dict[str, float]:
    blk = oof_global[name]
    return {
        "mae": float(blk["mae"]),
        "rmse": float(blk["rmse"]),
        "r2": float(blk["r2"]),
    }


def _water_reducer_ablation_report(
    comp_full: dict[str, Any],
    flex_full: dict[str, Any],
    comp_omit: dict[str, Any],
    flex_omit: dict[str, Any],
) -> dict[str, Any]:
    """full vs 去掉减水剂六列；delta = omit − full（MAE/RMSE 正表示去掉更差）。"""
    models = ("formula_only", "ridge", "hgb")
    full_slice: dict[str, Any] = {}
    omit_slice: dict[str, Any] = {}
    for task, full, omit in (
        ("compressive", comp_full, comp_omit),
        ("flexural", flex_full, flex_omit),
    ):
        fo = full["oof_global_metrics"]
        oo = omit["oof_global_metrics"]
        full_slice[task] = {m: _oof_metric_triplet(fo, m) for m in models}
        omit_slice[task] = {m: _oof_metric_triplet(oo, m) for m in models}

    delta: dict[str, Any] = {}
    formula_mae_check: dict[str, float] = {}
    for task in ("compressive", "flexural"):
        delta[task] = {}
        for m in models:
            fa = full_slice[task][m]
            ob = omit_slice[task][m]
            delta[task][m] = {
                "mae": ob["mae"] - fa["mae"],
                "rmse": ob["rmse"] - fa["rmse"],
                "r2": ob["r2"] - fa["r2"],
            }
        formula_mae_check[task] = float(delta[task]["formula_only"]["mae"])

    n_better = n_worse = n_tie = 0
    per_task_detail: dict[str, str] = {}
    for task in ("compressive", "flexural"):
        bits: list[str] = []
        for m in ("ridge", "hgb"):
            dm = float(delta[task][m]["mae"])
            base = max(float(full_slice[task][m]["mae"]), 1e-9)
            rel = abs(dm) / base
            if rel < 0.02 or abs(dm) < 1e-5:
                n_tie += 1
                bits.append(f"{m}:ΔMAE≈0(rel={rel:.4f})")
            elif dm > 0:
                n_worse += 1
                bits.append(f"{m}:去掉更差 ΔMAE={dm:+.6f}")
            else:
                n_better += 1
                bits.append(f"{m}:去掉更好 ΔMAE={dm:+.6f}")
        per_task_detail[task] = "; ".join(bits)

    rdc = float(delta["compressive"]["ridge"]["mae"])
    rdf = float(delta["flexural"]["ridge"]["mae"])
    hdc = float(delta["compressive"]["hgb"]["mae"])
    hdf = float(delta["flexural"]["hgb"]["mae"])
    ridge_both_better_omit = rdc < -1e-9 and rdf < -1e-9
    hgb_both_tie = abs(hdc) < 1e-9 and abs(hdf) < 1e-9

    if ridge_both_better_omit and hgb_both_tie:
        verdict = (
            "Ridge 残差在两任务上去掉减水剂六列后 OOF **略优**，HGB 上 OOF **完全不变**（六列对其冗余）。"
            "在当前小样本与弱可学习性下，六列更像**轻微拖 Ridge 后腿的冗余/噪声维**，"
            "而非稳定、可外推的物理效应；保留在 X 中并非必要。"
        )
    elif n_worse >= 3 and n_better == 0:
        verdict = (
            "多数对照下去掉减水剂列略抬高残差 OOF 误差，六列更像**弱有用的缺失/占位信号**，"
            "但仍不足以代表可辨识的物理机理。"
        )
    elif n_better >= 3 and n_worse == 0:
        verdict = (
            "多数对照下去掉减水剂列略降低残差 OOF 误差，六列在当前小样本下更像**噪声或轻度过拟合通道**，"
            "保留未必有益。"
        )
    elif n_tie >= 3:
        verdict = (
            "去掉减水剂六列后 OOF 与原先**接近不变**；与可学习性诊断一致，更像**无效保留字段**"
            "或仅产生噪声级波动。"
        )
    else:
        verdict = (
            "抗压/抗折、Ridge/HGB 上方向不一致或幅度很小：**不宜**把减水剂六列当作稳定增益；"
            "结论暧昧，需更多带标注减水剂的数据再判。"
        )

    return {
        "note_zh": (
            "同一 CSV、同一入模行集合；公式基线不变；GroupKFold 仅依赖样本数与 groups，"
            "fold 划分与 X 是否含减水剂六列无关。"
        ),
        "cv_n_samples": {
            "compressive": int(comp_full["cv_summary"]["n_samples"]),
            "flexural": int(flex_full["cv_summary"]["n_samples"]),
        },
        "cv_n_samples_match_full": {
            "compressive": int(comp_omit["cv_summary"]["n_samples"])
            == int(comp_full["cv_summary"]["n_samples"]),
            "flexural": int(flex_omit["cv_summary"]["n_samples"])
            == int(flex_full["cv_summary"]["n_samples"]),
        },
        "formula_only_mae_delta_omit_minus_full": formula_mae_check,
        "oof_unweighted_full": full_slice,
        "oof_unweighted_omit_water_reducer": omit_slice,
        "delta_omit_minus_full": delta,
        "per_task_residual_direction_zh": per_task_detail,
        "honest_verdict_zh": verdict,
    }


def _oof_dataframe(all_oof_rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(all_oof_rows)


def _compact_oof_metrics(comp: dict[str, Any], flex: dict[str, Any]) -> dict[str, Any]:
    """消融表用：仅保留各任务 OOF 指标与 effective_best。"""
    return {
        "compressive": {
            "oof_global_metrics": comp["oof_global_metrics"],
            "effective_best_method_oof_mae": comp["effective_best_method_oof_mae"],
        },
        "flexural": {
            "oof_global_metrics": flex["oof_global_metrics"],
            "effective_best_method_oof_mae": flex["effective_best_method_oof_mae"],
        },
    }


def _ablation_interpretation(
    primary: dict[str, Any],
    alternate: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """比较 primary 与 alternate 的非加权 MAE（抗压/抗折 formula、hgb）。"""
    lines: list[str] = []
    out: dict[str, Any] = {"label": label, "per_task": {}}
    for task in ("compressive", "flexural"):
        po = primary[task]["oof_global_metrics"]
        ao = alternate[task]["oof_global_metrics"]
        d: dict[str, Any] = {}
        for m in ("formula_only", "linear", "ridge", "hgb"):
            if m not in po or m not in ao:
                continue
            d[m] = {
                "mae_primary": po[m]["mae"],
                "mae_alternate": ao[m]["mae"],
                "delta_mae": float(ao[m]["mae"] - po[m]["mae"]),
                "r2_primary": po[m]["r2"],
                "r2_alternate": ao[m]["r2"],
            }
        out["per_task"][task] = d
        lines.append(f"{task}: {label} 下各模型 MAE 变化见 per_task")
    out["summary_lines"] = lines
    return out


def build_group_error_report(oof_df: pd.DataFrame) -> dict[str, Any]:
    """
    按 source_group 汇总：n、MAE、mean(final_pred - y_true)、mean(formula_pred - y_true)。
    区分 compressive / flexural；各 residual_model 单独一张表。
    """
    out: dict[str, Any] = {}
    for task in ("compressive", "flexural"):
        dft = oof_df[oof_df["task"] == task].copy()
        if dft.empty:
            out[task] = {}
            continue
        out[task] = {}
        for model in ("formula_only", "linear", "ridge", "hgb"):
            dm = dft[dft["residual_model"] == model].copy()
            if dm.empty:
                continue
            dm["abs_err"] = (dm["final_pred"] - dm["y_true"]).abs()
            dm["pred_minus_true"] = dm["final_pred"] - dm["y_true"]
            dm["formula_minus_true"] = dm["formula_pred"] - dm["y_true"]
            g = dm.groupby("source_group", dropna=False, observed=True)
            agg = g.agg(
                n=("row_id", "count"),
                mae=("abs_err", "mean"),
                mean_pred_minus_true=("pred_minus_true", "mean"),
                mean_formula_pred_minus_true=("formula_minus_true", "mean"),
            ).reset_index()
            out[task][model] = agg.to_dict(orient="records")
    return out


def _worst_source_group_by_mae(
    group_json: dict[str, Any], *, task: str, model: str
) -> dict[str, Any] | None:
    rows = (group_json.get(task) or {}).get(model)
    if not rows:
        return None
    worst = max(rows, key=lambda r: float(r.get("mae", 0.0)))
    return {"source_group": worst.get("source_group"), "mae": worst.get("mae")}


def write_group_error_report_md(
    path: Path,
    group_json: dict[str, Any],
    *,
    worst: dict[str, dict[str, Any | None]],
) -> None:
    """分组误差 Markdown：每任务一张总览（formula_only + hgb）。"""
    lines = [
        "# 按 source_group 的 OOF 误差汇总",
        "",
        "指标：n、MAE、mean(final_pred − y_true)、mean(formula_pred − y_true)。",
        "",
    ]
    for task in ("compressive", "flexural"):
        lines.append(f"## {task}")
        lines.append("")
        for model in ("formula_only", "hgb"):
            rows = (group_json.get(task) or {}).get(model)
            if not rows:
                continue
            lines.append(f"### {model}")
            lines.append("")
            lines.append(
                "| source_group | n | MAE | mean(pred-true) | mean(formula-true) |"
            )
            lines.append("|---|---:|---:|---:|---:|")
            for r in sorted(rows, key=lambda x: str(x.get("source_group", ""))):
                sg = str(r.get("source_group", "")).replace("|", "\\|")
                lines.append(
                    f"| {sg} | {r.get('n')} | {float(r.get('mae', 0.0)):.6f} | "
                    f"{float(r.get('mean_pred_minus_true', 0.0)):.6f} | "
                    f"{float(r.get('mean_formula_pred_minus_true', 0.0)):.6f} |"
                )
            lines.append("")
        w = worst.get(task)
        if w:
            lines.append(
                f"- **MAE 最大的组（{w.get('model', '')}）**: `{w.get('source_group')}` "
                f"(MAE≈{w.get('mae')})"
            )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# 专项诊断：已知最差 source_group（与 group 汇总报告一致）
WORST_GROUP_ROW_DIAG: dict[str, str] = {
    "compressive": "XUkun/C30_BASE",
    "flexural": "XUkun/C41_BASE",
}


def build_worst_group_row_diagnostics(oof_df: pd.DataFrame) -> dict[str, Any]:
    """
    最差组逐行：y_true、formula / ridge / hgb 的 OOF final_pred 及 pred-true；
    并汇总各模型 mean(pred-true) 用于判断系统性高/低估。
    """
    out: dict[str, Any] = {"groups": {}, "note": "pred_minus_true 为正表示整体高估真值。"}
    for task, group in WORST_GROUP_ROW_DIAG.items():
        sub = oof_df[
            (oof_df["task"] == task) & (oof_df["source_group"] == group)
        ].copy()
        if sub.empty:
            out["groups"][task] = {
                "source_group": group,
                "error": "OOF 表中未找到该组",
                "rows": [],
            }
            continue

        def _pred_for_model(rid: int, model: str) -> float | None:
            m = sub[(sub["row_id"] == rid) & (sub["residual_model"] == model)]
            if m.empty:
                return None
            return float(m["final_pred"].iloc[0])

        row_ids = sorted(int(x) for x in sub["row_id"].unique())
        rows_out: list[dict[str, Any]] = []
        for rid in row_ids:
            fo = sub[
                (sub["row_id"] == rid) & (sub["residual_model"] == "formula_only")
            ]
            if fo.empty:
                continue
            y_true = float(fo["y_true"].iloc[0])
            formula_pred = float(fo["final_pred"].iloc[0])
            rp = _pred_for_model(rid, "ridge")
            hp = _pred_for_model(rid, "hgb")
            rows_out.append(
                {
                    "row_id": rid,
                    "fold_id": int(fo["fold_id"].iloc[0]),
                    "y_true": y_true,
                    "formula_pred": formula_pred,
                    "ridge_pred": rp,
                    "hgb_pred": hp,
                    "formula_pred_minus_true": formula_pred - y_true,
                    "ridge_pred_minus_true": None
                    if rp is None
                    else (rp - y_true),
                    "hgb_pred_minus_true": None
                    if hp is None
                    else (hp - y_true),
                }
            )

        bias_summary: dict[str, Any] = {}
        for mname, col in (
            ("formula_only", "formula_pred_minus_true"),
            ("ridge", "ridge_pred_minus_true"),
            ("hgb", "hgb_pred_minus_true"),
        ):
            vals: list[float] = []
            for r in rows_out:
                v = r.get(col)
                if v is None:
                    continue
                fv = float(v)
                if np.isfinite(fv):
                    vals.append(fv)
            if not vals:
                bias_summary[mname] = {"mean_pred_minus_true": None, "interpretation": "无有效行"}
                continue
            mu = float(np.mean(vals))
            if mu > 1e-6:
                interp = "均值>0：该组内该模型整体偏高（pred 高于 true）。"
            elif mu < -1e-6:
                interp = "均值<0：该组内该模型整体偏低。"
            else:
                interp = "均值≈0：未见明显单向系统偏差。"
            bias_summary[mname] = {
                "mean_pred_minus_true": mu,
                "interpretation": interp,
            }

        out["groups"][task] = {
            "source_group": group,
            "n_rows": len(rows_out),
            "rows": rows_out,
            "bias_summary": bias_summary,
        }
    return out


def write_worst_group_row_diagnostics_md(path: Path, payload: dict[str, Any]) -> None:
    """最差组逐行诊断 Markdown。"""
    lines = [
        "# 最差 source_group 逐行 OOF 诊断",
        "",
        "列：y_true、formula_pred、ridge_pred、hgb_pred 及各自 pred−true。",
        "",
    ]
    for task in ("compressive", "flexural"):
        g = (payload.get("groups") or {}).get(task) or {}
        lines.append(f"## {task} — `{g.get('source_group', '')}`")
        lines.append("")
        if g.get("error"):
            lines.append(str(g["error"]))
            lines.append("")
            continue
        bs = g.get("bias_summary") or {}
        lines.append("### 组内系统性偏差（mean(pred−true)）")
        lines.append("")
        for m, b in bs.items():
            lines.append(f"- **{m}**: mean={b.get('mean_pred_minus_true')} — {b.get('interpretation', '')}")
        lines.append("")
        lines.append("### 逐行")
        lines.append("")
        lines.append(
            "| row_id | fold | y_true | formula | ridge | hgb | f−y | r−y | h−y |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in g.get("rows") or []:
            def _f(x: object) -> str:
                if x is None:
                    return ""
                return f"{float(x):.6f}"

            lines.append(
                f"| {r.get('row_id')} | {r.get('fold_id')} | {_f(r.get('y_true'))} | "
                f"{_f(r.get('formula_pred'))} | {_f(r.get('ridge_pred'))} | {_f(r.get('hgb_pred'))} | "
                f"{_f(r.get('formula_pred_minus_true'))} | {_f(r.get('ridge_pred_minus_true'))} | "
                f"{_f(r.get('hgb_pred_minus_true'))} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _diagnose_oof(
    oof_df: pd.DataFrame,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """基于 OOF 长表的简短诊断（JSON 可序列化）。"""
    out: dict[str, Any] = {}
    for task in ("compressive", "flexural"):
        dft = oof_df[oof_df["task"] == task].copy()
        if dft.empty:
            continue
        sub: dict[str, Any] = {"task": task}

        # 公式系统性偏差：mean(y_true - formula_pred)
        df_form = dft[dft["residual_model"] == "formula_only"].drop_duplicates(
            subset=["row_id", "fold_id"]
        )
        bias_f = float(np.mean(df_form["y_true"] - df_form["formula_pred"]))
        sub["formula_bias_mean_y_minus_formula"] = bias_f
        if abs(bias_f) < 1e-6:
            sub["formula_bias_note"] = "整体均值接近 0，未见明显系统性偏高/偏低。"
        elif bias_f > 0:
            sub["formula_bias_note"] = (
                "均值为正：整体上公式略低估真值（y_true 高于 formula_pred）。"
            )
        else:
            sub["formula_bias_note"] = "均值为负：整体上公式略高估真值。"

        # 各残差模型 OOF 整体偏差 mean(y_true - final_pred)
        bias_after: dict[str, float] = {}
        for m in ("linear", "ridge", "hgb"):
            dm = dft[dft["residual_model"] == m]
            bias_after[m] = float(np.mean(dm["y_true"] - dm["final_pred"]))
        sub["mean_y_minus_final_pred_by_model"] = bias_after

        dft_r = dft[dft["residual_model"].isin(("linear", "ridge", "hgb"))].copy()
        dft_r["_mae_row"] = (dft_r["y_true"] - dft_r["final_pred"]).abs()
        mean_mae = dft_r.groupby("residual_model", observed=True)["_mae_row"].mean()
        best = str(mean_mae.idxmin())
        sub["best_residual_model_by_mean_abs_error_on_oof"] = best
        b_best = bias_after.get(best, float("nan"))
        sub["bias_after_best_model"] = b_best
        sub["residual_reduced_abs_bias_vs_formula"] = bool(
            abs(float(b_best)) < abs(bias_f) - 1e-12
        )

        # 最差样本（按 best 模型的绝对误差）
        dm = dft[dft["residual_model"] == best].copy()
        dm["abs_err"] = np.abs(dm["y_true"] - dm["final_pred"])
        worst = (
            dm.sort_values("abs_err", ascending=False)
            .head(top_k)[
                [
                    "row_id",
                    "fold_id",
                    "source_group",
                    "y_true",
                    "formula_pred",
                    "residual_pred",
                    "final_pred",
                    "abs_err",
                ]
            ]
            .to_dict(orient="records")
        )
        sub[f"worst_{top_k}_samples_by_abs_error"] = worst

        # 按 source_group 的平均误差（final - y_true），公式与 best
        def _grp_mean_err(model: str) -> dict[str, float]:
            g = dft[dft["residual_model"] == model].copy()
            if g.empty:
                return {}
            g["_e"] = g["final_pred"] - g["y_true"]
            return g.groupby("source_group", dropna=False, observed=True)[
                "_e"
            ].mean().to_dict()

        sub["mean_final_minus_ytrue_by_source_group_formula_only"] = _grp_mean_err(
            "formula_only"
        )
        sub[
            f"mean_final_minus_ytrue_by_source_group_residual_{best}"
        ] = _grp_mean_err(str(best))

        out[task] = sub
    return out


def _write_diagnosis_markdown(
    path: Path, diag: dict[str, Any], automated: dict[str, Any]
) -> None:
    lines = [
        "# 强度残差 OOF 诊断摘要",
        "",
        "由 `train_eval._diagnose_oof` 与 `automated_summary` 自动生成。",
        "",
    ]
    if automated:
        lines.append("## OOF 指标摘要（启发式）")
        lines.append("")
        for task in ("compressive", "flexural"):
            b = automated.get(task)
            if not isinstance(b, dict):
                continue
            lines.append(f"### {task}")
            lines.append(
                f"- 最优残差学习器（三者中 OOF MAE 最低）: `{b.get('best_residual_learner_oof_mae', '')}`"
            )
            lines.append(
                f"- OOF MAE 公式: {b.get('oof_mae_formula_only', '')}；"
                f"最优残差学习器: {b.get('oof_mae_best_residual_learner', '')}；"
                f"综合推荐: `{b.get('effective_best_method_oof_mae', '')}`（MAE={b.get('oof_mae_effective_best', '')}）"
            )
            lines.append(
                f"- MAE 低于公式的残差学习器: {b.get('residual_models_with_lower_oof_mae_than_formula', [])}"
            )
            lines.append("")
        hint = automated.get("ui_integration_hint", {})
        lines.append(
            f"- 两任务「残差学习器」是否均优于公式（MAE）: "
            f"{hint.get('both_tasks_residual_beat_formula_mae', False)}"
        )
        lines.append(
            f"- 两任务综合最优是否均为残差（非纯公式）: "
            f"{hint.get('both_tasks_effective_best_is_residual_not_formula', False)}"
        )
        lines.append(
            f"- 两任务相对 MAE 降幅（相对公式）是否均 ≥3%: "
            f"{hint.get('meaningful_relative_drop_gte_3pct_both', False)}"
        )
        lines.append("")
    for task, block in diag.items():
        if not isinstance(block, dict):
            continue
        lines.append(f"## {task}")
        lines.append("")
        lines.append(
            f"- **公式基线整体偏差**（mean(y_true − formula_pred)）: "
            f"{block.get('formula_bias_mean_y_minus_formula', 'n/a')}"
        )
        lines.append(f"- 解读: {block.get('formula_bias_note', '')}")
        lines.append(
            f"- **OOF 上最优残差模型**（按 mean|error|）: "
            f"`{block.get('best_residual_model_by_mean_abs_error_on_oof', '')}`"
        )
        lines.append(
            f"- 最优模型后整体偏差 mean(y_true − final_pred): "
            f"{block.get('bias_after_best_model', 'n/a')}"
        )
        lines.append(
            f"- **是否缩小相对公式的 |偏差|**: "
            f"{block.get('residual_reduced_abs_bias_vs_formula', False)}"
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def fit_full_and_save(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    best_models: dict[str, str],
    formula_beats_all: dict[str, bool] | None = None,
    append_source_domain: bool = True,
    use_sample_weight: bool = True,
    weight_local: float = 1.0,
    weight_literature: float = 0.7,
    weight_fiber_missing_mult: float = 0.8,
    saved_residual_by_task: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    在全量数据上重训并保存残差模型。
    saved_residual_by_task 若给出则覆盖 best_models 中对应任务的残差学习器选择（用于固化默认落盘）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    templates = _residual_estimator_templates()
    formula_beats_all = formula_beats_all or {}
    sel_models = dict(best_models)
    if saved_residual_by_task:
        sel_models.update(saved_residual_by_task)
    out: dict[str, Any] = {}
    for task in ("compressive", "flexural"):
        (
            X,
            y_resid,
            _,
            _,
            feat_names,
            _,
            _,
            sample_weights,
            _dataset_stats,
        ) = build_xy_matrices(
            df,
            task=task,
            append_source_domain=append_source_domain,
            weight_local=weight_local,
            weight_literature=weight_literature,
            weight_fiber_missing_mult=weight_fiber_missing_mult,
        )
        key = sel_models.get(task, "hgb")
        if key not in templates:
            key = "hgb"
        model = clone(templates[key])
        if use_sample_weight:
            model.fit(X, y_resid, sample_weight=sample_weights)
        else:
            model.fit(X, y_resid)
        path = out_dir / f"lab_{task}_residual_{key}.joblib"
        fb = bool(formula_beats_all.get(task, False))
        joblib.dump(
            {
                "model": model,
                "feature_names": feat_names,
                "task": task,
                "kind": key,
                "formula_beats_all_residual_learners_oof_mae": fb,
                "note": (
                    "OOF 上纯公式 MAE 低于所有残差学习器；推理时建议优先使用公式基线。"
                    if fb
                    else ""
                ),
            },
            path,
        )
        out[task] = {
            "n_samples": int(X.shape[0]),
            "model_path": str(path.resolve()),
            "residual_model": key,
            "formula_beats_all_residual_learners_oof_mae": fb,
        }
    return out


def _automated_summary(comp: dict[str, Any], flex: dict[str, Any]) -> dict[str, Any]:
    """基于 OOF 全样本指标的简要自动结论（启发式）。"""
    summary: dict[str, Any] = {}
    for task, blk in (("compressive", comp), ("flexural", flex)):
        oof = blk["oof_global_metrics"]
        m0 = float(oof["formula_only"]["mae"])
        beat = [
            k
            for k in ("linear", "ridge", "hgb")
            if float(oof[k]["mae"]) < m0 - 1e-9
        ]
        br = str(blk["best_residual_learner_by_oof_mae"])
        m_br = float(oof[br]["mae"])
        eff = str(blk["effective_best_method_oof_mae"])
        m_eff = m0 if eff == "formula_only" else float(oof[eff]["mae"])
        summary[task] = {
            "best_residual_learner_oof_mae": br,
            "oof_mae_formula_only": m0,
            "oof_mae_best_residual_learner": m_br,
            "effective_best_method_oof_mae": eff,
            "oof_mae_effective_best": m_eff,
            "formula_beats_all_residual_learners_oof_mae": bool(
                blk["formula_beats_all_residual_learners_oof_mae"]
            ),
            "residual_models_with_lower_oof_mae_than_formula": beat,
            "all_three_residuals_beat_formula_mae": len(beat) == 3,
            "relative_mae_drop_effective_vs_formula": float((m0 - m_eff) / m0)
            if m0 > 1e-12
            else 0.0,
        }
    c_rel = summary["compressive"]["relative_mae_drop_effective_vs_formula"]
    f_rel = summary["flexural"]["relative_mae_drop_effective_vs_formula"]
    c_eff = summary["compressive"]["effective_best_method_oof_mae"]
    f_eff = summary["flexural"]["effective_best_method_oof_mae"]
    summary["ui_integration_hint"] = {
        "both_tasks_residual_beat_formula_mae": (
            summary["compressive"]["oof_mae_best_residual_learner"]
            < summary["compressive"]["oof_mae_formula_only"] - 1e-9
            and summary["flexural"]["oof_mae_best_residual_learner"]
            < summary["flexural"]["oof_mae_formula_only"] - 1e-9
        ),
        "both_tasks_effective_best_is_residual_not_formula": (
            c_eff != "formula_only" and f_eff != "formula_only"
        ),
        "meaningful_relative_drop_gte_3pct_both": c_rel >= 0.03 and f_rel >= 0.03,
    }

    # 固化默认策略说明 + 轻量 MAE 消融（相对公式）
    oc = comp["oof_global_metrics"]
    of = flex["oof_global_metrics"]
    m_cf = float(oc["formula_only"]["mae"])
    m_cr = float(oc["ridge"]["mae"])
    rel_drop_ridge_c = float((m_cf - m_cr) / m_cf) if m_cf > 1e-12 else 0.0
    m_ff = float(of["formula_only"]["mae"])
    m_fr = float(of["ridge"]["mae"])
    m_fh = float(of["hgb"]["mae"])
    rel_drop_ridge_f = float((m_ff - m_fr) / m_ff) if m_ff > 1e-12 else 0.0
    rel_drop_hgb_f = float((m_ff - m_fh) / m_ff) if m_ff > 1e-12 else 0.0

    summary["default_method_by_task"] = {
        "compressive": {
            "strategy": "formula_plus_ridge_residual",
            "description": "国标公式基线 + Ridge 残差（默认落盘与推荐推理组合）",
            "residual_learner": "ridge",
        },
        "flexural": {
            "strategy": "formula_only",
            "description": "抗折默认仅使用国标公式基线（残差模型可选，不作为默认）",
            "residual_learner": None,
        },
    }
    summary["light_ablation_mae_drop"] = {
        "compressive": {
            "formula_only": {"mae": m_cf, "relative_mae_drop_vs_formula": 0.0},
            "ridge": {"mae": m_cr, "relative_mae_drop_vs_formula": rel_drop_ridge_c},
        },
        "flexural": {
            "formula_only": {"mae": m_ff, "relative_mae_drop_vs_formula": 0.0},
            "ridge": {"mae": m_fr, "relative_mae_drop_vs_formula": rel_drop_ridge_f},
            "hgb": {"mae": m_fh, "relative_mae_drop_vs_formula": rel_drop_hgb_f},
        },
    }
    summary["practical_gain"] = bool(rel_drop_ridge_c >= 0.05 - 1e-12)
    summary["residual_not_recommended"] = bool(
        m_ff <= min(m_fr, m_fh) + 1e-12
    )

    return summary


def _print_dataset_usage_block(
    ds: dict[str, Any],
    *,
    n_comp: int,
    n_flex: int,
) -> None:
    """终端输出：合并表行数、各任务入模样本数、fiber_type 缺失保留行数。"""
    print(
        "[lab_strength_residual] 输入 CSV 总行数 n_rows_csv="
        f"{ds.get('n_rows_csv', 'n/a')}"
    )
    print(
        f"  标签可解析行数 n_rows_after_task_filter="
        f"{ds.get('n_rows_after_task_filter', 'n/a')}"
    )
    print(
        "  等价于「旧版仅因 fiber_type_enc 缺失而丢行」的样本数 "
        f"n_rows_dropped_due_to_missing_before_patch_equivalent="
        f"{ds.get('n_rows_dropped_due_to_missing_before_patch_equivalent', 'n/a')}"
    )
    print(
        f"  最终进入训练矩阵行数 n_rows_final_used={ds.get('n_rows_final_used', 'n/a')} "
        f"(抗压 OOF n_samples={n_comp}，抗折 OOF n_samples={n_flex})"
    )
    print(
        "  其中 fiber_type_enc 缺失、已用 -1.0 + fiber_type_missing_flag=1.0 保留的行数 "
        f"fiber_type_missing_rows_used={ds.get('fiber_type_missing_rows_used', 'n/a')}"
    )
    n_mr = int(ds.get("manual_review_rows_count") or 0)
    if n_mr:
        sgs = ds.get("manual_review_source_groups") or []
        print(
            f"  needs_manual_review 暂缓入模行数 manual_review_rows_count={n_mr} "
            f"source_groups={sgs}"
        )


def run_pipeline(
    csv_path: Path,
    out_dir: Path,
    *,
    save_models: bool,
    top_k_worst: int = 5,
    append_source_domain: bool = True,
    use_sample_weight: bool = True,
    weight_local: float = 1.0,
    weight_literature: float = 0.7,
    weight_fiber_missing_mult: float = 0.8,
) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    # 训练前强制闸门：不通过则中止，不写模型、不写报告（由调用方处理退出码）
    gate = validate_for_lab_strength_training(df)
    if not gate["ok"]:
        print("lab_strength 训练数据未通过闸门，中止（不进入 CV、不写模型与报告）:")
        for p in gate.get("problems") or []:
            print(" -", p)
        raise SystemExit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    comp = cv_eval_one_task(
        df,
        task="compressive",
        append_source_domain=append_source_domain,
        use_sample_weight=use_sample_weight,
        weight_local=weight_local,
        weight_literature=weight_literature,
        weight_fiber_missing_mult=weight_fiber_missing_mult,
    )
    flex = cv_eval_one_task(
        df,
        task="flexural",
        append_source_domain=append_source_domain,
        use_sample_weight=use_sample_weight,
        weight_local=weight_local,
        weight_literature=weight_literature,
        weight_fiber_missing_mult=weight_fiber_missing_mult,
    )
    dataset_stats = comp.get("dataset_stats") or {}

    comp_wr_off = cv_eval_one_task(
        df,
        task="compressive",
        append_source_domain=append_source_domain,
        use_sample_weight=use_sample_weight,
        weight_local=weight_local,
        weight_literature=weight_literature,
        weight_fiber_missing_mult=weight_fiber_missing_mult,
        include_lab_water_reducer_features=False,
    )
    flex_wr_off = cv_eval_one_task(
        df,
        task="flexural",
        append_source_domain=append_source_domain,
        use_sample_weight=use_sample_weight,
        weight_local=weight_local,
        weight_literature=weight_literature,
        weight_fiber_missing_mult=weight_fiber_missing_mult,
        include_lab_water_reducer_features=False,
    )
    water_reducer_feature_ablation = _water_reducer_ablation_report(
        comp, flex, comp_wr_off, flex_wr_off
    )

    # 消融：source_domain 特征关（样本权重规则不变）
    comp_dom_f = cv_eval_one_task(
        df,
        task="compressive",
        append_source_domain=False,
        use_sample_weight=use_sample_weight,
        weight_local=weight_local,
        weight_literature=weight_literature,
        weight_fiber_missing_mult=weight_fiber_missing_mult,
    )
    flex_dom_f = cv_eval_one_task(
        df,
        task="flexural",
        append_source_domain=False,
        use_sample_weight=use_sample_weight,
        weight_local=weight_local,
        weight_literature=weight_literature,
        weight_fiber_missing_mult=weight_fiber_missing_mult,
    )

    # 消融：样本权重关（拟合时不传 sample_weight；指标仍同时算加权与非加权）
    comp_sw_f = cv_eval_one_task(
        df,
        task="compressive",
        append_source_domain=append_source_domain,
        use_sample_weight=False,
        weight_local=weight_local,
        weight_literature=weight_literature,
        weight_fiber_missing_mult=weight_fiber_missing_mult,
    )
    flex_sw_f = cv_eval_one_task(
        df,
        task="flexural",
        append_source_domain=append_source_domain,
        use_sample_weight=False,
        weight_local=weight_local,
        weight_literature=weight_literature,
        weight_fiber_missing_mult=weight_fiber_missing_mult,
    )

    oof_df = _oof_dataframe(comp["oof_rows"] + flex["oof_rows"])
    oof_path = out_dir / "lab_strength_oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)

    diag = _diagnose_oof(oof_df, top_k=top_k_worst)
    diag_path = out_dir / "lab_strength_oof_diagnosis.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

    best_models = {
        "compressive": str(comp["best_residual_learner_by_oof_mae"]),
        "flexural": str(flex["best_residual_learner_by_oof_mae"]),
    }
    automated = _automated_summary(comp, flex)

    md_path = out_dir / "lab_strength_oof_diagnosis.md"
    _write_diagnosis_markdown(md_path, diag, automated)

    n_comp = int(comp["cv_summary"]["n_samples"])
    n_flex = int(flex["cv_summary"]["n_samples"])

    group_json = build_group_error_report(oof_df)
    group_json_path = out_dir / "lab_strength_group_error_by_source_group.json"
    with open(group_json_path, "w", encoding="utf-8") as f:
        json.dump(group_json, f, ensure_ascii=False, indent=2)

    worst_formula: dict[str, Any] = {}
    for task in ("compressive", "flexural"):
        w = _worst_source_group_by_mae(group_json, task=task, model="formula_only")
        if w:
            worst_formula[task] = {**w, "model": "formula_only"}
    group_worst_mae = {
        m: {
            "compressive": _worst_source_group_by_mae(
                group_json, task="compressive", model=m
            ),
            "flexural": _worst_source_group_by_mae(
                group_json, task="flexural", model=m
            ),
        }
        for m in ("formula_only", "hgb")
    }
    group_md_path = out_dir / "lab_strength_group_error_by_source_group.md"
    write_group_error_report_md(
        group_md_path, group_json, worst=worst_formula
    )

    worst_row_payload = build_worst_group_row_diagnostics(oof_df)
    worst_row_json = out_dir / "lab_strength_worst_groups_row_diagnostics.json"
    with open(worst_row_json, "w", encoding="utf-8") as f:
        json.dump(worst_row_payload, f, ensure_ascii=False, indent=2)
    worst_row_md = out_dir / "lab_strength_worst_groups_row_diagnostics.md"
    write_worst_group_row_diagnostics_md(worst_row_md, worst_row_payload)

    primary_compact = _compact_oof_metrics(comp, flex)
    dom_off_compact = _compact_oof_metrics(comp_dom_f, flex_dom_f)
    sw_off_compact = _compact_oof_metrics(comp_sw_f, flex_sw_f)

    source_domain_ablation = {
        "primary_append_source_domain_true": primary_compact,
        "alternate_append_source_domain_false": dom_off_compact,
        "interpretation_unweighted_mae": _ablation_interpretation(
            primary_compact,
            dom_off_compact,
            label="关闭 source_domain 特征列",
        ),
    }
    sample_weight_ablation = {
        "primary_use_sample_weight_true": primary_compact,
        "alternate_use_sample_weight_false": sw_off_compact,
        "interpretation_unweighted_mae": _ablation_interpretation(
            primary_compact,
            sw_off_compact,
            label="关闭训练 sample_weight",
        ),
    }

    report: dict[str, Any] = {
        "data": str(csv_path.resolve()),
        "optional_tracing_summary": summarize_tracing_columns(df),
        "default_method_by_task": automated["default_method_by_task"],
        "practical_gain": automated["practical_gain"],
        "residual_not_recommended": automated["residual_not_recommended"],
        "training_options": {
            "append_source_domain": append_source_domain,
            "use_sample_weight": use_sample_weight,
            "weight_local": weight_local,
            "weight_literature": weight_literature,
            "weight_fiber_missing_mult": weight_fiber_missing_mult,
            "source_domain_note": (
                "source_domain 为可选试验特征；默认仍写入矩阵，但不对其效果作对外主卖点表述。"
            ),
        },
        "group_error_by_source_group_json": str(group_json_path.resolve()),
        "group_error_by_source_group_md": str(group_md_path.resolve()),
        "worst_groups_row_diagnostics_json": str(worst_row_json.resolve()),
        "worst_groups_row_diagnostics_md": str(worst_row_md.resolve()),
        "diagnostics": {
            "source_domain_ablation": source_domain_ablation,
            "sample_weight_ablation": sample_weight_ablation,
            "light_ablation_mae_drop": automated["light_ablation_mae_drop"],
        },
        "group_worst_mae_by_task": group_worst_mae,
        "n_rows_csv": dataset_stats.get("n_rows_csv"),
        "n_rows_after_task_filter": dataset_stats.get("n_rows_after_task_filter"),
        "n_rows_dropped_due_to_missing_before_patch_equivalent": dataset_stats.get(
            "n_rows_dropped_due_to_missing_before_patch_equivalent"
        ),
        "n_rows_final_used": dataset_stats.get("n_rows_final_used"),
        "fiber_type_missing_rows_used": dataset_stats.get(
            "fiber_type_missing_rows_used"
        ),
        "manual_review_rows_count": dataset_stats.get("manual_review_rows_count", 0),
        "manual_review_source_groups": dataset_stats.get(
            "manual_review_source_groups", []
        ),
        "manual_review_samples": dataset_stats.get("manual_review_samples", []),
        "water_reducer_feature_summary": dataset_stats.get(
            "water_reducer_feature_summary", {}
        ),
        "water_reducer_learnability": dataset_stats.get(
            "water_reducer_learnability", {}
        ),
        "water_reducer_feature_ablation": water_reducer_feature_ablation,
        "oof_predictions_csv": str(oof_path.resolve()),
        "oof_diagnosis_json": str(diag_path.resolve()),
        "oof_diagnosis_md": str(md_path.resolve()),
        "compressive": {
            "cv_fold_metrics": comp["cv_summary"],
            "oof_global_metrics": comp["oof_global_metrics"],
            "best_residual_learner_by_oof_mae": comp["best_residual_learner_by_oof_mae"],
            "formula_beats_all_residual_learners_oof_mae": comp[
                "formula_beats_all_residual_learners_oof_mae"
            ],
            "effective_best_method_oof_mae": comp["effective_best_method_oof_mae"],
        },
        "flexural": {
            "cv_fold_metrics": flex["cv_summary"],
            "oof_global_metrics": flex["oof_global_metrics"],
            "best_residual_learner_by_oof_mae": flex["best_residual_learner_by_oof_mae"],
            "formula_beats_all_residual_learners_oof_mae": flex[
                "formula_beats_all_residual_learners_oof_mae"
            ],
            "effective_best_method_oof_mae": flex["effective_best_method_oof_mae"],
        },
        "best_residual_models": best_models,
        "automated_summary": automated,
    }
    # 逐行丢行诊断（抗压/抗折各自 build_xy_matrices 统计；明细有条数上限）
    dropped_block = {
        "compressive": (comp.get("dataset_stats") or {}).get(
            "dropped_rows_diagnosis"
        ),
        "flexural": (flex.get("dataset_stats") or {}).get("dropped_rows_diagnosis"),
    }
    report["dropped_rows_diagnosis"] = dropped_block
    dr_path = out_dir / "lab_strength_dropped_rows.json"
    with open(dr_path, "w", encoding="utf-8") as f:
        json.dump(dropped_block, f, ensure_ascii=False, indent=2)

    saved_residual_by_task = {
        "compressive": "ridge",
        "flexural": str(flex["best_residual_learner_by_oof_mae"]),
    }
    report["saved_residual_by_task"] = saved_residual_by_task

    if save_models:
        report["saved_models"] = fit_full_and_save(
            df,
            out_dir,
            best_models=best_models,
            formula_beats_all={
                "compressive": bool(
                    comp["formula_beats_all_residual_learners_oof_mae"]
                ),
                "flexural": bool(flex["formula_beats_all_residual_learners_oof_mae"]),
            },
            append_source_domain=append_source_domain,
            use_sample_weight=use_sample_weight,
            weight_local=weight_local,
            weight_literature=weight_literature,
            weight_fiber_missing_mult=weight_fiber_missing_mult,
            saved_residual_by_task=saved_residual_by_task,
        )

    _print_dataset_usage_block(dataset_stats, n_comp=n_comp, n_flex=n_flex)
    print(
        "[lab_strength_residual] 分组误差报告:",
        group_json_path.resolve(),
        group_md_path.resolve(),
    )
    print(
        "[lab_strength_residual] 最差组逐行诊断:",
        worst_row_json.resolve(),
        worst_row_md.resolve(),
    )
    print("[lab_strength_residual] 丢行诊断:", dr_path.resolve())

    with open(out_dir / "lab_strength_residual_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
