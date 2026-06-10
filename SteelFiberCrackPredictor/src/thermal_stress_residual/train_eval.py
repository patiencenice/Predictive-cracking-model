from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold

from src.thermal_stress_residual.dataset import (
    build_xy_matrices,
    regression_metrics,
)


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


def cv_eval(
    df,
    *,
    target_col: str = "axial_stress_mpa",
    n_splits: int = 5,
) -> dict[str, Any]:
    X, y_resid, y_true, formula, feat_names, groups, row_indices, stats = build_xy_matrices(
        df, target_col=target_col
    )
    n = X.shape[0]
    if n < 3:
        raise ValueError(f"有效样本过少（{n}），至少需要 3 条以上记录。")

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

    for _fold_id, (train_idx, test_idx) in enumerate(split_iter):
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
            est.fit(X_tr, y_res_tr)
            res_te = est.predict(X_te)
            y_hat = f_te + res_te
            fold_metrics[key].append(regression_metrics(y_true_te, y_hat))
            oof_arr[test_idx] = res_te

    oof_global: dict[str, dict[str, float]] = {
        "formula_only": regression_metrics(y_true, formula),
    }
    for key, oof_arr in (
        ("linear", oof_lin),
        ("ridge", oof_ridge),
        ("hgb", oof_hgb),
    ):
        final = formula + oof_arr
        oof_global[key] = regression_metrics(y_true, final)

    mae_f = float(oof_global["formula_only"]["mae"])
    mae_residuals = {k: float(oof_global[k]["mae"]) for k in ("linear", "ridge", "hgb")}
    best_residual = min(mae_residuals, key=mae_residuals.get)
    default_method = (
        "formula_only"
        if mae_f < mae_residuals[best_residual] - 1e-9
        else best_residual
    )

    cv_summary: dict[str, Any] = {
        "target_col": target_col,
        "n_samples": int(n),
        "cv": cv_name,
        "n_folds": len(split_iter),
        "n_unique_groups": int(len(np.unique(groups))) if groups is not None else None,
    }
    for k in names:
        mae_l = [m["mae"] for m in fold_metrics[k]]
        cv_summary[k] = {
            "mae": _mean_std(mae_l),
            "rmse": _mean_std([m["rmse"] for m in fold_metrics[k]]),
            "r2": _mean_std([m["r2"] for m in fold_metrics[k]]),
        }

    return {
        "dataset_stats": stats,
        "cv_fold_metrics": cv_summary,
        "oof_global_metrics": oof_global,
        "default_method": default_method,
        "feature_names": feat_names,
        "row_indices": row_indices.tolist(),
        "groups": groups.tolist() if groups is not None else [],
    }


def run_pipeline(
    data_path: Path,
    out_dir: Path,
    *,
    target_col: str = "axial_stress_mpa",
    save_models: bool = False,
    n_splits: int = 5,
) -> dict[str, Any]:
    import pandas as pd

    from src.thermal_stress_residual.dataset import prepare_training_frame

    df = pd.read_csv(data_path)
    df = prepare_training_frame(df)
    rep = cv_eval(df, target_col=target_col, n_splits=n_splits)
    rep["data_path"] = str(data_path)
    rep["target_col"] = target_col
    rep["n_rows_csv"] = int(len(df))

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "thermal_stress_residual_report.json"
    report_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    if save_models:
        X, y_resid, _yt, formula, feat_names, _grp, _ri, _stats = build_xy_matrices(
            df, target_col=target_col
        )
        bundle = {
            "feature_names": feat_names,
            "target_col": target_col,
            "default_method": rep["default_method"],
            "formula": "sigma_T = R * E * alpha * delta_T",
        }
        method = rep["default_method"]
        if method != "formula_only":
            est = _residual_estimator_templates()[method]
            est.fit(X, y_resid)
            bundle["residual_model"] = est
            bundle["residual_model_name"] = method
        joblib.dump(bundle, out_dir / "thermal_stress_residual_bundle.joblib")

    return rep
