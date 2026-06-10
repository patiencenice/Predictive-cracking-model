"""
训练管线：与 FEATURE_COLUMNS 对齐的划分、标准化、三任务 XGBoost、指标与落盘。
供 train_model.py 与 model_bootstrap 复用，避免两套超参漂移。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.features import FEATURE_COLUMNS
from src.paths import OUTPUTS_DIR

DEFAULT_CONFIG: dict[str, Any] = {
    "training": {
        "test_size": 0.2,
        "random_state": 42,
        "stratify_risk": True,
    },
    "xgboost": {
        "regressor_width": {
            "n_estimators": 280,
            "max_depth": 5,
            "learning_rate": 0.06,
            "subsample": 0.88,
            "colsample_bytree": 0.88,
            "reg_lambda": 1.0,
            "min_child_weight": 1,
        },
        "regressor_density": {
            "n_estimators": 260,
            "max_depth": 5,
            "learning_rate": 0.06,
            "subsample": 0.88,
            "colsample_bytree": 0.88,
            "reg_lambda": 1.0,
            "min_child_weight": 1,
        },
        "classifier_risk": {
            "n_estimators": 280,
            "max_depth": 5,
            "learning_rate": 0.06,
            "subsample": 0.88,
            "colsample_bytree": 0.88,
            "reg_lambda": 1.0,
            "min_child_weight": 1,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_model_config(path: Path) -> dict[str, Any]:
    base = copy.deepcopy(DEFAULT_CONFIG)
    if not path.exists():
        return base
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not raw:
        return base
    return _deep_merge(base, raw)


def _xgb_params(section: str, config: dict[str, Any], random_state: int) -> dict[str, Any]:
    xgb = config.get("xgboost", {})
    base = DEFAULT_CONFIG["xgboost"].get(section, {})
    override = xgb.get(section, {})
    params = {**base, **override}
    params["random_state"] = random_state
    params["n_jobs"] = -1
    return params


def cross_validate_models(
    X: pd.DataFrame,
    y_width: pd.Series,
    y_density: pd.Series,
    y_risk: pd.Series,
    config: dict[str, Any],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    K 折交叉验证（每折内单独 fit StandardScaler 与三个 XGBoost）。
    分类折划分优先 StratifiedKFold（与 stratify_risk 思路一致）；不可行时退回 KFold。

    返回：各折指标、均值与标准差，便于论文/报告引用。
    """
    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"特征表缺少列: {missing}")

    Xo = X[FEATURE_COLUMNS].astype(np.float64, copy=False).reset_index(drop=True)
    yw = y_width.astype(np.float64, copy=False).reset_index(drop=True)
    yd = y_density.astype(np.float64, copy=False).reset_index(drop=True)
    yr = y_risk.astype(int, copy=False).reset_index(drop=True)

    n = len(Xo)
    if n < n_splits * 3:
        raise ValueError(
            f"样本量 {n} 过少：建议至少 n_splits×3 条以上再作 {n_splits} 折交叉验证。"
        )

    tcfg = config.get("training", {})
    rs_base = int(tcfg.get("random_state", random_state))

    use_stratify_cfg = bool(tcfg.get("stratify_risk", True))
    stratify_used = False
    if use_stratify_cfg:
        try:
            splits = list(
                StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=random_state
                ).split(Xo, yr)
            )
            stratify_used = True
        except ValueError:
            splits = list(
                KFold(
                    n_splits=n_splits, shuffle=True, random_state=random_state
                ).split(Xo)
            )
    else:
        splits = list(
            KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(Xo)
        )

    fold_rows: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        X_tr = Xo.iloc[train_idx]
        X_val = Xo.iloc[val_idx]
        yw_tr, yw_val = yw.iloc[train_idx], yw.iloc[val_idx]
        yd_tr, yd_val = yd.iloc[train_idx], yd.iloc[val_idx]
        yr_tr, yr_val = yr.iloc[train_idx], yr.iloc[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        rs = rs_base + fold_idx
        reg_w = XGBRegressor(
            **{
                k: v
                for k, v in _xgb_params("regressor_width", config, rs).items()
                if k != "eval_metric"
            }
        )
        reg_d = XGBRegressor(
            **{
                k: v
                for k, v in _xgb_params("regressor_density", config, rs + 17).items()
                if k != "eval_metric"
            }
        )
        clf = XGBClassifier(**_xgb_params("classifier_risk", config, rs + 29))

        reg_w.fit(X_tr_s, yw_tr)
        reg_d.fit(X_tr_s, yd_tr)

        if len(np.unique(yr_tr)) < 2:
            raise ValueError(
                f"第 {fold_idx} 折训练集中开裂风险类别少于 2 类，无法训练分类器。"
                "请减少折数或增加少数类样本。"
            )

        clf.fit(X_tr_s, yr_tr)

        pw = reg_w.predict(X_val_s)
        pd_ = reg_d.predict(X_val_s)
        pr = clf.predict(X_val_s)

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "crack_width": {
                    "mae": float(mean_absolute_error(yw_val, pw)),
                    "rmse": float(np.sqrt(mean_squared_error(yw_val, pw))),
                    "r2": float(r2_score(yw_val, pw)),
                },
                "crack_density": {
                    "mae": float(mean_absolute_error(yd_val, pd_)),
                    "rmse": float(np.sqrt(mean_squared_error(yd_val, pd_))),
                    "r2": float(r2_score(yd_val, pd_)),
                },
                "cracking_risk": {
                    "accuracy": float(accuracy_score(yr_val, pr)),
                    "macro_f1": float(
                        f1_score(yr_val, pr, average="macro", zero_division=0)
                    ),
                    "weighted_f1": float(
                        f1_score(yr_val, pr, average="weighted", zero_division=0)
                    ),
                },
            }
        )

    def _agg(key_path: list[str]) -> tuple[float, float]:
        vals = []
        for row in fold_rows:
            d: Any = row
            for k in key_path:
                d = d[k]
            vals.append(float(d))
        a = np.asarray(vals, dtype=np.float64)
        return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0

    summary = {
        "crack_width": {
            "mae_mean": _agg(["crack_width", "mae"])[0],
            "mae_std": _agg(["crack_width", "mae"])[1],
            "rmse_mean": _agg(["crack_width", "rmse"])[0],
            "rmse_std": _agg(["crack_width", "rmse"])[1],
            "r2_mean": _agg(["crack_width", "r2"])[0],
            "r2_std": _agg(["crack_width", "r2"])[1],
        },
        "crack_density": {
            "mae_mean": _agg(["crack_density", "mae"])[0],
            "mae_std": _agg(["crack_density", "mae"])[1],
            "rmse_mean": _agg(["crack_density", "rmse"])[0],
            "rmse_std": _agg(["crack_density", "rmse"])[1],
            "r2_mean": _agg(["crack_density", "r2"])[0],
            "r2_std": _agg(["crack_density", "r2"])[1],
        },
        "cracking_risk": {
            "accuracy_mean": _agg(["cracking_risk", "accuracy"])[0],
            "accuracy_std": _agg(["cracking_risk", "accuracy"])[1],
            "macro_f1_mean": _agg(["cracking_risk", "macro_f1"])[0],
            "macro_f1_std": _agg(["cracking_risk", "macro_f1"])[1],
            "weighted_f1_mean": _agg(["cracking_risk", "weighted_f1"])[0],
            "weighted_f1_std": _agg(["cracking_risk", "weighted_f1"])[1],
        },
    }

    report: dict[str, Any] = {
        "n_splits": n_splits,
        "n_samples": n,
        "stratify_risk_requested": use_stratify_cfg,
        "stratified_splits_used": stratify_used,
        "random_state": random_state,
        "folds": fold_rows,
        "summary_mean_std": summary,
    }

    if verbose:
        print(f"=== {n_splits}-fold CV (n={n}) ===\n")
        w = summary["crack_width"]
        print(
            "crack_width: "
            f"MAE={w['mae_mean']:.4f}+/-{w['mae_std']:.4f}, "
            f"RMSE={w['rmse_mean']:.4f}+/-{w['rmse_std']:.4f}, "
            f"R2={w['r2_mean']:.4f}+/-{w['r2_std']:.4f}"
        )
        d = summary["crack_density"]
        print(
            "crack_density: "
            f"MAE={d['mae_mean']:.4f}+/-{d['mae_std']:.4f}, "
            f"RMSE={d['rmse_mean']:.4f}+/-{d['rmse_std']:.4f}, "
            f"R2={d['r2_mean']:.4f}+/-{d['r2_std']:.4f}"
        )
        r = summary["cracking_risk"]
        print(
            "cracking_risk: "
            f"Accuracy={r['accuracy_mean']:.4f}+/-{r['accuracy_std']:.4f}, "
            f"Macro-F1={r['macro_f1_mean']:.4f}+/-{r['macro_f1_std']:.4f}, "
            f"Weighted-F1={r['weighted_f1_mean']:.4f}+/-{r['weighted_f1_std']:.4f}"
        )

    return report


def fit_models_save(
    X: pd.DataFrame,
    y_width: pd.Series,
    y_density: pd.Series,
    y_risk: pd.Series,
    models_dir: Path,
    config: dict[str, Any],
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    系统训练主入口：划分数据 → sklearn Pipeline(标准化+XGBoost) 训练三任务 →
    评估 → 保存推理用拆分 pkl（兼容现有 Web） + outputs/ 下完整 Pipeline 与报表。
    """
    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"特征表缺少列（须与 FEATURE_COLUMNS 一致）: {missing}")

    # 仅使用与推理端一致的列顺序与数值类型
    Xo = X[FEATURE_COLUMNS].astype(np.float64, copy=False)
    y_risk = y_risk.astype(int, copy=False)

    tcfg = config.get("training", {})
    test_size = float(tcfg.get("test_size", 0.2))
    rs = int(tcfg.get("random_state", 42))
    stratify_flag = bool(tcfg.get("stratify_risk", True))

    # 训练集 / 测试集划分（分类任务可按开裂风险分层）
    stratify = y_risk if stratify_flag else None
    try:
        splits = train_test_split(
            Xo,
            y_width,
            y_density,
            y_risk,
            test_size=test_size,
            random_state=rs,
            stratify=stratify,
        )
    except ValueError:
        splits = train_test_split(
            Xo,
            y_width,
            y_density,
            y_risk,
            test_size=test_size,
            random_state=rs,
            stratify=None,
        )

    X_train, X_test, yw_tr, yw_te, yd_tr, yd_te, yr_tr, yr_te = splits

    # XGBoost 超参（与 config/model_config.yaml 一致）
    kw_w = _xgb_params("regressor_width", config, rs)
    kw_d = _xgb_params("regressor_density", config, rs + 1)
    kw_c = _xgb_params("classifier_risk", config, rs + 2)

    # —— Pipeline：一步完成「标准化 + 树模型」，便于导出与复现 ——
    # 注意：三个任务各有一条 Pipeline，但 scaler 均在 X_train 上拟合，均值方差一致；
    # 推理端仍使用单独保存的 feature_scaler.pkl + 三模型 pkl（见下方拆包落盘）。

    pipe_width = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "xgb",
                XGBRegressor(
                    **{k: v for k, v in kw_w.items() if k != "eval_metric"}
                ),
            ),
        ]
    )
    pipe_density = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "xgb",
                XGBRegressor(
                    **{k: v for k, v in kw_d.items() if k != "eval_metric"}
                ),
            ),
        ]
    )
    pipe_risk = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("xgb", XGBClassifier(**kw_c)),
        ]
    )

    pipe_width.fit(X_train, yw_tr)
    pipe_density.fit(X_train, yd_tr)

    n_cls_tr = len(np.unique(yr_tr))
    n_cls_all = len(np.unique(y_risk))
    if n_cls_tr < 2:
        raise ValueError(
            "训练集中「开裂风险」类别少于 2 类，无法训练多分类模型。"
            "请增加样本或调整标签（需同时包含 0/1/2 中至少两类）。"
        )
    if n_cls_all < 2:
        raise ValueError("全数据集开裂风险仅单一类别，请检查标签列 cracking_risk。")

    pipe_risk.fit(X_train, yr_tr)

    # 在原始特征矩阵的测试集上 predict（Pipeline 内部会先 scaler）
    pred_w = pipe_width.predict(X_test)
    pred_d = pipe_density.predict(X_test)
    pred_r = pipe_risk.predict(X_test)

    rmse_w = float(np.sqrt(mean_squared_error(yw_te, pred_w)))
    rmse_d = float(np.sqrt(mean_squared_error(yd_te, pred_d)))
    metrics: dict[str, Any] = {
        "crack_width": {
            "test_mae": float(mean_absolute_error(yw_te, pred_w)),
            "test_rmse": rmse_w,
            "test_r2": float(r2_score(yw_te, pred_w)),
        },
        "crack_density": {
            "test_mae": float(mean_absolute_error(yd_te, pred_d)),
            "test_rmse": rmse_d,
            "test_r2": float(r2_score(yd_te, pred_d)),
        },
        "cracking_risk": {
            "test_accuracy": float(accuracy_score(yr_te, pred_r)),
            "test_macro_f1": float(
                f1_score(yr_te, pred_r, average="macro", zero_division=0)
            ),
            "test_weighted_f1": float(
                f1_score(yr_te, pred_r, average="weighted", zero_division=0)
            ),
        },
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # 从 Pipeline 拆出与历史推理文件名一致的组件（不改 SteelFiberCrackPredictor 加载逻辑）
    reg_w = pipe_width.named_steps["xgb"]
    reg_d = pipe_density.named_steps["xgb"]
    clf = pipe_risk.named_steps["xgb"]
    # 任选一条 Pipeline 的 scaler 作为全局 feature_scaler（三者对 X_train 拟合结果相同）
    scaler = pipe_width.named_steps["scaler"]

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg_w, models_dir / "crack_regressor.pkl")
    joblib.dump(reg_d, models_dir / "crack_density_regressor.pkl")
    joblib.dump(clf, models_dir / "crack_classifier.pkl")
    joblib.dump(scaler, models_dir / "feature_scaler.pkl")

    importance = {
        "crack_width": dict(
            zip(FEATURE_COLUMNS, reg_w.feature_importances_.tolist(), strict=True)
        ),
        "crack_density": dict(
            zip(FEATURE_COLUMNS, reg_d.feature_importances_.tolist(), strict=True)
        ),
        "cracking_risk": dict(
            zip(FEATURE_COLUMNS, clf.feature_importances_.tolist(), strict=True)
        ),
    }
    with open(models_dir / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(importance, f, indent=2, ensure_ascii=False)

    with open(models_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # —— outputs/：完整 Pipeline、指标与特征重要性副本，便于归档与论文材料 ——
    out_root = OUTPUTS_DIR
    out_pipe = out_root / "pipelines"
    out_root.mkdir(parents=True, exist_ok=True)
    out_pipe.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe_width, out_pipe / "pipeline_crack_width.joblib")
    joblib.dump(pipe_density, out_pipe / "pipeline_crack_density.joblib")
    joblib.dump(pipe_risk, out_pipe / "pipeline_cracking_risk.joblib")
    with open(out_root / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_root / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(importance, f, indent=2, ensure_ascii=False)

    if verbose:
        print("--- 测试集指标 ---")
        print(
            f"裂缝宽度: MAE={metrics['crack_width']['test_mae']:.4f} mm, "
            f"RMSE={metrics['crack_width']['test_rmse']:.4f}, "
            f"R2={metrics['crack_width']['test_r2']:.4f}"
        )
        print(
            f"裂缝密度: MAE={metrics['crack_density']['test_mae']:.4f}, "
            f"RMSE={metrics['crack_density']['test_rmse']:.4f}, "
            f"R2={metrics['crack_density']['test_r2']:.4f}"
        )
        print(
            f"开裂风险(0/1/2): 准确率={metrics['cracking_risk']['test_accuracy']:.4f}, "
            f"宏平均 F1={metrics['cracking_risk']['test_macro_f1']:.4f}, "
            f"加权 F1={metrics['cracking_risk']['test_weighted_f1']:.4f}"
        )
        print(f"已保存模型与 scaler 至: {models_dir.resolve()}")
        print(f"Pipeline 与报表已写入: {out_root.resolve()}")

    return metrics
