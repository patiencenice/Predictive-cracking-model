"""
仅针对 crack_width 回归：在固定 XGBoost 算法前提下，系统比较
  - 全特征 / 核心子特征 / 全特征+工程派生特征
  - 是否使用 StandardScaler
并通过训练集上的随机搜索 + 交叉验证调参，在统一 hold-out 测试集上汇报 R² / MAE / RMSE。

用法:
  py optimize_crack_width_r2.py
  py optimize_crack_width_r2.py --csv data/training_data.csv --n-iter 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBRegressor

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import FEATURE_COLUMNS
from src.paths import CONFIG_YAML, OUTPUTS_DIR, PROJECT_ROOT
from src.train_utils import load_model_config

# ---------- 核心特征子集（抗裂机理上更直接相关，用于与全量 28 维对比）----------
CRACK_WIDTH_CORE_FEATURES: tuple[str, ...] = (
    "fiber_content",
    "aspect_ratio",
    "tensile_strength",
    "fiber_content_x_aspect_ratio",
    "w_b_ratio",
    "binder_content",
    "cube_strength_mpa",
    "sand_ratio",
    "mixing_water",
    "curing_days",
    "temperature",
    "humidity",
)

# ---------- 工程派生特征列名（与用户需求一致；避免与已有 sand_ratio 语义混淆的列名冲突）----------
DERIVED_FEATURE_NAMES: tuple[str, ...] = (
    "fiber_factor",  # 掺量×长径比×抗拉，表征纤维阻裂能力量级
    "fly_ash_ratio",  # 粉煤灰占胶材比例
    "slag_ratio",  # 矿粉占胶材比例（slag_powder / binder_content）
    "log_curing_days",  # log(1+龄期)，缓和天数大端的非线性
    "temp_humidity",  # 温湿度交互，反映环境干燥/养护倾向
)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """在已含 FEATURE_COLUMNS 的 DataFrame 上追加派生列；binder 过小处做数值稳定处理。"""
    out = df.copy()
    bc = out["binder_content"].astype(np.float64).values
    # 避免除零：胶材接近 0 时用 nan 再填 0（树模型可处理；若极少出现）
    bc_safe = np.where(bc > 1e-9, bc, np.nan)
    out["fiber_factor"] = (
        out["fiber_content"].astype(np.float64)
        * out["aspect_ratio"].astype(np.float64)
        * out["tensile_strength"].astype(np.float64)
    )
    out["fly_ash_ratio"] = out["fly_ash"].astype(np.float64) / bc_safe
    out["slag_ratio"] = out["slag_powder"].astype(np.float64) / bc_safe
    out["fly_ash_ratio"] = out["fly_ash_ratio"].fillna(0.0)
    out["slag_ratio"] = out["slag_ratio"].fillna(0.0)
    out["log_curing_days"] = np.log1p(out["curing_days"].astype(np.float64))
    out["temp_humidity"] = (
        out["temperature"].astype(np.float64) * out["humidity"].astype(np.float64)
    )
    return out


def build_feature_matrix(
    df: pd.DataFrame, mode: str
) -> tuple[pd.DataFrame, list[str]]:
    """
    mode:
      - full: 与线上一致的 FEATURE_COLUMNS
      - core: CRACK_WIDTH_CORE_FEATURES 子集
      - extended: FEATURE_COLUMNS + 派生特征
    """
    base = df[FEATURE_COLUMNS].astype(np.float64, copy=False)
    if mode == "full":
        return base, list(FEATURE_COLUMNS)
    if mode == "core":
        cols = list(CRACK_WIDTH_CORE_FEATURES)
        return base[cols], cols
    if mode == "extended":
        ext = add_derived_features(base)
        cols = list(FEATURE_COLUMNS) + list(DERIVED_FEATURE_NAMES)
        return ext[cols], cols
    raise ValueError(f"未知 mode: {mode}")


def make_width_pipeline(use_scaler: bool, random_state: int) -> Pipeline:
    """
    统一用 Pipeline 包一层，便于 RandomizedSearchCV 同时搜索是否标准化：
    - use_scaler=True：StandardScaler
    - use_scaler=False：恒等变换（与「不用 scaler」等价）
    """
    if use_scaler:
        prep: Any = StandardScaler()
    else:
        # 恒等映射，保持接口与参数前缀 reg__ 一致
        prep = FunctionTransformer()
    reg = XGBRegressor(
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        reg_lambda=1.0,
    )
    return Pipeline([("prep", prep), ("reg", reg)])


def param_distributions_for_search() -> dict[str, Any]:
    """XGBoost 回归调参分布（与需求中的超参一一对应）。"""
    return {
        "reg__n_estimators": randint(80, 420),
        "reg__max_depth": randint(3, 10),
        "reg__learning_rate": uniform(0.02, 0.18),
        "reg__min_child_weight": randint(1, 12),
        "reg__subsample": uniform(0.55, 0.45),
        "reg__colsample_bytree": uniform(0.55, 0.45),
    }


def evaluate_on_test(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    """在固定测试集上计算 R²、MAE、RMSE。"""
    pred = model.predict(X_test)
    return {
        "test_r2": float(r2_score(y_test, pred)),
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_rmse": float(
            np.sqrt(mean_squared_error(y_test, pred))
        ),
    }


def main() -> None:
    # Windows 控制台默认 GBK，避免打印中文说明时因上标等字符报错
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="crack_width 回归 R² 系统优化实验")
    ap.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "training_data.csv",
        help="含 FEATURE_COLUMNS 与 crack_width 的 CSV",
    )
    ap.add_argument("--n-iter", type=int, default=35, help="随机搜索每方案迭代次数")
    ap.add_argument("--cv", type=int, default=5, help="交叉验证折数")
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="划分与模型随机种子（与 train_utils 默认一致便于对比）",
    )
    args = ap.parse_args()

    if not args.csv.exists():
        ex = PROJECT_ROOT / "data" / "training_data.example.csv"
        raise SystemExit(f"未找到 {args.csv}，可使用示例: {ex}")

    cfg = load_model_config(CONFIG_YAML)
    tcfg = cfg.get("training", {})
    test_size = float(tcfg.get("test_size", 0.2))
    rs = int(tcfg.get("random_state", args.random_state))
    stratify_flag = bool(tcfg.get("stratify_risk", True))

    df = pd.read_csv(args.csv)
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            raise SystemExit(f"CSV 缺少特征列: {c}")
    if "crack_width" not in df.columns:
        raise SystemExit("CSV 缺少目标列 crack_width")
    if "cracking_risk" not in df.columns:
        raise SystemExit("CSV 缺少 cracking_risk（用于与训练一致的 stratify 划分）")

    y = df["crack_width"].astype(np.float64)
    y_risk = df["cracking_risk"].astype(int)
    stratify = y_risk if stratify_flag else None
    try:
        idx_train, idx_test = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=rs,
            stratify=stratify,
        )
    except ValueError:
        idx_train, idx_test = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=rs,
            stratify=None,
        )

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)
    y_train = y.iloc[idx_train].values
    y_test = y.iloc[idx_test].values

    rows: list[dict[str, Any]] = []
    modes = ("full", "core", "extended")
    scaler_flags = (True, False)

    param_dist = param_distributions_for_search()

    # 注意：XGBoost 单棵树按特征值排序后选分裂点；逐特征标准化是单调仿射变换，
    # 不改变样本在各特征上的相对顺序，因此常与「不缩放」得到等价的树结构与预测。
    # 本脚本仍保留 scaler 对比以满足实验设计；若两行指标完全一致，属预期现象。

    for mode in modes:
        X_tr_df, col_names = build_feature_matrix(df_train, mode)
        X_te_df, _ = build_feature_matrix(df_test, mode)
        X_train_m = X_tr_df.values
        X_test_m = X_te_df.values

        for use_scaler in scaler_flags:
            pipe = make_width_pipeline(use_scaler, rs)
            # 训练集内随机搜索 + K 折交叉验证，以 R² 为优化目标
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_dist,
                n_iter=args.n_iter,
                cv=args.cv,
                scoring="r2",
                random_state=rs,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            search.fit(X_train_m, y_train)
            best = search.best_estimator_
            metrics = evaluate_on_test(best, X_test_m, y_test)
            row = {
                "experiment_kind": "random_search_cv",
                "feature_mode": mode,
                "use_standard_scaler": use_scaler,
                "n_features": len(col_names),
                "feature_list": col_names,
                "best_cv_r2_mean": float(search.best_score_),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                **metrics,
                "best_params": {
                    k.replace("reg__", ""): v
                    for k, v in search.best_params_.items()
                    if k.startswith("reg__")
                },
            }
            rows.append(row)

    # ---------- 基线：YAML 默认超参 + 全特征 + StandardScaler（不调参，便于对照）----------
    cfg_w = cfg.get("xgboost", {}).get("regressor_width", {})
    base_params = {
        "n_estimators": int(cfg_w.get("n_estimators", 280)),
        "max_depth": int(cfg_w.get("max_depth", 5)),
        "learning_rate": float(cfg_w.get("learning_rate", 0.06)),
        "subsample": float(cfg_w.get("subsample", 0.88)),
        "colsample_bytree": float(cfg_w.get("colsample_bytree", 0.88)),
        "reg_lambda": float(cfg_w.get("reg_lambda", 1.0)),
        "min_child_weight": int(cfg_w.get("min_child_weight", 1)),
    }
    X_tr_full, _ = build_feature_matrix(df_train, "full")
    X_te_full, _ = build_feature_matrix(df_test, "full")
    scaler_b = StandardScaler()
    X_tr_s = scaler_b.fit_transform(X_tr_full.values)
    X_te_s = scaler_b.transform(X_te_full.values)
    baseline_reg = XGBRegressor(
        random_state=rs,
        n_jobs=-1,
        tree_method="hist",
        **base_params,
    )
    baseline_reg.fit(X_tr_s, y_train)
    base_metrics = evaluate_on_test(baseline_reg, X_te_s, y_test)
    rows.append(
        {
            "experiment_kind": "yaml_baseline",
            "feature_mode": "full",
            "use_standard_scaler": True,
            "n_features": len(FEATURE_COLUMNS),
            "feature_list": list(FEATURE_COLUMNS),
            "best_cv_r2_mean": None,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            **base_metrics,
            "best_params": {**base_params, "note": "yaml_baseline_no_search"},
        }
    )

    # ---------- 汇总表：按测试集 R2 排序（含 yaml 基线）----------
    tuned = [r for r in rows if r.get("best_cv_r2_mean") is not None]
    tuned.sort(key=lambda x: x["test_r2"], reverse=True)
    best = tuned[0] if tuned else rows[-1]
    global_best = max(rows, key=lambda x: x["test_r2"])

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUTPUTS_DIR / "crack_width_optimize_table.csv"
    report_path = OUTPUTS_DIR / "crack_width_optimize_report.json"

    # 扁平化便于 CSV（特征列表用分号连接）
    flat = []
    for r in rows:
        fp = r["best_params"]
        flat.append(
            {
                "experiment_kind": r.get("experiment_kind", ""),
                "feature_mode": r["feature_mode"],
                "use_standard_scaler": r["use_standard_scaler"],
                "n_features": r["n_features"],
                "best_cv_r2_mean": r["best_cv_r2_mean"],
                "test_r2": r["test_r2"],
                "test_mae": r["test_mae"],
                "test_rmse": r["test_rmse"],
                "best_params_json": json.dumps(
                    fp, ensure_ascii=False, default=str
                ),
            }
        )
    pd.DataFrame(flat).to_csv(table_path, index=False, encoding="utf-8-sig")

    # 推荐语：避免特殊 Unicode，便于 Windows 默认编码下打印
    recommendation = (
        "在「相同 hold-out 测试集、仅 crack_width」前提下，"
        f"随机搜索+交叉验证方案中测试集 R2 最高为：feature_mode={best['feature_mode']}, "
        f"use_standard_scaler={best['use_standard_scaler']}, "
        f"test_r2={best['test_r2']:.6f}, test_mae={best['test_mae']:.6f}, "
        f"test_rmse={best['test_rmse']:.6f}。"
        " 关于 StandardScaler：对基于排序分裂的 XGBoost 树模型，逐特征标准化通常不改变分裂结构，"
        "故 scaler=True/False 的测试指标可能完全相同；若相同，可任选其一部署，不必强行保留 scaler。"
        " 请同时对照 yaml 基线行（yaml_baseline）：若小样本下单次划分波动大，建议结合 K 折或更多数据再定稿。"
        f" 本次全部实验（含基线）中测试集 R2 最高者为 {global_best.get('experiment_kind')}"
        f" / feature_mode={global_best['feature_mode']}, test_r2={global_best['test_r2']:.6f}。"
    )

    scaler_invariance_note_cn = (
        "对梯度提升树（按排序选分裂阈值），逐特征单调变换（含 StandardScaler）"
        "一般不改变样本在轴上的先后次序，故可能与未缩放得到相同预测与指标。"
    )

    report = {
        "csv": str(args.csv.resolve()),
        "split": {"test_size": test_size, "random_state": rs},
        "search": {"n_iter": args.n_iter, "cv": args.cv},
        "derived_features": list(DERIVED_FEATURE_NAMES),
        "core_features": list(CRACK_WIDTH_CORE_FEATURES),
        "rows": rows,
        "best_random_search_by_test_r2": {
            "feature_mode": best["feature_mode"],
            "use_standard_scaler": best["use_standard_scaler"],
            "test_r2": best["test_r2"],
            "test_mae": best["test_mae"],
            "test_rmse": best["test_rmse"],
            "best_params": best["best_params"],
        },
        "global_best_by_test_r2": {
            "experiment_kind": global_best.get("experiment_kind"),
            "feature_mode": global_best["feature_mode"],
            "use_standard_scaler": global_best["use_standard_scaler"],
            "test_r2": global_best["test_r2"],
            "test_mae": global_best["test_mae"],
            "test_rmse": global_best["test_rmse"],
        },
        "recommendation_cn": recommendation,
        "scaler_invariance_note_cn": scaler_invariance_note_cn,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # 控制台打印对比表（关键结果）
    print("\n=== crack_width 优化实验：测试集指标对比（含基线）===\n")
    print(
        pd.DataFrame(flat)[
            [
                "experiment_kind",
                "feature_mode",
                "use_standard_scaler",
                "best_cv_r2_mean",
                "test_r2",
                "test_mae",
                "test_rmse",
            ]
        ].to_string(index=False)
    )
    print(f"\n{recommendation}\n")
    print(f"已保存: {table_path.resolve()}")
    print(f"已保存: {report_path.resolve()}")


if __name__ == "__main__":
    main()
