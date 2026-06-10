"""
按分组列做 GroupKFold（优先 `source_group`，缺省或全空时回退 `source_doi`），
仅评估 crack_width 回归（优先指标：CV 平均 R²）。

三组策略：only_user / only_literature / merged；比较 CV mean R² 并写出 leaderboard。

不修改 train_model.py；本脚本独立用于文献增强实验。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import FEATURE_COLUMNS
from src.literature_pipeline.constants import USER_SOURCE_DOI_PLACEHOLDER
from src.literature_pipeline.crack_width_definition_filter import (
    definition_id_counts,
    emit_mixed_warning_if_needed,
    ensure_crack_width_definition_id_column,
    filter_by_crack_width_family,
    mixed_definition_warning_text,
)
from src.paths import CONFIG_YAML, OUTPUTS_DIR, PROJECT_ROOT
from src.train_utils import load_model_config, _xgb_params


def _resolve_cv_group_labels(df: pd.DataFrame) -> tuple[np.ndarray, str, str]:
    """
    GroupKFold 使用的分组标签：
    1) 若存在 source_group 且不全空 → 用 source_group，空位用 source_doi 回填；
    2) 否则整表使用 source_doi。
    用户本地可用 USER_LOCAL_LAB/BATCH_001 等形式区分批次。
    """
    doi = df["source_doi"].astype(str).str.strip()
    if "source_group" not in df.columns:
        return doi.values, "source_doi", "无 source_group 列，使用 source_doi"
    sg = df["source_group"].astype(str).str.strip()
    empty = sg.isna() | (sg == "") | (sg == "nan")
    if bool(empty.all()):
        return doi.values, "source_doi", "source_group 全空，回退 source_doi"
    out = sg.where(~empty, doi)
    return out.values, "source_group", "source_group 优先，空位回退 source_doi"


def _prepare_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, str, str]:
    """提取特征矩阵、y、groups、sample_weight，以及分组列说明。"""
    X = df[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["crack_width"], errors="coerce")
    mask = y.notna() & X.notna().all(axis=1)
    X = X.loc[mask]
    y = y.loc[mask].values
    sub = df.loc[mask]
    groups, gcol, gnote = _resolve_cv_group_labels(sub)
    if "sample_weight" in df.columns:
        sw = pd.to_numeric(df.loc[mask, "sample_weight"], errors="coerce").fillna(1.0).values
    else:
        sw = np.ones(len(y), dtype=np.float64)
    return X, y, groups, sw, gcol, gnote


def _filter_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "only_user":
        return df[df["source_doi"].astype(str) == USER_SOURCE_DOI_PLACEHOLDER].copy()
    if strategy == "only_literature":
        return df[df["source_doi"].astype(str) != USER_SOURCE_DOI_PLACEHOLDER].copy()
    if strategy == "merged":
        return df.copy()
    raise ValueError(strategy)


def cv_groupkfold_crack_width(
    df: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """返回各折 R²/MAE/RMSE 及均值方差；样本过少或单组时返回 None。"""
    X, y, groups, sw, gcol, gnote = _prepare_xy(df)
    # 至少 2 条才可作回归；GroupKFold 至少需 2 个分组（见下方 n_g 判断）
    if len(y) < 2:
        return None
    n_g = len(np.unique(groups))
    # 组数不足则无法分组（避免虚假分组）
    if n_g < 2:
        return {
            "error": (
                f"唯一分组（列={gcol}）少于 2 组，无法做 GroupKFold；"
                "请补充多来源文献、或为本地数据设置多个 source_group（如 USER_LOCAL_LAB/BATCH_001）。"
            ),
            "n_samples": int(len(y)),
            "n_groups": int(n_g),
            "grouping_column": gcol,
            "grouping_note": gnote,
        }

    n_splits_eff = min(n_splits, n_g)
    gkf = GroupKFold(n_splits=n_splits_eff)

    kw = _xgb_params("regressor_width", config, random_state)
    kw = {k: v for k, v in kw.items() if k != "eval_metric"}

    r2_list: list[float] = []
    mae_list: list[float] = []
    rmse_list: list[float] = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        sw_tr = sw[train_idx]
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("xgb", XGBRegressor(**kw)),
            ]
        )
        # 样本权重传入 XGBoost 回归器
        pipe.fit(X_tr, y_tr, xgb__sample_weight=sw_tr)
        pred = pipe.predict(X_te)
        # 单点验证集无法定义 R²，记为 nan，汇总时用 nanmean
        if len(y_te) < 2:
            r2_list.append(float("nan"))
        else:
            r2_list.append(float(r2_score(y_te, pred)))
        mae_list.append(float(mean_absolute_error(y_te, pred)))
        rmse_list.append(
            float(np.sqrt(mean_squared_error(y_te, pred)))
        )

    r2_arr = np.array(r2_list, dtype=np.float64)
    return {
        "n_samples": int(len(y)),
        "n_groups": int(n_g),
        "n_splits": int(n_splits_eff),
        "grouping_column": gcol,
        "grouping_note": gnote,
        "distinct_groups": sorted({str(x) for x in np.unique(groups)}),
        "r2_mean": float(np.nanmean(r2_arr)),
        "r2_std": float(np.nanstd(r2_arr)),
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "folds": {
            "r2": r2_list,
            "mae": mae_list,
            "rmse": rmse_list,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--merged-csv",
        type=Path,
        default=OUTPUTS_DIR / "training_data_merged.csv",
    )
    ap.add_argument(
        "--user-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "training_data.csv",
    )
    ap.add_argument(
        "--literature-csv",
        type=Path,
        default=OUTPUTS_DIR / "literature_training_data.csv",
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--crack-width-family",
        type=str,
        default=None,
        metavar="ID",
        help="仅保留该 crack_width_definition_id 的样本后再做 CV（与 run_literature_pipeline 一致）",
    )
    args = ap.parse_args()

    cfg = load_model_config(CONFIG_YAML)

    # 若尚无 merged，尝试从 user + literature 现场合并
    if args.merged_csv.exists():
        df_merged = pd.read_csv(args.merged_csv)
    elif args.user_csv.exists() and args.literature_csv.exists():
        from src.literature_pipeline.merge_datasets import merge_user_and_literature

        df_merged = merge_user_and_literature(args.user_csv, args.literature_csv)
    else:
        raise SystemExit(
            "缺少 training_data_merged.csv，且无法从 user+literature 合并；请先运行 run_literature_pipeline.py"
        )

    df_merged = ensure_crack_width_definition_id_column(df_merged)
    counts_pre_def = definition_id_counts(df_merged)
    if not args.crack_width_family:
        emit_mixed_warning_if_needed(counts_pre_def)
    df_merged, def_stats = filter_by_crack_width_family(
        df_merged, args.crack_width_family
    )
    print(
        f"[crack_width_definition_id] 过滤前: {def_stats['n_before']} -> "
        f"过滤后: {def_stats['n_after']}（剔除 {def_stats['dropped']}）"
    )
    if len(df_merged) == 0:
        raise SystemExit(
            "过滤后无样本：请调整 --crack-width-family 或补充对应 crack_width_definition_id 的数据。"
        )
    mix_warn_txt = (
        mixed_definition_warning_text(counts_pre_def)
        if not args.crack_width_family
        else None
    )
    definition_filter_meta: dict[str, Any] = {
        "crack_width_family": def_stats.get("family"),
        "n_samples_before_definition_filter": def_stats["n_before"],
        "n_samples_after_definition_filter": def_stats["n_after"],
        "dropped_by_definition_filter": def_stats["dropped"],
        "crack_width_definition_id_counts_before_filter": def_stats["counts_before"],
        "crack_width_definition_id_counts_after_filter": def_stats["counts_after"],
        "mixed_definition_warning": mix_warn_txt,
    }

    strategies = ("only_user", "only_literature", "merged")
    results: dict[str, Any] = {}
    leaderboard_rows: list[dict[str, Any]] = []

    for st in strategies:
        sub = _filter_strategy(df_merged, st)
        res = cv_groupkfold_crack_width(
            sub, n_splits=args.n_splits, random_state=args.random_state, config=cfg
        )
        results[st] = res
        if res and "error" in res:
            leaderboard_rows.append(
                {
                    "strategy": st,
                    "status": "error",
                    "note": res.get("error", ""),
                    "n_samples": res.get("n_samples"),
                    "n_groups": res.get("n_groups"),
                    "grouping_column": res.get("grouping_column"),
                    "grouping_note": res.get("grouping_note"),
                    "crack_width_family": definition_filter_meta["crack_width_family"],
                    "n_samples_before_definition_filter": definition_filter_meta[
                        "n_samples_before_definition_filter"
                    ],
                    "n_samples_after_definition_filter": definition_filter_meta[
                        "n_samples_after_definition_filter"
                    ],
                    "dropped_by_definition_filter": definition_filter_meta[
                        "dropped_by_definition_filter"
                    ],
                }
            )
        elif res is None:
            leaderboard_rows.append(
                {
                    "strategy": st,
                    "status": "skipped",
                    "note": "样本过少",
                    "crack_width_family": definition_filter_meta["crack_width_family"],
                    "n_samples_before_definition_filter": definition_filter_meta[
                        "n_samples_before_definition_filter"
                    ],
                    "n_samples_after_definition_filter": definition_filter_meta[
                        "n_samples_after_definition_filter"
                    ],
                    "dropped_by_definition_filter": definition_filter_meta[
                        "dropped_by_definition_filter"
                    ],
                }
            )
        else:
            leaderboard_rows.append(
                {
                    "strategy": st,
                    "status": "ok",
                    "n_samples": res["n_samples"],
                    "n_groups": res["n_groups"],
                    "n_splits": res["n_splits"],
                    "grouping_column": res.get("grouping_column"),
                    "grouping_note": res.get("grouping_note"),
                    "cv_r2_mean": res["r2_mean"],
                    "cv_r2_std": res["r2_std"],
                    "cv_mae_mean": res["mae_mean"],
                    "cv_mae_std": res["mae_std"],
                    "cv_rmse_mean": res["rmse_mean"],
                    "cv_rmse_std": res["rmse_std"],
                    "crack_width_family": definition_filter_meta["crack_width_family"],
                    "n_samples_before_definition_filter": definition_filter_meta[
                        "n_samples_before_definition_filter"
                    ],
                    "n_samples_after_definition_filter": definition_filter_meta[
                        "n_samples_after_definition_filter"
                    ],
                    "dropped_by_definition_filter": definition_filter_meta[
                        "dropped_by_definition_filter"
                    ],
                }
            )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    lb_path = OUTPUTS_DIR / "leaderboard.csv"
    pd.DataFrame(leaderboard_rows).to_csv(lb_path, index=False, encoding="utf-8-sig")

    cv_path = OUTPUTS_DIR / "cv_report_groupkfold.json"

    def _json_safe(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _json_safe(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_json_safe(v) for v in x]
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        return x

    out_cv = {
        "definition_filter": definition_filter_meta,
        **results,
    }
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(out_cv), f, indent=2, ensure_ascii=False)

    # 最优策略（按 cv_r2_mean）
    ok_rows = [r for r in leaderboard_rows if r.get("status") == "ok"]
    best = max(ok_rows, key=lambda x: x.get("cv_r2_mean", -1e9)) if ok_rows else None
    summary_path = OUTPUTS_DIR / "best_model_summary.md"
    fam_line = definition_filter_meta.get("crack_width_family")
    lines = [
        "# crack_width GroupKFold 结果摘要",
        "",
        f"- 本次 `crack_width_definition_id` 过滤："
        f"`{fam_line}`（未指定则为 null，使用全部分类口径）",
        f"- 过滤前/后样本数：{definition_filter_meta['n_samples_before_definition_filter']} / "
        f"{definition_filter_meta['n_samples_after_definition_filter']}",
        "- 优先指标：**CV 平均 R²**（分组列：**优先 `source_group`，否则 `source_doi`**，避免同组泄漏）",
        "- 折数上限：`min(n_splits, 不同分组数)`",
        "",
    ]
    if best:
        lines.append(f"- **推荐策略（本次运行）**：`{best['strategy']}`")
        lines.append(f"  - CV R² mean = {best['cv_r2_mean']:.6f} ± {best['cv_r2_std']:.6f}")
        lines.append(
            f"  - CV MAE mean = {best['cv_mae_mean']:.6f}；CV RMSE mean = {best['cv_rmse_mean']:.6f}"
        )
    else:
        lines.append(
            "- 无可用策略完成分组交叉验证，请检查数据量与 `source_group` / `source_doi` 列。"
        )
    lines.append("")
    lines.append("完整折间指标见 `outputs/cv_report_groupkfold.json`。")
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(pd.DataFrame(leaderboard_rows).to_string(index=False))
    print(f"\n已保存: {lb_path.resolve()}\n{cv_path.resolve()}\n{summary_path.resolve()}")


if __name__ == "__main__":
    main()
