"""
抗压/抗折：公式基线（国标） + 残差模型 — 训练与交叉验证评估。

用法（在 SteelFiberCrackPredictor 目录下）:
  py scripts/build_lab_strength_example_csv.py   # 可选：重新生成示例训练集
  py train_lab_strength_residual.py
  py train_lab_strength_residual.py --data data/lab_strength_training.example.csv --out outputs/lab_strength --save-models

数据 CSV 需包含 lab_strength 特征列（见 src.lab_strength_residual.lab_mix_features.LAB_STRENGTH_FEATURE_COLUMNS），以及:
  - compressive_true, flexural_true  (MPa)
可选:
  - source_group  (字符串，有则优先 GroupKFold)
  - lab_specimen, lab_cube_edge_mm, lab_prism_*, lab_beam_*, lab_loading_* （缺省见 src.lab_strength_residual.dataset.LAB_DEFAULTS）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from src.lab_strength_residual.train_eval import run_pipeline


def _print_oof_formula_hgb(rep: dict) -> None:
    """终端输出：抗压/抗折的 formula_only 与 hgb 的 OOF MAE、R²。"""
    for task in ("compressive", "flexural"):
        oof = rep[task]["oof_global_metrics"]
        fo = oof["formula_only"]
        hg = oof["hgb"]
        print(
            f"  [{task}] formula_only  OOF MAE={fo['mae']:.6f}  R2={fo['r2']:.6f}  |  "
            f"hgb  OOF MAE={hg['mae']:.6f}  R2={hg['r2']:.6f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="抗压/抗折 公式+残差 训练与评估")
    p.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data" / "lab_strength_training.example.csv",
        help="训练数据 CSV",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "outputs" / "lab_strength",
        help="报告与模型输出目录",
    )
    p.add_argument(
        "--save-models",
        action="store_true",
        help="在全量数据上拟合并保存 joblib",
    )
    p.add_argument(
        "--top-k-worst",
        type=int,
        default=5,
        help="OOF 诊断中输出的最差样本条数",
    )
    p.add_argument(
        "--no-source-domain",
        action="store_true",
        help="不在特征矩阵末尾追加 source_domain（默认追加）",
    )
    p.add_argument(
        "--no-sample-weights",
        action="store_true",
        help="训练残差学习器时不使用 sample_weight（默认使用）",
    )
    args = p.parse_args()
    rep = run_pipeline(
        args.data,
        args.out,
        save_models=args.save_models,
        top_k_worst=args.top_k_worst,
        append_source_domain=not args.no_source_domain,
        use_sample_weight=not args.no_sample_weights,
    )
    print("--- OOF 指标（formula_only vs hgb）---")
    _print_oof_formula_hgb(rep)
    print("--- 固化默认策略 default_method_by_task ---")
    dm = rep.get("default_method_by_task") or {}
    print(" ", dm)
    print(
        f"  practical_gain={rep.get('practical_gain')}  "
        f"residual_not_recommended={rep.get('residual_not_recommended')}"
    )
    print("--- 入模行数（与 JSON 顶栏一致）---")
    print(f"  n_rows_csv={rep.get('n_rows_csv')}")
    print(f"  n_rows_after_task_filter={rep.get('n_rows_after_task_filter')}")
    print(
        "  n_rows_dropped_due_to_missing_before_patch_equivalent="
        f"{rep.get('n_rows_dropped_due_to_missing_before_patch_equivalent')}"
    )
    print(f"  n_rows_final_used={rep.get('n_rows_final_used')}")
    print(
        "  fiber_type_missing_rows_used="
        f"{rep.get('fiber_type_missing_rows_used')}"
    )
    wr = rep.get("water_reducer_feature_summary") or {}
    print("--- 减水剂特征入模语义统计 water_reducer_feature_summary ---")
    if isinstance(wr, dict) and "error" not in wr:
        print(
            f"  n_rows_csv={wr.get('n_rows_csv')}  known={wr.get('n_rows_water_reducer_known')}  "
            f"unknown={wr.get('n_rows_water_reducer_unknown')}  zero_rate={wr.get('n_rows_water_reducer_zero')}  "
            f"adjusted_ok={wr.get('n_rows_adjusted_w_b_ratio_available')}  "
            f"adjusted_missing={wr.get('n_rows_adjusted_w_b_ratio_missing')}"
        )
        print(
            "  减水剂类型(语义): none="
            f"{wr.get('n_rows_water_reducer_type_none')}  unknown="
            f"{wr.get('n_rows_water_reducer_type_unknown')}  known="
            f"{wr.get('n_rows_water_reducer_type_known')}"
        )
        print(
            "  source_groups_all_unknown:",
            wr.get("source_groups_all_water_reducer_unknown"),
        )
        print(
            "  source_groups_with_some_known:",
            wr.get("source_groups_with_some_water_reducer_known"),
        )
    else:
        print(" ", wr)

    wl = rep.get("water_reducer_learnability") or {}
    if isinstance(wl, dict) and wl.get("n_rows_analyzed"):
        c = wl.get("conclusions") or {}
        print("--- 减水剂可学习性 water_reducer_learnability（只读统计）---")
        print(
            f"  n_analyzed={wl.get('n_rows_analyzed')}  "
            f"type_learnable={c.get('type_dimension_learnable')}  "
            f"rate_nonzero_var_learnable={c.get('rate_dimension_learnable_nonzero_variation')}"
        )
        print(" ", c.get("honest_summary_zh"))
        print(" ", c.get("effect_vs_missing_pattern_zh"))

    wa = rep.get("water_reducer_feature_ablation") or {}
    if isinstance(wa, dict) and wa.get("delta_omit_minus_full"):
        print("--- 减水剂六列消融 water_reducer_feature_ablation（OOF，omit 减 full）---")
        d = wa["delta_omit_minus_full"]
        for task in ("compressive", "flexural"):
            td = d.get(task) or {}
            print(
                f"  [{task}] formula ΔMAE={td.get('formula_only', {}).get('mae')}  "
                f"ridge ΔMAE={td.get('ridge', {}).get('mae')}  hgb ΔMAE={td.get('hgb', {}).get('mae')}"
            )
        print(" ", wa.get("honest_verdict_zh"))

    n_c = rep["compressive"]["cv_fold_metrics"]["n_samples"]
    n_f = rep["flexural"]["cv_fold_metrics"]["n_samples"]
    print(f"  抗压 cv n_samples={n_c}，抗折 cv n_samples={n_f}")
    gw = rep.get("group_worst_mae_by_task") or {}
    print("--- 按组 MAE 最大（formula_only）---")
    for t in ("compressive", "flexural"):
        wf = (gw.get("formula_only") or {}).get(t)
        if wf:
            print(f"  [{t}] source_group={wf.get('source_group')}  MAE={wf.get('mae')}")
    print("完成。报告:", (args.out / "lab_strength_residual_report.json").resolve())
    if args.save_models:
        print("模型目录:", args.out.resolve())


if __name__ == "__main__":
    main()
