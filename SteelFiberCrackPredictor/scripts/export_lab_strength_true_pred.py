"""
导出抗压/抗折强度的实测值与预测值（OOF，默认方法与论文图 7-5 一致）。

数据来源：outputs/lab_strength/lab_strength_oof_predictions.csv
默认残差模型：outputs/lab_strength/lab_strength_residual_report.json → default_method_by_task

用法：
  py scripts/export_lab_strength_true_pred.py
  py scripts/export_lab_strength_true_pred.py --out outputs/lab_strength
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF_CSV = PROJECT_ROOT / "outputs" / "lab_strength" / "lab_strength_oof_predictions.csv"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "outputs" / "lab_strength" / "lab_strength_residual_report.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "lab_strength"

TASK_CN = {"compressive": "抗压", "flexural": "抗折"}


def _load_default_residual_models(report_path: Path) -> dict[str, str]:
    with report_path.open(encoding="utf-8") as f:
        rep = json.load(f)
    dm = rep.get("default_method_by_task") or {}
    out: dict[str, str] = {}
    for task in ("compressive", "flexural"):
        block = dm.get(task) or {}
        learner = block.get("residual_learner")
        out[task] = "formula_only" if learner in (None, "", "null") else str(learner)
    return out


def _task_export_frame(
    df: pd.DataFrame, task: str, residual_model: str
) -> pd.DataFrame:
    sub = df[(df["task"] == task) & (df["residual_model"] == residual_model)].copy()
    if sub.empty:
        raise ValueError(f"OOF 数据中无 task={task!r}, residual_model={residual_model!r} 记录")
    sub = sub.sort_values("row_id").drop_duplicates(subset=["row_id"], keep="first")
    return pd.DataFrame(
        {
            "row_id": sub["row_id"].astype(int),
            "source_group": sub["source_group"],
            "fold_id": sub["fold_id"].astype(int),
            "实测值_MPa": sub["y_true"],
            "公式基线_MPa": sub["formula_pred"],
            "残差修正_MPa": sub["residual_pred"],
            "预测值_MPa": sub["final_pred"],
            "残差模型": residual_model,
        }
    )


def export_true_pred(
    oof_csv: Path,
    report_json: Path,
    out_dir: Path,
) -> dict[str, Path]:
    df = pd.read_csv(oof_csv)
    models = _load_default_residual_models(report_json)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    task_frames: dict[str, pd.DataFrame] = {}

    for task in ("compressive", "flexural"):
        model = models[task]
        exp = _task_export_frame(df, task, model)
        task_frames[task] = exp
        path = out_dir / f"lab_strength_{task}_true_pred.csv"
        exp.to_csv(path, index=False, encoding="utf-8-sig")
        paths[task] = path

    def _rename_task_cols(frame: pd.DataFrame, cn: str) -> pd.DataFrame:
        return frame.rename(
            columns={
                "实测值_MPa": f"{cn}_实测值_MPa",
                "预测值_MPa": f"{cn}_预测值_MPa",
                "公式基线_MPa": f"{cn}_公式基线_MPa",
                "残差修正_MPa": f"{cn}_残差修正_MPa",
                "残差模型": f"{cn}_残差模型",
            }
        )

    comp_w = _rename_task_cols(task_frames["compressive"], TASK_CN["compressive"])
    flex_w = _rename_task_cols(task_frames["flexural"], TASK_CN["flexural"])
    wide = comp_w.merge(
        flex_w,
        on=["row_id", "source_group", "fold_id"],
        how="outer",
    )
    wide_path = out_dir / "lab_strength_compressive_flexural_true_pred.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
    paths["combined"] = wide_path

    meta = {
        "oof_csv": str(oof_csv.resolve()),
        "report_json": str(report_json.resolve()),
        "default_residual_model_by_task": models,
        "n_rows_compressive": int((df["task"] == "compressive").sum() // 4),
        "export_files": {k: str(v.resolve()) for k, v in paths.items()},
    }
    meta_path = out_dir / "lab_strength_true_pred_export_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    paths["meta"] = meta_path
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="导出抗压/抗折实测值与预测值 CSV")
    ap.add_argument("--oof", type=Path, default=DEFAULT_OOF_CSV)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT_JSON)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    paths = export_true_pred(args.oof, args.report, args.out)
    models = json.loads(paths["meta"].read_text(encoding="utf-8"))[
        "default_residual_model_by_task"
    ]
    print("已导出（默认 OOF 方法，与图 7-5 一致）：")
    print(f"  抗压 residual_model={models['compressive']}: {paths['compressive']}")
    print(f"  抗折 residual_model={models['flexural']}: {paths['flexural']}")
    print(f"  合并宽表: {paths['combined']}")
    print(f"  元数据: {paths['meta']}")


if __name__ == "__main__":
    main()
