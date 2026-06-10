"""
对 Blank-C50 / SF-C50 / BF-C50 / PPF-C50 四组执行真实模型预测（不重训）。
输出论文用表格 CSV 与终端摘要。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_processor import validate_and_transform
from src.lab_strength_residual.dataset import (
    LAB_DEFAULTS,
    _one_row_feature_vector_with_reason,
    row_compressive_formula_prediction,
    row_flexural_formula_prediction,
)
from src.paths import MODELS_DIR
from src.predictor import SteelFiberCrackPredictor

LAB_REPORT = _ROOT / "outputs" / "lab_strength" / "lab_strength_residual_report.json"
LAB_MERGED = _ROOT / "data" / "lab_strength_training_merged.csv"
OUT_DIR = _ROOT / "outputs" / "paper_c50_predictions"
OUT_CSV = OUT_DIR / "c50_yimeng_prediction_table.csv"

# 论文 C50 四组 → source_group（徐坤易梦 28d）
SAMPLES = {
    "Blank-C50": {
        "source_group": "XUkun/C38_BASE",
        "mix_label": "基准（无纤维）",
        "fiber_material": "钢纤维",  # enc=0 占位；掺量=0，见 prepared_review 说明
        "fiber_type": "端钩型",
        "fiber_content": 0.0,
        "aspect_ratio": 0.0,
        "tensile_strength": 0.0,
        "test_flexural": 6.81,
        "test_compressive": 59.53,
        "test_crack_mm": 0.36,
    },
    "SF-C50": {
        "source_group": "XUkun/C39_BASE",
        "mix_label": "钢纤维",
        "fiber_material": "钢纤维",
        "fiber_type": "端钩型",
        "fiber_content": 0.318471,
        "aspect_ratio": 46.666667,
        "tensile_strength": 1200.0,
        "test_flexural": 6.96,
        "test_compressive": 62.00,
        "test_crack_mm": 0.18,
    },
    "BF-C50": {
        "source_group": "XUkun/C40_BASE",
        "mix_label": "玄武岩纤维",
        "fiber_material": "玄武岩纤维",
        "fiber_type": "端钩型",
        "fiber_content": 0.150943,
        "aspect_ratio": 705.882353,
        "tensile_strength": 2600.0,
        "test_flexural": 6.85,
        "test_compressive": 60.50,
        "test_crack_mm": 0.25,
    },
    "PPF-C50": {
        "source_group": "XUkun/C41_BASE",
        "mix_label": "聚丙烯纤维",
        "fiber_material": "聚丙烯纤维",
        "fiber_type": "端钩型",
        "fiber_content": 0.10989,
        "aspect_ratio": 291.666667,
        "tensile_strength": 550.0,
        "test_flexural": 6.81,
        "test_compressive": 60.00,
        "test_crack_mm": 0.34,
    },
}

# C50 共用配合比（论文 partial / merged 一致）
COMMON_MIX = {
    "strength_grade": "C50",
    "concrete_type": "普通混凝土",
    "binder_content": 516.0,
    "cement_content": 330.0,
    "fly_ash": 93.0,
    "slag_grade": "S95",
    "slag_powder": 93.0,
    "mixing_water": 165.0,
    "w_b_ratio": 0.32,
    "sand_type": "天然砂",
    "sand_content": 686.0,
    "sand_ratio": 38.999431495167705,
    "stone_content": 1073.0,
    "admixture": "减水剂",
    "admixture_dosage": 10.8,
    "water_reducer_type": "无",
    "water_reduction_rate_unknown": False,
    "water_reduction_rate_pct": 0.0,
    "curing_days": 28,
    "temperature": 20.0,
    "humidity": 95.0,
    "casting_method": "常规",
}


def _load_default_residual_models(report_path: Path) -> dict[str, str | None]:
    with report_path.open(encoding="utf-8") as f:
        rep = json.load(f)
    dm = rep.get("default_method_by_task") or {}
    out: dict[str, str | None] = {}
    for task in ("compressive", "flexural"):
        learner = (dm.get(task) or {}).get("residual_learner")
        out[task] = None if learner in (None, "", "null") else str(learner)
    return out


def _predict_lab_strength_row(
    row: pd.Series,
    task: str,
    residual_learner: str | None,
    lab_dir: Path,
) -> dict[str, float | str]:
    formula = (
        row_compressive_formula_prediction(row)
        if task == "compressive"
        else row_flexural_formula_prediction(row)
    )
    if residual_learner is None:
        return {
            "formula_mpa": float(formula),
            "residual_mpa": 0.0,
            "pred_mpa": float(formula),
            "residual_model": "formula_only",
        }
    path = lab_dir / f"lab_{task}_residual_{residual_learner}.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"缺少残差模型: {path}")
    bundle = joblib.load(path)
    model = bundle["model"]
    # 推理不需要真值；用公式作占位使 y_resid=0 通过校验
    row_inf = row.copy()
    row_inf["compressive_true"] = float(formula) if task == "compressive" else float(
        row.get("compressive_true", formula)
    )
    row_inf["flexural_true"] = float(formula) if task == "flexural" else float(
        row.get("flexural_true", formula)
    )
    tup, meta = _one_row_feature_vector_with_reason(
        row_inf, task=task, append_source_domain=True
    )
    if tup is None:
        raise RuntimeError(f"特征向量构建失败 ({task}): {meta}")
    x, _, _, formula_check = tup
    assert abs(formula_check - formula) < 1e-6
    resid = float(model.predict(x.reshape(1, -1))[0])
    final = float(formula) + resid
    return {
        "formula_mpa": float(formula),
        "residual_mpa": resid,
        "pred_mpa": final,
        "residual_model": residual_learner,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_lab = pd.read_csv(LAB_MERGED)
    residual_by_task = _load_default_residual_models(LAB_REPORT)
    lab_dir = LAB_REPORT.parent

    predictor = SteelFiberCrackPredictor(model_dir=str(MODELS_DIR))

    rows_out: list[dict] = []

    for label, spec in SAMPLES.items():
        sg = spec["source_group"]
        sub = df_lab[df_lab["source_group"] == sg]
        if sub.empty:
            raise ValueError(f"merged CSV 中无 {sg}")
        lab_row = sub.iloc[0]

        user_inputs = {**COMMON_MIX, **spec}
        user_inputs.pop("source_group", None)
        user_inputs.pop("mix_label", None)
        user_inputs.pop("test_flexural", None)
        user_inputs.pop("test_compressive", None)
        user_inputs.pop("test_crack_mm", None)

        valid, X, msg, extra, _ = validate_and_transform(
            user_inputs, emit_streamlit_warnings=False
        )
        if not valid:
            raise RuntimeError(f"{label} 开裂模型输入无效: {msg}")

        crack_res = predictor.predict_all(X, extra)
        preds = crack_res["predictions"]
        state = preds.get("state_dimension") or {}
        risk = preds.get("risk_dimension") or {}

        comp = _predict_lab_strength_row(
            lab_row,
            "compressive",
            residual_by_task["compressive"],
            lab_dir,
        )
        flex = _predict_lab_strength_row(
            lab_row,
            "flexural",
            residual_by_task["flexural"],
            lab_dir,
        )

        rows_out.append(
            {
                "试样": label,
                "source_group": sg,
                "纤维类型": spec["mix_label"],
                "纤维材质编码用": spec["fiber_material"],
                "纤维掺量_%": spec["fiber_content"],
                "长径比": spec["aspect_ratio"],
                "纤维抗拉_MPa": spec["tensile_strength"],
                "强度等级": "C50",
                "水胶比": 0.32,
                "胶材_kg_m3": 516.0,
                "龄期_d": 28,
                "养护温度_C": 20.0,
                "养护湿度_%": 95.0,
                "浇筑方式": "常规",
                "预测抗折_MPa": round(flex["pred_mpa"], 3),
                "预测抗压_MPa": round(comp["pred_mpa"], 3),
                "预测裂缝宽度_mm": round(
                    float(state.get("crack_width_mm", preds.get("crack_width", 0))), 4
                ),
                "开裂风险概率P": round(
                    float(state.get("risk_probability", 0)), 4
                ),
                "开裂风险等级": str(risk.get("alert_level", "")),
                "抗压公式基线_MPa": round(comp["formula_mpa"], 3),
                "抗压残差修正_MPa": round(comp["residual_mpa"], 3),
                "抗压残差模型": comp["residual_model"],
                "抗折公式基线_MPa": round(flex["formula_mpa"], 3),
                "抗折残差修正_MPa": round(flex["residual_mpa"], 3),
                "抗折残差模型": flex["residual_model"],
                "试验抗折_MPa": spec["test_flexural"],
                "试验抗压_MPa": spec["test_compressive"],
                "试验裂缝宽度_mm": spec["test_crack_mm"],
            }
        )

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("=== C50 四组真实模型预测（不重训）===")
    print(f"开裂模型目录: {MODELS_DIR}")
    print(f"力学默认策略: {residual_by_task}")
    print(f"试件协议默认: {LAB_DEFAULTS}")
    print()
    paper_cols = [
        "试样",
        "预测抗折_MPa",
        "预测抗压_MPa",
        "预测裂缝宽度_mm",
        "开裂风险等级",
        "试验抗折_MPa",
        "试验抗压_MPa",
        "试验裂缝宽度_mm",
    ]
    print(out_df[paper_cols].to_string(index=False))
    print()
    print(f"完整表已写入: {OUT_CSV}")


if __name__ == "__main__":
    main()
