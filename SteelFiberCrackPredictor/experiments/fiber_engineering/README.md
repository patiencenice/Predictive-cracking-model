# 工程抗裂纤维体系（Phase 1 解释层）

**不进 `FEATURE_COLUMNS`；不改 `predictor.py` / `train_model.py`。**

## 输入字段

见 `field_spec.yaml`：基础参数 + 工程抗裂参数 + 高级占位。

## 派生

`derive_fiber_engineering_features(row)` →

- `fiber_diameter_mm` = `fiber_length_mm / aspect_ratio`
- `fiber_constraint_index` = 体积分数 × E_f 归一 × 长径比归一
- `fiber_engineering_summary` / `bridge_explanation_zh` / `thermal_fiber_note_zh`

## UI

侧栏 `src/fiber_inputs.py`；机理 Tab `visualizer._show_fiber_engineering_bridge_section`。
