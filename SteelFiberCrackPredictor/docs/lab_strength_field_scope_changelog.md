# 试验估算 / lab_strength 字段作用域与变更说明

本文说明 **抗压** 与 **抗折** 在公式层、网页「试验估算」与 `lab_strength_residual` 训练矩阵中的输入解耦方式，以及减水剂扩展特征的行为。

## 1. 公式层（`src/lab_formula_gb.py`）

| 作用域 | 函数 | 主要输入 |
|--------|------|----------|
| **仅抗压** | `compressive_formula_pred_mpa` | 试件类型、立方体边长或棱柱体 `b×h×L`、**抗压加载方式** `loading_compression`、`cube_strength_mpa`（fcu,k）。**不读取**梁跨、**不读取**抗折加载方式。 |
| **仅抗折** | `flexural_formula_pred_mpa` | 试件类型、**梁 `b×h×跨度`**、**抗折加载方式** `loading_flexural`（三点/四点）、`cube_strength_mpa`、纤维体积掺量。**不读取**抗压加载方式。 |

训练数据中行级公式：`dataset.row_compressive_formula_prediction` **不读取** `lab_loading_flexural`；`row_flexural_formula_prediction` 读取梁参数与 `lab_loading_flexural`。

## 2. 网页「试验估算」（`src/visualizer.py` + `src/lab_experiment.py`）

- **估算范围**：`仅抗压` / `抗压与抗折`（`lab_estimate_scope`）。
- **仅抗压**：不展示抗折加载方式选择；`estimate_strengths(..., compute_flexural=False)`，抗折指标为 **「—」**，抗折公式不计算。
- **抗压与抗折**：必须选择「抗折试验 · 加载方式」后参与抗折公式；**抗压数值仍不依赖该字段**（抗压走 `estimate_compressive_strength` → `compressive_formula_pred_mpa`）。

## 3. 训练 CSV 与矩阵列（`LAB_STRENGTH_FEATURE_COLUMNS`）

在 `src.features.FEATURE_COLUMNS` 基础上增加减水剂 6 列（见 `src/lab_strength_residual/lab_mix_features.py`）。其中 **与试验几何/加载相关的列** 在公式中的使用方式为：

### 3.1 仅影响抗压公式基线（残差模型 `task="compressive"` 时公式项仍只用下列逻辑算 compressive_formula_pred）

- CSV / 逻辑字段（若存在）：`lab_specimen`、`lab_cube_edge_mm`、`lab_prism_b_mm`、`lab_prism_h_mm`、`lab_prism_l_mm`、`lab_loading_compression`  
- 与 `compressive_formula_pred_mpa` 一致：**不使用** `lab_loading_flexural`、`lab_beam_*`。

### 3.2 仅影响抗折公式基线（`task="flexural"`）

- `lab_specimen`、`lab_beam_b_mm`、`lab_beam_h_mm`、`lab_beam_span_mm`、`lab_loading_flexural`  
- **不使用** `lab_loading_compression` 计算抗折公式。

### 3.3 两套任务共用的表格列（配合比、环境、纤维编码等）

`FEATURE_COLUMNS` 内各列（如 `curing_days`、`w_b_ratio`、`fiber_content` 等）在 **抗压 / 抗折残差模型** 中都会进入特征向量 `X`；它们不替代国标公式，而是作为数据驱动残差项的输入。  
**例外**：减水剂扩展列对公式基线无直接闭合式，仅以数值 + missing flag 供残差模型使用。

## 4. 减水剂扩展特征（不编造缺失的减水率）

| 列名 | 含义 |
|------|------|
| `water_reducer_type`（可选，训练前可映射为 `water_reducer_type_enc`） | 类型文本或编码来源 |
| `water_reducer_type_enc` / `water_reducer_type_missing_flag` | 类型编码；-1 与 flag=1 表示缺失 |
| `water_reduction_rate_pct` / `water_reduction_rate_missing_flag` | 减水率（%）；缺失时为 -1 与 flag=1，**不填充假数** |
| `adjusted_w_b_ratio` / `adjusted_w_b_ratio_missing_flag` | 仅在 `w_b_ratio` 与减水率均可解析时计算 `w_b * (1 - rate/100)`，否则 -1 与 flag=1 |

原始 CSV 可无上述列；`scripts/build_lab_strength_example_csv.py` 等会为示例行调用 `lab_mix_extra_row_vector` 写入 6 列。

## 5. 变更摘要（实现要点）

1. 抗压公式路径与 `lab_loading_flexural` 解耦（代码与文档一致）。  
2. UI 通过「估算范围」区分：**抗折加载方式仅在「抗压与抗折」时出现并参与抗折计算**。  
3. `estimate_strengths` 增加 `compute_flexural`；仅抗压时不输出抗折公式基线（非数占位在界面显示为「—」）。  
4. 残差训练：`build_xy_matrices(..., task=...)` 中公式列随 task 切换；`X` 使用完整 `LAB_STRENGTH_FEATURE_COLUMNS`（含减水剂 6 列）。
