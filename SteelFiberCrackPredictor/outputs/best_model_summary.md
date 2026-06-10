# crack_width GroupKFold 结果摘要

- 本次 `crack_width_definition_id` 过滤：`CW_MAX_SURFACE_MM`（未指定则为 null，使用全部分类口径）
- 过滤前/后样本数：123 / 3
- 优先指标：**CV 平均 R²**（分组列：**优先 `source_group`，否则 `source_doi`**，避免同组泄漏）
- 折数上限：`min(n_splits, 不同分组数)`

- **推荐策略（本次运行）**：`only_literature`
  - CV R² mean = nan ± nan
  - CV MAE mean = 0.033443；CV RMSE mean = 0.033443

完整折间指标见 `outputs/cv_report_groupkfold.json`。