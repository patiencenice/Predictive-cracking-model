# 分组评估建议（GB/T 50082-2024 上下文）

> 草稿：不与现有 `GroupKFold(source_group)` 冲突，供后续实验设计参考。

## 1. 推荐分层键（优先级从高到低）

1. **`standard_method_id`**（或 `gbt50082:method_id`）：收缩非接触 / 收缩接触 / 早裂平板等，**防止跨方法混评**。  
2. **`data_tier`**：与 `docs/data_contract_crack_gbt50082.md` 一致（`A_lab_native` / `B_literature_verified` / `C_literature_extracted`）。  
3. **`source_group`**：文献或实验室批次。  
4. **`literature_key` + `table_or_figure_ref`**：同一论文不同表不可合并为一组误差解释。

## 2. 交叉验证策略建议

- **首选**：`GroupKFold` 按 **`source_group`** 或 **`literature_key`**。  
- **当方法混杂时**：先按 **`standard_method_id`** 分层抽样或分模型评估，再汇总。  
- **小样本组（n=1）**：单独报告尾部误差，不把单点 MAE 写入总表主结论。

## 3. 报告拆分维度

- 按 **方法** 输出 MAE / RMSE / 校准曲线；  
- 按 **养护温度/湿度区间**（若列存在）做分层箱线图（仅展示层，不自动调权，除非另开任务）。
