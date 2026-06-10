# Origin 散点图数据说明

数据文件：`outputs\evaluation\origin_plot_data.xlsx`

本表仅整理既有离线评估/ OOF 导出结果，**未重新训练或推理**。
## 列定义（每个 Sheet 相同）

| 列 | 字母 | 含义 | Origin 用法 |
|----|------|------|-------------|
| Measured | A | 实测值 | 横轴（X） |
| Predicted | B | 预测值 | 纵轴（Y） |
| y=x | C | y=x 参考线 | 与 A 同值；作第二组 Y 或 Line 图可画 1:1 线 |

## Sheet 与论文图对应关系

| Sheet | 论文章节 | 对应论文图 | 说明 |
|-------|----------|------------|------|
| 抗压强度 | 7.3.1 / 7.3.3 | **图7-5 (a)** 抗压强度预测值与实测值对比 | 双子图左 panel |
| 抗折强度 | 7.3.1 / 7.3.3 | **图7-5 (b)** 抗折强度预测值与实测值对比 | 双子图右 panel |
| 裂缝宽度 | 7.3.2 / 7.3.3 | **图7-3**（建议）裂缝宽度预测值与实测值对比 | hold-out 测试集；单位 mm |
| 裂缝密度 | 7.3.2 / 7.3.3 | **图7-4**（建议）裂缝密度预测值与实测值对比 | hold-out 测试集；单位 条/m² |

> 图7-3、图7-4 编号若与定稿不一致，请在 Word 中统一调整后，Origin 仍按本表 A/B 列作散点即可。

## 数据来源

| Sheet | 源 CSV | 行数 |
|-------|--------|------|
| 抗压强度 | `outputs\lab_strength\lab_strength_compressive_true_pred.csv` | 33 |
| 抗折强度 | `outputs\lab_strength\lab_strength_flexural_true_pred.csv` | 33 |
| 裂缝宽度 | `outputs\evaluation\crack_width_true_pred.csv` | 25 |
| 裂缝密度 | `outputs\evaluation\crack_density_true_pred.csv` | 25 |

## Origin 导入提示

1. `File` → `Import` → `Single ASCII` / Excel，选中对应 Sheet。
2. 散点：X = **Measured**，Y = **Predicted**。
3. y=x：将 **Measured** 作 X，**y=x** 作 Y，线型设为虚线；或与散点共用 X=Measured、Y=y=x。
4. 力学强度子图单位：MPa；裂缝宽度：mm；裂缝密度：条/m²。

## 未纳入本表的文件

- `cracking_risk_true_pred.csv`：分类任务，不适合 Measured/Predicted 回归散点，请用混淆矩阵等图。

## 重新生成

```powershell
cd d:\cursor\code\SteelFiberCrackPredictor
py -3 scripts/export_origin_plot_data.py
```
