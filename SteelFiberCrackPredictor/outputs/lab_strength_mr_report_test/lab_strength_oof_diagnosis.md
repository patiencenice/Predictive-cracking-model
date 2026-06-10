# 强度残差 OOF 诊断摘要

由 `train_eval._diagnose_oof` 与 `automated_summary` 自动生成。

## OOF 指标摘要（启发式）

### compressive
- 最优残差学习器（三者中 OOF MAE 最低）: `ridge`
- OOF MAE 公式: 2.91053076133046；最优残差学习器: 2.864903647188436；综合推荐: `ridge`（MAE=2.864903647188436）
- MAE 低于公式的残差学习器: ['ridge']

### flexural
- 最优残差学习器（三者中 OOF MAE 最低）: `ridge`
- OOF MAE 公式: 0.45691845640602086；最优残差学习器: 0.5524586806477934；综合推荐: `formula_only`（MAE=0.45691845640602086）
- MAE 低于公式的残差学习器: []

- 两任务「残差学习器」是否均优于公式（MAE）: False
- 两任务综合最优是否均为残差（非纯公式）: False
- 两任务相对 MAE 降幅（相对公式）是否均 ≥3%: False

## compressive

- **公式基线整体偏差**（mean(y_true − formula_pred)）: 1.8545322702933666
- 解读: 均值为正：整体上公式略低估真值（y_true 高于 formula_pred）。
- **OOF 上最优残差模型**（按 mean|error|）: `ridge`
- 最优模型后整体偏差 mean(y_true − final_pred): 0.8432760084138851
- **是否缩小相对公式的 |偏差|**: True

## flexural

- **公式基线整体偏差**（mean(y_true − formula_pred)）: 0.42094019593633936
- 解读: 均值为正：整体上公式略低估真值（y_true 高于 formula_pred）。
- **OOF 上最优残差模型**（按 mean|error|）: `ridge`
- 最优模型后整体偏差 mean(y_true − final_pred): 0.13566939589669164
- **是否缩小相对公式的 |偏差|**: True
