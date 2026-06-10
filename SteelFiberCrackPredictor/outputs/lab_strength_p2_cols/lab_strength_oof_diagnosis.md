# 强度残差 OOF 诊断摘要

由 `train_eval._diagnose_oof` 与 `automated_summary` 自动生成。

## OOF 指标摘要（启发式）

### compressive
- 最优残差学习器（三者中 OOF MAE 最低）: `hgb`
- OOF MAE 公式: 0.9361464634960486；最优残差学习器: 1.0006763667696388；综合推荐: `formula_only`（MAE=0.9361464634960486）
- MAE 低于公式的残差学习器: []

### flexural
- 最优残差学习器（三者中 OOF MAE 最低）: `hgb`
- OOF MAE 公式: 0.03528352964345153；最优残差学习器: 0.03515005288881129；综合推荐: `hgb`（MAE=0.03515005288881129）
- MAE 低于公式的残差学习器: ['hgb']

- 两任务「残差学习器」是否均优于公式（MAE）: False
- 两任务综合最优是否均为残差（非纯公式）: False
- 两任务相对 MAE 降幅（相对公式）是否均 ≥3%: False

## compressive

- **公式基线整体偏差**（mean(y_true − formula_pred)）: -0.5158514616799547
- 解读: 均值为负：整体上公式略高估真值。
- **OOF 上最优残差模型**（按 mean|error|）: `hgb`
- 最优模型后整体偏差 mean(y_true − final_pred): 0.004124291887041072
- **是否缩小相对公式的 |偏差|**: True

## flexural

- **公式基线整体偏差**（mean(y_true − formula_pred)）: -0.014186578502360566
- 解读: 均值为负：整体上公式略高估真值。
- **OOF 上最优残差模型**（按 mean|error|）: `hgb`
- 最优模型后整体偏差 mean(y_true − final_pred): 0.0001519947952150121
- **是否缩小相对公式的 |偏差|**: True
