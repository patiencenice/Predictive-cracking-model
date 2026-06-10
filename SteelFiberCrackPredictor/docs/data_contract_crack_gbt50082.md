# 裂缝与耐久试验数据契约（GB/T 50082-2024 对齐 · 草稿）

## 1. 目的

约束 `SteelFiberCrackPredictor` 中与 **裂缝宽度 / 裂缝密度 / 开裂风险** 及 **50082 相关试验** 的数据列语义，避免：

- 将 **按标准公式推算的值** 误标为 **量测真值标签**；  
- 将 **不同试验方法** 的结果混在同一标签定义下训练；  
- 缺少 **试件、养护、环境、报告** 元数据时仍强行入模。

## 2. 字段层级

| 层级 | 含义 | 示例 |
|------|------|------|
| `L0_raw` | 仪器/文献原始记录，未经标准公式 | 读数、扫描表单元格 |
| `L1_protocol` | 已按 `crack_protocol_schema.yaml` 对齐列名与单位，尚未换算 | `crack_width_mm` |
| `L2_derived` | 由国标规则 **仅** 派生的几何/时间归一、量纲检查、基线分量 | 见 `gbt50082_feature_derivation.py` |
| `L3_gate` | 闸门输出：通过/拒收/需人工复核 | 见 `gbt50082_data_gate.py` |
| `L4_label` | **允许作为监督信号** 的列；须独立溯源 | 通常为 `L0_raw` 经质控，**不是**标准公式直接输出 |

## 3. 硬规则

1. **标准公式输出 ∈ {L2_derived, 基线分量}**，默认 **∉ L4_label**。  
2. **`baseline + residual`**：标准公式仅进入 **baseline** 或 **派生特征**；残差目标相对的量须与 `data_contract` 一致声明。  
3. **`data_tier` / `source_group` / `standard_method_id`** 必须在训练 CSV 或 sidecar 中可追溯（见 `config/crack_protocol_schema.yaml`）。

## 4. 与现有工程的关系

- **不修改** `train_model.py`、`predictor.py`、`features.py` **核心签名**；本契约供新闸门与特征模块消费。  
- 主开裂链路接入前，须完成 **YAML 条款补全 + 单元测试（条文对照表）**。
