# 工程开裂环境场 + 施工工艺场（Phase 1）

解释层扩展：侧栏输入 → `derive_environment_engineering_features` → 机理 / 温度应力 / 高级分析展示。

**不进入** `FEATURE_COLUMNS`，不改 `predictor` / `train_model` / pkl。

## 侧栏层级

1. **环境条件**：温度、湿度、风速、昼夜温差、暴晒等级  
2. **养护制度**：龄期、养护方式  
3. **施工工艺**：浇筑方式、振捣质量、泌水倾向、施工季节  
4. **结构条件**：构件类型  

## 派生指标

| 字段 | 说明 |
|------|------|
| `evaporation_risk_index` | T × 风速 × (1−RH/100) |
| `thermal_gradient_risk` | ΔT_day-night × 暴晒等级系数 |
| `surface_shrinkage_risk` | 风速 + 温度 + 低湿度组合 |
| `environment_engineering_summary` | 蒸发/温差/养护/表层/热裂缝倾向文案 |

规范见 `field_spec.yaml`，实现见 `derive.py`。
