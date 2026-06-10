# 温度应力解释板块（Phase 1）

**更新日期：** 2026-05-22  
**状态：** 工程解释链已闭合（本轮通过）；**冻结扩展**，待真实数据后再做校准与验证。

本目录为**解释增强模块**，与主开裂预测（`crack_width` / `crack_density` / `cracking_risk`）**并行**，不替代主模型输出，**不**并入主 `FEATURE_COLUMNS`，**不**修改 `train_model.py` / `predictor.py` 主训练与推理链路，**不重训**。

---

## 阶段结论（2026-05-22）

温度应力模块已形成完整**工程解释链**（Tab③「开裂机理」展示）：

**温度变化 → 温度应变 → 约束温度应力 → 应力-抗拉能力比 η → 开裂风险解释**

实现路径：

| 环节 | 公式 / 量 | 代码 |
|------|-----------|------|
| 温度应变 | `ε_T = α × ΔT` | `engineering_chain.py` |
| 约束温度应力 | `σ_T* = R × E × ε_T`（等价 `R × E × α × ΔT`） | 同上 |
| 开裂判据 | `η = σ_T* / f_t` | 同上 |
| 无量纲辅助指数 | `thermal_stress_index = R × α_norm × E_norm × g(ΔT)` | `derive.py`（算法未因 UI 升级而改动） |

**后续暂时不要：** 继续扩展公式、将本模块接入主训练链路、把 `thermal_stress_index` 写入 `FEATURE_COLUMNS`。

---

## 重要边界（冻结口径）

1. **σ_T*** 为**工程解释应力标量**，按 `R·E·ε` 量级展示，**不代表**真实有限元温度应力，**不做** FE / PDE / 热力耦合求解。
2. **f_t** 优先级：`splitting_tensile_strength_mpa`（实测劈裂抗拉）→ `flexural_strength_mpa`（备选）→ `cube_strength_mpa` 按 **GB 50010-2010 表 4.1.3-2** 经验映射；缺失不填 0。
3. **η** 用于解释温度作用对开裂风险的**贡献与分级**，**不替代**主模型输出的 `cracking_risk` 概率 **P** 及预警带。
4. **η 阈值**（η &lt; 0.6 低风险；0.6~1.0 中风险；&gt; 1.0 高风险）仅为 **Phase 1 工程解释分级**，**未经真实数据标定**；后续须用项目数据校准，不得直接当作规范判据。
5. **主链路不变：** 不改 `train_model.py`，不改 `predictor.py`，不改 `FEATURE_COLUMNS`，不重训；`derive.py` 中 `thermal_stress_index` 算法保持与本轮通过时一致。

其它技术边界：

| 说明 | 内容 |
|------|------|
| **非真实应力** | `thermal_stress_index` **不是** MPa 级温度应力，**不**拟合 FE/Abaqus/热力耦合 PDE。 |
| **不替代 crack_*** | 不改变、不覆盖主任务标签与主推理输出。 |
| **禁止未知→0** | 关键输入缺失时，对应指数输出 **-1**，并置 `*_missing_flag = 1`；**禁止**把未知静默写成 0。 |

---

## 待真实数据后开展（暂不实施）

在提供温度、裂缝、强度等**真实数据**之前，**不进行**下列工作；数据到位后再立项：

| 序号 | 工作项 | 目的 |
|------|--------|------|
| 1 | **η 与 `crack_width` 的关系分析** | 检验温度解释判据与实测/标注缝宽的一致性 |
| 2 | **η 与 `cracking_risk` 的相关性分析** | 对比解释层 η 与主模型风险输出的关联，避免重复叙事 |
| 3 | **η 阈值校准** | 用数据重定低/中/高分级，替代当前 Phase 1 示意阈值 |
| 4 | **解释增益验证** | 评估温度应力模块是否对工程判断有增量信息（相对仅用主模型） |

### 扩展可选输入（Phase 1+，2026-05-23）

侧栏与 CSV 可携带下列**可选**列（见 `field_spec.yaml`），**不进入** `FEATURE_COLUMNS`：

| 类别 | 字段 | 说明 |
|------|------|------|
| 实测强度 | `splitting_tensile_strength_mpa`, `flexural_strength_mpa` | η 分母 f_t 优先实测劈裂抗拉 |
| 温度路径 | `core_peak_temperature_c`, `surface_temperature_c`, `time_to_peak_temperature_h`, `cooling_rate_c_per_h` | 解释用，不强制进 σ_T* 基线 |
| 试验约束 | `restraint_percent`, `restraint_code` | 原始 R0/R50/R100，≠ `restraint_factor_R` |
| 裂缝观测 | `thermal_crack_observed`, `thermal_crack_time_h`, `thermal_crack_width_mm`, `apparent_crack_filtered` | 仅验证 η–开裂关系 |

上述分析均为**离线研究**，默认仍**不**将 `thermal_stress_index` 或 η 并入主训练特征，除非单独评审并更新阶段文档。

---

## UI 展示层级（Tab③，已实现）

1. 工程结论（基于 η 的报告式语句）  
2. 工程公式链（Step 1~3）  
3. 工程变量说明卡片  
4. 温度 → 应变 → 应力 → 开裂判据流程图  
5. 与 `crack_width` / `cracking_risk` 的关系说明  
6. 模块底部公式免责声明  
7. **高级诊断（开发/研究）**（默认折叠：`missing_flag`、derive 路径、JSON）

---

## 文件

| 文件 | 职责 |
|------|------|
| `field_spec.yaml` | A/B/C 类字段与枚举、归一化参考常量说明 |
| `optional_fields.py` | 扩展可选输入、`resolve_f_t_for_eta`、`derive_thermal_optional_context` |
| `derive.py` | `derive_thermal_stress_features(row)`、`thermal_stress_explain_sentence_zh` |
| `engineering_chain.py` | `derive_thermal_engineering_display(row)`、`thermal_engineering_conclusion_zh`（仅展示层，不改 `thermal_stress_index`） |

---

## 离线诊断

在项目根目录执行：

```bash
python scripts/diagnose_thermal_stress_inputs.py --csv data/training_data.csv
```

生成 `outputs/thermal_stress/thermal_input_diagnosis.json`（只读审计，不训练）。

---

## 与主项目阶段的关系

- **主开裂数据治理 / 重训：** 见 `outputs/crack_governance/PHASE_STATUS.md`（当前优先治理模板人工补值，非温度模块扩展）。
- **模型权重自检：** 见 `outputs/model_integrity/model_loading_integrity_report.md`。
