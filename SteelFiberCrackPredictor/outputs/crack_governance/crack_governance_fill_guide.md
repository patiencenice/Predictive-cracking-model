# 主开裂训练表治理列人工填写说明

> 对应模板：`outputs/crack_governance/crack_governance_fill_template.csv`  
> 源数据：`data/training_data.csv`（120 行）  
> 填写完成后，由专人将治理列合并回主表（本阶段不提供自动回写脚本，避免误覆盖）。

---

## 1. 模板用途

- 在**不重训、不改 `FEATURE_COLUMNS`** 的前提下，为每一行补齐 **sidecar 治理 metadata**。
- 模板中的 **原始输入/标签列** 仅供对照，请勿在模板里修改（应在主表 `training_data.csv` 中改特征/标签）。
- **治理列与 `fill_note` / `reviewer` / `review_date` 由人工填写**；导出脚本不会预填。

---

## 2. 列说明

| 列名 | 是否由人工填写 | 说明 |
|------|----------------|------|
| `row_index` | 否 | 与 `training_data.csv` 行号一致（0 起），用于回并 |
| `strength_grade` | 否 | 由 `strength_grade_enc` 解码，仅便于阅读 |
| `fiber_type` | 否 | 由 `fiber_type_enc` 解码，仅便于阅读 |
| `fiber_content` / `aspect_ratio` / `w_b_ratio` 等 | 否 | 源表快照 |
| `crack_width` / `crack_density` / `cracking_risk` | 否 | 源表标签快照 |
| `source_group` | **是** | 分组键 |
| `data_tier` | **是** | 数据分层 |
| `needs_manual_review` | **是** | 是否暂缓入模 |
| `crack_width_definition_id` | **是** | 裂缝宽度测量口径 |
| `literature_key` / `source_doi` / `table_or_figure_ref` | **是** | 文献溯源（可选） |
| `fill_note` | **是** | 填写依据、疑点说明 |
| `reviewer` | **是** | 填写人 |
| `review_date` | **是** | 复核日期（建议 `YYYY-MM-DD`） |

---

## 3. `source_group` 怎么填

**含义：** 交叉验证与误差诊断的分组键，防止同一文献/同一配比系列同时出现在训练折与验证折。

**推荐格式：**

- 本地试验：`USER_LOCAL_LAB/<批次或试件系列>`，例如 `USER_LOCAL_LAB/BATCH_2024_A`
- 文献：`/<DOI 或 literature_key>/<表号或配比系列>`，例如 `10.1000/xxx/paper1/Table3_mixA`

**要求：**

- 同一论文、同一表、同一配比系列 → **同一 `source_group`**
- 不同表号 / 不同系列 → **不同 `source_group`**
- **禁止**无依据批量填写 `USER_LOCAL_LAB/DEFAULT` 或随机占位
- 暂时无法判断分组 → **留空**，并设 `needs_manual_review=1`

---

## 4. `data_tier` 允许值

仅允许以下取值（与 `data/TRAINING_DATA_SPEC.txt`、`config/crack_protocol_schema.yaml` 一致）：

| 取值 | 含义 |
|------|------|
| `A_lab_native` | 本地试验，协议与标签已人工核对 |
| `B_literature_verified` | 文献数据，协议与标签已核对 |
| `C_literature_extracted` | 文献抽取/半自动，尚未充分核实 |
| *(空)* | 未分级 |

**禁止：**

- 单字母 `A` / `B` / `C`（与诊断脚本口径不一致）
- 无溯源时标 `A_lab_native`

---

## 5. `needs_manual_review` 何时填 `1`

| 填 `1`（暂缓入模） | 填 `0`（可进入训练候选，仍须满足 A 类其它条件） |
|-------------------|-----------------------------------------------|
| 无法确认 `source_group` | 分组、口径、标签均已核对 |
| 裂缝宽度测量口径不明 | |
| 文献来源/表号无法对应 | |
| 标签与试验记录疑似不一致 | |
| 单位或定义存疑 | |
| 任何不确定是否应进主 OOF 的行 | |

**重要：**

- **不要留空。** 空值在 `diagnose_crack_training_governance.py` 中视同**暂缓**（与当前 120 行状态一致）。
- **不要默认填 `0`。**

---

## 6. `crack_width_definition_id` 允许值

| ID | 含义 |
|----|------|
| `CW_MAX_SURFACE_MM` | 构件表面实测**最大**裂缝宽度（mm） |
| `CW_MEAN_MM` | 多条裂缝宽度**平均**值 |
| `CW_INNER_MM` | 内部或剖开观测宽度 |
| `CW_UNSPECIFIED` | 口径不明（仅作占位，**不能**与 A 类同时使用） |

- 不同 ID **不可混为同一训练池** 的主 OOF（见 `outputs/label_definition.md`）。
- 未确定口径 → 留空 + `needs_manual_review=1`，或暂填 `CW_UNSPECIFIED` 且 **不得** 标 `A_lab_native`。

---

## 7. 文献溯源列（可选但建议）

| 列 | 何时填 |
|----|--------|
| `literature_key` | 文献行：内部论文 id 或短 DOI |
| `source_doi` | 有 DOI 时；纯本地可用 `USER_LOCAL_LAB` 等**有记录**标识 |
| `table_or_figure_ref` | 表号/图号，如 `Table 3` |

文献行建议 **`literature_key` + `table_or_figure_ref` 至少填一项**。

---

## 8. 哪些情况下不能标 A 类

行须**同时**满足下列全部条件，才可在诊断中计为 **A 类候选**（`tier_A_candidate`）：

1. `data_tier` = `A_lab_native`
2. `needs_manual_review` = `0`（且非空）
3. `source_group` 非空
4. `crack_width_definition_id` 为合法 ID，且 **≠ `CW_UNSPECIFIED`**
5. `crack_width`、`crack_density`、`cracking_risk` 标签完整有效

**不得标 A 的典型情况：**

- 无 `source_group`
- `needs_manual_review` 为空或 = `1`
- 口径为 `CW_UNSPECIFIED` 或未填
- 合成/演示数据、三类风险完全均衡且无任何溯源记录
- 不同裂缝宽度定义混在同一「A 池」

---

## 9. 当前 120 行为什么默认不能进 A 类

依据 `outputs/crack_governance/crack_training_governance.json`（结构接入后、人工补值前）：

| 状态 | 行数 |
|------|------|
| 暂缓（`hold_pending`） | **120** |
| A 类候选（`tier_A_candidate`） | **0** |
| `source_group` 非空 | **0** |
| 治理列非空 | **0** |

**原因归纳：**

1. 治理列虽已建列，但**内容全空**。
2. `needs_manual_review` 为空 → 诊断规则**一律视作暂缓**，不会自动当作 `0`。
3. `source_group`、`data_tier`、`crack_width_definition_id` 均未填写 → 无法满足 A 类条件。
4. 当前表呈现**三类 `cracking_risk` 各 40 条**的均衡分布，且无文献/批次溯源，在补全治理信息前应视为**不可审计数据集**，不宜作为可信主 OOF 或重训依据。

---

## 10. 填写后自检

补值完成后请运行（只读，不重训）：

```bash
py scripts/diagnose_crack_training_governance.py
```

关注：`tier_ABC_hold_counts`、`group_audit`、`illegal_data_tier_values`。

---

## 11. 与训练链路的关系

- 治理列 **不进入 `FEATURE_COLUMNS`**，**不是**主模型输入特征。
- 本阶段 **不修改** `train_model.py` / `predictor.py`。
- **在 A 类候选 > 0 且 `source_group` 可审计之前，不要重训。**
