# lab_strength 训练数据 — 可选追溯与分层列

以下列**不是** `LAB_STRENGTH_FEATURE_COLUMNS` 的一部分，**不参与**残差模型特征矩阵；闸门**不要求**存在。用于合并去重审计、文献复现与按层级筛选扩展数据。

| 列名 | 含义 | 建议取值 |
|------|------|----------|
| `literature_key` | 文献稳定标识（DOI 短码、内部论文 id 等） | 有文献行尽量填；无则留空 |
| `table_or_figure_ref` | 原文表格/图号与行意 | 如 `Table 3` |
| `extraction_batch_id` | 本行进入仓库的抽取或整理批次 | 如 `yimeng_2026-04` |
| `data_tier` | 数据可信度/来源分层 | `A_lab_native`：本地试验直连；`B_literature_verified`：文献已人工核对协议与标签；`C_literature_extracted`：自动/半自动抽取待核对；空：未分级 |
| `row_uid` | 行级稳定 id（可选） | 仅在有明确命名规则时填写，勿随机编造 |

与现有列关系：`source_group` 仍用于分组与权重；`needs_manual_review` 控制是否进入公式基线训练；`strength_grade` / `strength_grade_enc` 保留等级信息。本表列**禁止**用于改写 `compressive_true` / `flexural_true`。

## compressive 协议治理扩展列（本轮新增）

以下列用于 **compressive 协议显式化与候选集筛选**，仍不进入残差特征矩阵：

| 字段名 | 类型 | 推荐取值 | 是否必填（进入 compressive 候选时） | 缺失是否建议触发 `needs_manual_review` | 是否参与 strict/relaxed |
|---|---|---|---|---|---|
| `lab_specimen` | 字符串（枚举） | `立方体（边长可变）` / `棱柱体（轴心抗压）` / `梁式试件（抗折）` | 是 | 是 | 是 |
| `lab_cube_edge_mm` | 浮点 | 常见 `100/150/200`；strict 目标 `150` | 当 `lab_specimen` 为立方体时是 | 是 | 是 |
| `lab_loading_compression` | 字符串（枚举） | 与 `LOADING_COMPRESSION` 一致 | 是 | 是 | 是 |
| `lab_curing_regime` | 字符串（枚举） | `standard` / `non_standard` / `unknown` | 是 | 是 | 是 |
| `lab_curing_note` | 字符串 | 原文养护说明 | 否（追溯） | 否（但建议补） | 否（辅助解释） |
| `cube_strength_mpa_semantics` | 字符串（枚举） | `fcu_k_design` / `cube_test_mean` / `cube_test_representative` / `unknown` | 是 | 是 | 是（作为协议闭合前置） |
| `cube_strength_mpa_source_note` | 字符串 | 来源、换算、单位口径说明 | 否（追溯） | 否（但建议补） | 否（辅助解释） |
| `lab_protocol_closed_flag_compressive` | 整数/布尔 | `1`=协议闭合；`0/空`=未闭合 | 是（strict/relaxed 都依赖） | 是（未填或非 1） | 是 |

说明：
- 本仓库当前阶段只做字段治理与诊断，不自动编造协议值。
- 旧表补列时统一补空，不将历史行默认判为 `lab_protocol_closed_flag_compressive=1`。
