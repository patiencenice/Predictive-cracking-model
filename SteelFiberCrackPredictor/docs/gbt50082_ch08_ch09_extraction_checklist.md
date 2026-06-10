# GB/T 50082-2024 第 8、9 章 — 字段提取清单（Excel/CSV 填表用）

> **PDF 状态**：桌面版 `GB/T 50082-2024` 为 **169 页扫描版**，程序无法自动读出条文号。  
> 下列 **带「待核对」** 的数值来自公开解读与 2009 版惯例，填入 `config/gbt50082_methods.yaml` 前 **必须** 在 PDF 中逐条核对。  
> **禁止**将标准公式输出直接写入 `crack_width` / `crack_density` 训练标签（见 `docs/data_contract_crack_gbt50082.md`）。

---

## 一、怎么用（3 步）

1. 打开国标 PDF **目录**，定位 **第 8 章（收缩）**、**第 9 章（早期抗裂）** 起止页，填到下方「页码记录表」。  
2. 按本章表格，从 PDF **试件、环境、观测、公式、报告** 五类抄录到 Excel：  
   - 收缩：`data/gbt50082/template_ch08_shrinkage.csv`  
   - 早裂：`data/gbt50082/template_ch09_early_cracking.csv`  
3. 试验 **实测结果行**（若有）映射到 `data/training_data.csv` 或文献 raw CSV，并填写 `standard_method_id`、`data_tier=A_lab_native`。

---

## 二、页码记录表（请人工填写）

| 章节 | 标准条文 | PDF 起始页 | PDF 结束页 | 核对人 | 日期 |
|------|----------|------------|------------|--------|------|
| 第 8 章 收缩试验 | 8.1 非接触法 | | | | |
| | 8.2 接触法 | | | | |
| | 8.3 波纹管法 | | | | |
| 第 9 章 早期抗裂试验 | 9.x（按 PDF 小节） | | | | |
| 第 3 章 基本规定 | 3.3 试验报告 | | | | |

---

## 三、第 8 章 — 收缩试验（协议提取）

### 8.1 非接触法 `shrinkage_noncontact`

| 提取字段 | 写入 YAML/CSV 列 | 说明 | 待核对参考 |
|----------|------------------|------|------------|
| 条文号 | `clause_ref` | 如 8.1.x | — |
| 适用范围 | 适用范围 | 早龄期、无约束、无介质交换 | 公开解读 |
| 试件形状 | `specimen_shape_id` | 棱柱体 | — |
| 试件尺寸 mm | `specimen_length_mm` 等 | 常见 **100×100×515**（待核对） | 行业解读 |
| 初凝时刻读数 | `observation_note` | 初凝测初始变形 | 公开解读 |
| 观测间隔 | `observation_interval_h` | 如每 1 h（待核对） | 公开解读 |
| 环境温度 ℃ | `test_temperature_c` | | |
| 环境 RH % | `test_rh_pct` | | |
| 收缩率公式 | `结果计算公式` | **只作 L2 基线**，不作标签 | 须抄原文公式编号 |
| 报告必填项 | `报告必填项` | 委托/试件/条件/结果 | 见 3.3 章 |

### 8.2 接触法 `shrinkage_contact`

| 提取字段 | 说明 |
|----------|------|
| 试件类型 | 立方体或棱柱体（以 PDF 为准） |
| 测量装置 | 千分表 / 位移传感器 |
| 标距长度 mm | 用于应变换算 |
| 环境稳定要求 | T、RH 控制 |
| 收缩率公式 | 条文编号 + 表达式 |

### 8.3 波纹管法 `shrinkage_corrugated_pipe`（2024 新增）

| 提取字段 | 待核对参考 |
|----------|------------|
| 波纹管材质 | 低密度聚乙烯 |
| 管长 mm | **420±2** |
| 外径 mm | **80±1** |
| 内径 mm | **60±1** |
| 壁厚 mm | **0.5±0.05** |
| 每组试件数 | **3** |
| 试验温度 ℃ | **20±2** |
| 自收缩定义 | 无约束、无与外界介质交换 |

### 第 8 章 — 实测数据行（若要做训练/质控）

填 `template_ch08_shrinkage.csv`，**一行 = 一次试件观测序列或一个龄期点**：

| CSV 列 | 必填 | 角色 |
|--------|------|------|
| `standard_method_id` | 是 | `shrinkage_noncontact` / `contact` / `corrugated_pipe` |
| `source_doi` / `source_group` | 是 | 试验批次溯源 |
| `data_tier` | 是 | 建议 `A_lab_native` |
| `specimen_id` | 是 | 试件编号 |
| `age_h` 或 `age_d` | 是 | 观测龄期 |
| `length_change_mm` 或 `shrinkage_strain_ue` | 是 | **L0 量测** |
| `shrinkage_rate_formula` | 否 | L2 公式输出（禁止当标签） |
| `temperature_c`, `rh_pct` | 推荐 | 环境 |

---

## 四、第 9 章 — 早期抗裂试验（协议提取）

### 方法 ID：`early_cracking_plate`

| 提取字段 | 写入位置 | 说明 |
|----------|----------|------|
| 条文号 | `clause_ref` | 9.x.x |
| 平板尺寸 mm | `specimen_length_mm` × `width` × `thickness` | **须从 PDF 核对** |
| 骨料最大粒径限制 | `max_aggregate_mm` | |
| 约束方式 | `restraint_type` | 平板约束 / 环约束等 |
| 风场 | `wind_speed_ms` | 与侧栏 `wind_speed_ms` 可对齐 |
| 环境温度 ℃ | `test_temperature_c` | |
| 环境湿度 % | `test_rh_pct` | |
| 观测开始时刻 | `observation_start_h` | 如浇筑后 |
| 观测结束时刻 | `observation_end_h` | |
| 记录间隔 | `observation_interval_min` | |
| 裂缝判定 | `crack_criterion_note` | 可见裂缝/宽度阈值 |
| 统计指标 | 见下表 | 与主模型映射 |

### 第 9 章指标 → 主模型字段映射

| 50082 统计量（PDF 原文术语） | 建议协议列 | 主模型列 | 能否作 L4 标签 |
|------------------------------|------------|----------|----------------|
| 最大裂缝宽度 | `crack_width_mm` | `crack_width` | **是**（须量测溯源） |
| 单位面积裂缝条数/长度 | `crack_count` / `crack_length_mm` | 可派生 | 是 |
| 裂缝面积率 / 单位面积裂缝数目 | 派生 → | `crack_density` | **是**（定义须固定） |
| 开裂等级 / 是否开裂 | 派生 → | `cracking_risk` | **是**（规则须在 PDF 中固定） |
| 公式换算“理论裂缝” | — | — | **否** |

### 第 9 章 — 实测数据行

填 `template_ch09_early_cracking.csv`：

| CSV 列 | 必填 | 说明 |
|--------|------|------|
| `standard_method_id` | 是 | 固定 `early_cracking_plate` |
| `crack_width_definition_id` | 是 | 建议 `CW_MAX_SURFACE_MM`（与主链路一致） |
| `mix_design_ref` | 推荐 | 配合比编号 |
| `fiber_content`, `w_b_ratio`, … | 推荐 | 若与主模型联合分析，对齐 `FEATURE_COLUMNS` |
| `crack_width_mm` | 条件 | 有则填 |
| `crack_density_per_m2` | 条件 | 按 PDF 统计口径 |
| `cracking_risk` | 条件 | 0/1/2 须附分级规则 |
| `observation_time_h` | 推荐 | 裂缝出现/量测时刻 |

---

## 五、与 SteelFiberCrackPredictor 的接入点

| 产物 | 路径 |
|------|------|
| 方法注册表 | `config/gbt50082_methods.yaml` |
| 协议列 schema | `config/crack_protocol_schema.yaml` |
| 数据闸门 | `src/standards/gbt50082_data_gate.py` |
| 合并训练 | 实测行 + 治理列 → `data/training_data.csv` → `py scripts/retrain_world_db.py` |
| 可信度 Tab | `standard_method_id` 非空 → 分路径方法论加分 |

---

## 六、核对完成后

```bash
py scripts/validate_gbt50082_extraction_csv.py data/gbt50082/template_ch09_early_cracking.csv
# 更新 YAML 后
py -c "from src.standards.gbt50082_rules import list_method_ids; print(list_method_ids())"
```

---

## 七、版权提示

本清单仅列 **字段名与工程映射**，不复制标准正文。公式与表格数值须由持有人对照 **正版 PDF** 填写。
