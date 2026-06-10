# 第一批文献抽取计划（paper_01 ~ paper_04）

## 对应文献与用途

### paper_01.csv
- Saradar A., Gandomi A.H., et al. (2018)
- Title: Restrained Shrinkage Cracking of Fiber-Reinforced High-Strength Concrete
- 用途：第一优先级，主抽样文献

### paper_02.csv
- Sunaga D., Sasaki H., et al. (2020)
- Title: Crack width evaluation of fiber-reinforced cementitious composites
- 用途：第二优先级，服务 crack_width 主线

### paper_03.csv
- 中文文献：钢纤维混凝土裂缝宽度计算方法研究
- 用途：中文背景 + 可尝试抽表

### paper_04.csv
- Banthia N., Gupta R. (1996)
- Title: Restrained Shrinkage Cracking in Fiber Reinforced Concrete—A Novel Test Technique
- 用途：方法学文献，优先提取试验定义与标签口径

## 统一规则
1. 当前训练主线只保留 crack_width。
2. crack_width_unit 统一为 mm。
3. crack_width_definition_id 当前默认使用 CW_MAX_SURFACE_MM。
4. source_group 建议写成 DOI/T3、DOI/T4、DOI/SERIES_A 这种格式。
5. 只从论文表格或明确数值结果中整理结构化数据，不读取 PDF 正文作为模型特征。
6. 所有 CSV 文件使用 UTF-8 with BOM（utf-8-sig）保存，方便 Excel 打开。
7. 如发现文件已存在，先检查内容再决定是否覆盖。
