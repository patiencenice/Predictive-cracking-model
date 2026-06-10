# C30 温度应力 raw 数据分析（含文献库对照）

**分析时间（UTC）：** 2026-05-23T11:31:35+00:00

## 1. 结论摘要

- **bmp：** 9 张（3×3 矩阵，文件名可解析 w/b 与 R 档）
- **独立 C30 时序 xls 指纹：** 5 条（≠ 9）
- **文献库：** 无温度应力试验矩阵；`data/literature` 仅服务 crack_width 抽取
- **9 张 bmp 各不相同，但独立 C30 时序 xls 仅 5 条指纹；R0/R50/R100 三档约束无法从现有 xls 文件名区分，文献库亦无温度应力矩阵可对表。**

## 2. 文献库对照（间接）

C30 温度应力 3×3 矩阵为独立实验 ground truth；与 data/literature 无直接行级映射，仅 w_b_ratio 量纲可与主训练表对照。

- `data\literature\example_raw_extracted.csv`：w/b ∈ [0.38, 0.4]，中位数 0.39（裂缝/主训练表 w_b_ratio，非温度应力原始试验）
- `data\training_data.csv`：w/b ∈ [0.3015, 0.4991]，中位数 0.3993（裂缝/主训练表 w_b_ratio，非温度应力原始试验）

## 3. 独立 xls 指纹目录

- `data/thermal_stress/raw/温度应力/C30.xls`：10843 行，时长 96.362 h，T_max 37.8148 ℃，σ [-1.665, 0.3155] MPa
- `data/thermal_stress/raw/温度应力/C30温度应力/2016-05-12-1122.xls`：13585 行，时长 120.747 h，T_max 35.6901 ℃，σ [-0.9408, 0.0967] MPa
- `data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/2016-05-12-1122.xls`：13586 行，时长 120.747 h，T_max 35.6901 ℃，σ [-0.9408, 0.0967] MPa
- `data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/C30基准72.xls`：8264 行，时长 73.432 h，T_max 35.6901 ℃，σ [-0.9408, 0.0967] MPa
- `data/thermal_stress/raw/温度应力/C30温度应力/C30/金隅C30/2016-05-23-1153.xls`：7817 行，时长 69.459 h，T_max 35.2506 ℃，σ [-0.8556, 0.0956] MPa

## 4. w/b 弱假设（不可自动导入）

- w/b≈0.48 ← `data/thermal_stress/raw/温度应力/C30温度应力/C30/金隅C30/2016-05-23-1153.xls` （置信度 **low**）：金隅C30 子目录独立试验；时长较短(≈69h)；可能与较高 w/b 档对应（待实验记录确认）
- w/b≈0.44 ← `data/thermal_stress/raw/温度应力/C30温度应力/2016-05-12-1122.xls` （置信度 **low**）：C30温度应力 根目录与 基准C30 同指纹长试验(≈120h)；可能为中档 w/b（待确认）
- w/b≈0.36 ← `data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/C30基准72.xls` （置信度 **low**）：文件名含「72」、时长≈73h，为 2016-05-12 同试验截断版；可能为低 w/b（待确认）
- w/b≈None ← `data/thermal_stress/raw/温度应力/C30.xls` （置信度 **low**）：根目录独立长试验(≈96h)，T_max 更高、拉应力更大；可能为标定/对照试件，未必在 3×3 矩阵内

## 5. 九组状态

- **C30_wb0p36_R0**：stem 同名 xls=无，显式映射=无 → **pending_manual_mapping**
- **C30_wb0p36_R50**：stem 同名 xls=无，显式映射=无 → **pending_manual_mapping**
- **C30_wb0p36_R100**：stem 同名 xls=无，显式映射=无 → **pending_manual_mapping**
- **C30_wb0p44_R0**：stem 同名 xls=无，显式映射=无 → **pending_manual_mapping**
- **C30_wb0p44_R50**：stem 同名 xls=无，显式映射=无 → **pending_manual_mapping**
- **C30_wb0p44_R100**：stem 同名 xls=无，显式映射=无 → **pending_manual_mapping**
- **C30_wb0p48_R0**：stem 同名 xls=无，显式映射=无 → **pending_manual_mapping**
- **C30_wb0p48_R50**：stem 同名 xls=无，显式映射=无 → **pending_manual_mapping**
- **C30_wb0p48_R100**：stem 同名 xls=无，显式映射=无 → **pending_manual_mapping**

## 6. 建议下一步

- 对照实验记录/Origin 工程，确认每个 w/b 对应哪一份日期 xls
- 确认 R0/R50/R100 是否各有独立 xls（当前目录可能缺失 6 份）
- 在 c30_group_source_map.json 填写 9 行路径后运行 import_c30_temperature_stress.py
- 或把 xls 重命名为 0.36R0.xls 等与 bmp 同名（方式 A）
