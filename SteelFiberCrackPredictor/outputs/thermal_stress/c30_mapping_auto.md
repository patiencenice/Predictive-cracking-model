# C30 温度应力自动映射报告（数据库综合）

**生成时间（UTC）：** 2026-05-23T11:58:33+00:00

## 1. 结论

- 已自动对应 **9** 组 bmp → **3** 份独立 xls
- w/b：由目录结构 + 曲线指纹 + 训练表 w/b 档校验
- R 档：来自 bmp 文件名（库内无 9 条独立应力曲线）

## 2. w/b 档 → xls

- **w/b=0.36** → `data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/C30基准72.xls` （置信度 0.99）— 文件名72≈73h；为2016-05-12长试验截断前缀；基准C30子目录
- **w/b=0.44** → `data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/2016-05-12-1122.xls` （置信度 0.93）— 完整长试验≈120h；有表头Sheet4；与基准72同曲线前缀
- **w/b=0.48** → `data/thermal_stress/raw/温度应力/C30温度应力/C30/金隅C30/2016-05-23-1153.xls` （置信度 0.96）— 金隅C30独立子目录；唯一较低T_max与σ_t指纹

## 3. 九组明细

- **C30_wb0p36_R0** ← `C30温度应力/0.36R0.bmp` → `data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/C30基准72.xls` (w/b 0.99, R=R0 来自 bmp)
- **C30_wb0p36_R100** ← `C30温度应力/0.36R100.bmp` → `data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/C30基准72.xls` (w/b 0.99, R=R100 来自 bmp)
- **C30_wb0p36_R50** ← `C30温度应力/0.36R50.bmp` → `data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/C30基准72.xls` (w/b 0.99, R=R50 来自 bmp)
- **C30_wb0p44_R0** ← `C30温度应力/0.44R0.bmp` → `data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/2016-05-12-1122.xls` (w/b 0.93, R=R0 来自 bmp)
- **C30_wb0p44_R100** ← `C30温度应力/0.44R100.bmp` → `data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/2016-05-12-1122.xls` (w/b 0.93, R=R100 来自 bmp)
- **C30_wb0p44_R50** ← `C30温度应力/0.44R50.bmp` → `data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/2016-05-12-1122.xls` (w/b 0.93, R=R50 来自 bmp)
- **C30_wb0p48_R0** ← `C30温度应力/0.48R0.bmp` → `data/thermal_stress/raw/温度应力/C30温度应力/C30/金隅C30/2016-05-23-1153.xls` (w/b 0.96, R=R0 来自 bmp)
- **C30_wb0p48_R100** ← `C30温度应力/0.48R100.bmp` → `data/thermal_stress/raw/温度应力/C30温度应力/C30/金隅C30/2016-05-23-1153.xls` (w/b 0.96, R=R100 来自 bmp)
- **C30_wb0p48_R50** ← `C30温度应力/0.48R50.bmp` → `data/thermal_stress/raw/温度应力/C30温度应力/C30/金隅C30/2016-05-23-1153.xls` (w/b 0.96, R=R50 来自 bmp)

## 4. 数据库证据

- training_data w/b≈0.36：25 行
- training_data w/b≈0.44：23 行
- training_data w/b≈0.48：21 行
- 曲线前缀：0.36 为 0.44 同试验截断 — 已验证 (max Δσ=0.0 MPa)

## 5. 局限

- 文献库无温度应力 3×3 矩阵，R 档不能从 xls 内容区分
- 同 w/b 三档 R 共用一条仪器应力曲线，R 仅写入元数据供公式 σ=R·E·α·ΔT
- C30.xls 为对照试件（高 T_max/大拉应力），未纳入九组矩阵
