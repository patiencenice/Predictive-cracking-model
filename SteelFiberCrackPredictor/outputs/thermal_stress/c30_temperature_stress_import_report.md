# C30 温度应力试验数据导入报告

**导入时间（UTC）：** 2026-05-23T11:58:33+00:00
**原始目录：** `data\thermal_stress\raw`

## 1. 成功读取的文件

### EXCEL（7）
- `data\thermal_stress\raw\温度应力\C30.xls`
- `data\thermal_stress\raw\温度应力\C30温度应力\2016-05-12-1122.xls`
- `data\thermal_stress\raw\温度应力\C30温度应力\C30\基准C30\2016-05-12-1122.xls`
- `data\thermal_stress\raw\温度应力\C30温度应力\C30\基准C30\C30基准72.xls`
- `data\thermal_stress\raw\温度应力\C30温度应力\C30\金隅C30\2016-05-23-1153.xls`
- `data\thermal_stress\raw\温度应力\C50新.xls`
- `data\thermal_stress\raw\温度应力\温度应力计算.xlsx`

### BMP（9）
- `data\thermal_stress\raw\温度应力\C30温度应力\0.36R0.bmp`
- `data\thermal_stress\raw\温度应力\C30温度应力\0.36R100.bmp`
- `data\thermal_stress\raw\温度应力\C30温度应力\0.36R50.bmp`
- `data\thermal_stress\raw\温度应力\C30温度应力\0.44R0.bmp`
- `data\thermal_stress\raw\温度应力\C30温度应力\0.44R100.bmp`
- `data\thermal_stress\raw\温度应力\C30温度应力\0.44R50.bmp`
- `data\thermal_stress\raw\温度应力\C30温度应力\0.48R0.bmp`
- `data\thermal_stress\raw\温度应力\C30温度应力\0.48R100.bmp`
- `data\thermal_stress\raw\温度应力\C30温度应力\0.48R50.bmp`

### OTHER（12）
- `data\thermal_stress\raw\温度应力\C30普通水泥混凝土.et`
- `data\thermal_stress\raw\温度应力\C30温度应力\C30\基准C30\UNTITLED.opj`
- `data\thermal_stress\raw\温度应力\C30温度应力\C30\金隅C30\UNTITLED.opj`
- `data\thermal_stress\raw\温度应力\C30温度应力\温度应力试验机说明书.pdf`
- `data\thermal_stress\raw\温度应力\C50.xls`
- `data\thermal_stress\raw\温度应力\C50井壁混凝土温度应力.opj`
- `data\thermal_stress\raw\温度应力\C50井壁混凝土温度应力20160622.opj`
- `data\thermal_stress\raw\温度应力\C50温度应力.opj`
- `data\thermal_stress\raw\温度应力\C70.xls`
- `data\thermal_stress\raw\温度应力\C70温度应力.opj`
- `data\thermal_stress\raw\温度应力\基准C30.opj`
- `data\thermal_stress\raw\温度应力\数据表头.xls`

## 2. Excel sheet 与列审计

### `data\thermal_stress\raw\温度应力\C30.xls`
- 读取：成功
- 解析分组：w/b=None，None
- Sheet `Sheet1`：0 行
- Sheet `Sheet2`：0 行
- Sheet `Sheet3`：0 行
- Sheet `Sheet4`：10842 行
- 字段映射：{"time_h": "时间", "specimen_temperature_c": "试件温度", "axial_stress_mpa": "轴向应力", "deformation_um": "累计复位变形"}

### `data\thermal_stress\raw\温度应力\C30温度应力\2016-05-12-1122.xls`
- 读取：成功
- 解析分组：w/b=None，None
- Sheet `Sheet1`：0 行
- Sheet `Sheet2`：0 行
- Sheet `Sheet3`：0 行
- Sheet `Sheet4`：13585 行
- 字段映射：{"time_h": "col_0", "specimen_temperature_c": "col_1", "axial_stress_mpa": "col_2", "deformation_um": "col_3"}

### `data\thermal_stress\raw\温度应力\C30温度应力\C30\基准C30\2016-05-12-1122.xls`
- 读取：成功
- 解析分组：w/b=None，None
- Sheet `Sheet1`：0 行
- Sheet `Sheet2`：0 行
- Sheet `Sheet3`：0 行
- Sheet `Sheet4`：13585 行
- 字段映射：{"time_h": "时间", "specimen_temperature_c": "试件温度", "axial_stress_mpa": "轴向应力", "deformation_um": "累计复位变形"}

### `data\thermal_stress\raw\温度应力\C30温度应力\C30\基准C30\C30基准72.xls`
- 读取：成功
- 解析分组：w/b=None，None
- Sheet `Sheet1`：0 行
- Sheet `Sheet2`：0 行
- Sheet `Sheet3`：0 行
- Sheet `Sheet4`：8263 行
- 字段映射：{"time_h": "时间", "specimen_temperature_c": "试件温度", "axial_stress_mpa": "轴向应力", "deformation_um": "累计复位变形"}

### `data\thermal_stress\raw\温度应力\C30温度应力\C30\金隅C30\2016-05-23-1153.xls`
- 读取：成功
- 解析分组：w/b=None，None
- Sheet `Sheet1`：0 行
- Sheet `Sheet2`：0 行
- Sheet `Sheet3`：0 行
- Sheet `Sheet4`：7816 行
- 字段映射：{"time_h": "时间", "specimen_temperature_c": "试件温度", "axial_stress_mpa": "轴向应力", "deformation_um": "累计复位变形"}

### `data\thermal_stress\raw\温度应力\C50新.xls`
- 读取：失败
- 错误：`'DataFrame' object has no attribute 'str'`
- 解析分组：w/b=None，None
- Sheet `Sheet4`：8527 行
- Sheet `Sheet1`：8527 行
- Sheet `Sheet2`：0 行
- Sheet `Sheet3`：0 行

### `data\thermal_stress\raw\温度应力\温度应力计算.xlsx`
- 读取：成功
- 解析分组：w/b=None，None
- Sheet `Sheet1`：13 行
- 字段映射：{"time_h": "nan", "specimen_temperature_c": "3d弹性模量E3", "axial_stress_mpa": "a", "deformation_um": "H(t)"}

## 3. 分组映射（3×3）

- **C30_wb0p36_R0** [explicit_map]：8263 点；T_max=35.6901；σ_t_max=-0.9408
- **C30_wb0p36_R50** [explicit_map]：8263 点；T_max=35.6901；σ_t_max=-0.9408
- **C30_wb0p36_R100** [explicit_map]：8263 点；T_max=35.6901；σ_t_max=-0.9408
- **C30_wb0p44_R0** [explicit_map]：13585 点；T_max=35.6901；σ_t_max=-0.9408
- **C30_wb0p44_R50** [explicit_map]：13585 点；T_max=35.6901；σ_t_max=-0.9408
- **C30_wb0p44_R100** [explicit_map]：13585 点；T_max=35.6901；σ_t_max=-0.9408
- **C30_wb0p48_R0** [explicit_map]：7816 点；T_max=35.2506；σ_t_max=-0.8556
- **C30_wb0p48_R50** [explicit_map]：7816 点；T_max=35.2506；σ_t_max=-0.8556
- **C30_wb0p48_R100** [explicit_map]：7816 点；T_max=35.2506；σ_t_max=-0.8556

## 4. 缺失分组（missing_groups）

- 无

## 5. 需人工复核


## 6. 未映射 Excel

- `data\thermal_stress\raw\温度应力\C30.xls`
- `data\thermal_stress\raw\温度应力\C30温度应力\2016-05-12-1122.xls`
- `data\thermal_stress\raw\温度应力\C50新.xls`
- `data\thermal_stress\raw\温度应力\温度应力计算.xlsx`

## 7. 输出产物

- `data\thermal_stress\c30_temperature_stress_timeseries.csv`（88992 行）
- `data\thermal_stress\c30_temperature_stress_summary.csv`（9 行）
- `data\thermal_stress\c30_group_source_map.json`
- `outputs\thermal_stress\c30_temperature_stress_import_report.json`
- 已导入 9 组，缺失 0 组

## 8. 边界说明

- 未并入 `training_data.csv`；未修改主模型训练/推理。
- `restraint_percent` ≠ `restraint_factor_R`（后续单独映射）。
- bmp 仅用于核对与展示，未作为主数据源抠数。
- **方式 A**：同名 stem 自动匹配；**方式 B**：`c30_group_source_map.json`。
- 方式 A 禁止同一 xls 重复分配；方式 B 允许映射文件显式共用同一 xls。
