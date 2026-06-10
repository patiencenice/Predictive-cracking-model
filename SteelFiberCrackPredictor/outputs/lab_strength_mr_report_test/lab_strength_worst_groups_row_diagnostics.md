# 最差 source_group 逐行 OOF 诊断

列：y_true、formula_pred、ridge_pred、hgb_pred 及各自 pred−true。

## compressive — `XUkun/C30_BASE`

OOF 表中未找到该组

## flexural — `XUkun/C41_BASE`

### 组内系统性偏差（mean(pred−true)）

- **formula_only**: mean=-2.1938976507500696 — 均值<0：该组内该模型整体偏低。
- **ridge**: mean=-0.6426424910153301 — 均值<0：该组内该模型整体偏低。
- **hgb**: mean=-1.9593113832220723 — 均值<0：该组内该模型整体偏低。

### 逐行

| row_id | fold | y_true | formula | ridge | hgb | f−y | r−y | h−y |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 35 | 4 | 7.240000 | 5.046102 | 6.597358 | 5.280689 | -2.193898 | -0.642642 | -1.959311 |
