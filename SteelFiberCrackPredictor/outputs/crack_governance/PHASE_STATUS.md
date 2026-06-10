# 主开裂数据治理 — 阶段状态（检查点）

**更新日期：** 2026-05-22  
**阶段：** 人工补值（暂停新脚本 / 新训练 / grouped evaluation）

---

## 阶段结论

**主开裂数据治理工具链已完成，数据本体仍未人工闭合；在 A 类候选样本出现之前，不进行模型重训。**

**P0 模型权重自检（2026-05-22，已确认）：** 当前本机推理权重可用，未触发合成 bootstrap；离线指标可作为历史 hold-out 参考，但由于训练元数据不足，不能强证明与 pkl 完全同源。主开裂模型下一步仍应**优先推进治理模板人工补值**，而不是重训。

---

## 已完成（工具链）

| 步骤 | 产物 / 脚本 |
|------|-------------|
| 治理列结构 | `data/training_data.csv` 已追加 7 列空 sidecar 列（不进 `FEATURE_COLUMNS`） |
| 只读诊断 | `scripts/diagnose_crack_training_governance.py` → `crack_training_governance.json` |
| 人工模板 | `outputs/crack_governance/crack_governance_fill_template.csv` |
| 填写说明 | `outputs/crack_governance/crack_governance_fill_guide.md` |
| 安全回并 | `scripts/apply_crack_governance_fill_template.py`（支持 `--dry-run`） |
| 列初始化 | `scripts/init_crack_governance_columns.py` |
| 模板导出 | `scripts/export_crack_governance_fill_template.py` |

回并写入主表后，默认自动重跑 `diagnose_crack_training_governance.py`（可用 `--skip-diagnose` 跳过）。

---

## 当前数据状态（补值前）

- 治理列：**结构在、内容空**
- 诊断快照：`hold_pending = 120`，`tier_A_candidate = 0`，`source_group` 非空行 = 0
- **不宜重训**；**不进入 grouped evaluation**

---

## 人工补值流程（由操作者执行）

1. 在 Excel 中编辑：`outputs/crack_governance/crack_governance_fill_template.csv`
2. 参考：`crack_governance_fill_guide.md`
3. 校验（不写盘）：
   ```bash
   py scripts/apply_crack_governance_fill_template.py --dry-run
   ```
4. 回并主表：
   ```bash
   py scripts/apply_crack_governance_fill_template.py
   ```
5. 查看更新后的 `crack_training_governance.json`，确认 A/B/C/暂缓计数与 `group_audit`

---

## 本阶段明确不做

- 新训练脚本、GroupKFold 重训脚本
- 新 UI、新模型、新特征、新诊断链路
- 修改 `train_model.py` / `features.py` / `FEATURE_COLUMNS` / `predictor.py`
- 自动标 A 类、自动填 `source_group` / `crack_width_definition_id`
- 模型重训（直至 `tier_A_candidate > 0` 且分组可审计）
- **温度应力 Phase 1 公式扩展或接入主训练**（模块已冻结，见 `experiments/thermal_stress/README.md`）

---

## 温度应力模块（Phase 1，并行冻结）

**状态（2026-05-22）：** 工程解释链已通过验收并冻结扩展。

链式：**温度变化 → 温度应变 → 约束温度应力（σ_T*）→ η → 开裂风险解释**；η **不替代** 主模型 `cracking_risk` 概率 P。

**待真实数据（温度 / 裂缝 / 强度）后再做：** η–`crack_width` 关系、η–`cracking_risk` 相关性、η 阈值校准、解释增益验证。详见 `experiments/thermal_stress/README.md`。

### C30 温度应力验证集导入（独立管线，2026-05-23）

- **状态：** 9/9 组已导入；**公式+残差** 点级训练已完成（独立管线，不进主模型）
- **映射：** w/b 三档各共用 1 份 xls（基准72 / 基准C30·2016-05-12 / 金隅2016-05-23）；R 档来自 bmp 标签 → `restraint_percent/100`
- **训练产物：** `outputs/thermal_stress/residual_model/`（点级 HGB）、`data/thermal_stress/c30_thermal_stress_training_point.csv`
- **注意：** 同 w/b 下 R0/R50/R100 共用仪器曲线，y_true 相同、仅公式 R 不同；CV 按 `source_file` 分组（3 折）
- **命令：** `py scripts/auto_map_c30_from_database.py --apply --reimport` → `py scripts/build_thermal_stress_training_csv.py` → `py train_thermal_stress_residual.py --save-models`
- **自动映射报告：** `outputs/thermal_stress/c30_mapping_auto.{md,json}`

---

## 进入下一阶段的门槛（建议）

- `tier_A_candidate` > 0（诊断 JSON）
- `source_group` 非空且 `group_audit.n_groups` 合理
- 无非法 `data_tier` / `crack_width_definition_id` / `needs_manual_review`

满足后再讨论：评估可信（grouped OOF）或训练策略；此前保持主链路不变。

---

## 模型权重部署与协作说明

部署或协作交付主开裂模型时，请将以下文件作为**同一模型版本**一并归档，不得拆散：

- `models/crack_regressor.pkl`
- `models/crack_density_regressor.pkl`
- `models/crack_classifier.pkl`
- `models/feature_scaler.pkl`
- `models/training_metrics.json`（及同步副本 `outputs/training_metrics.json` 若存在）

**缺少任一 pkl 的环境不得直接对外展示离线 hold-out 指标**（指标与权重可能不对应）。若环境中缺少 pkl，应**先基于真实 CSV** 执行 `py -m src.train_model --csv data/training_data.csv` 训练并固定权重，再展示指标。

**禁止**依赖 App 首次启动时的 `ensure_default_models()` 静默合成 bootstrap 作为正式生产模型；该路径仅用于本地演示，会覆盖 pkl 与 metrics，与历史真实 CSV 训练结果脱节。

详见：`outputs/model_integrity/model_loading_integrity_report.md`。
