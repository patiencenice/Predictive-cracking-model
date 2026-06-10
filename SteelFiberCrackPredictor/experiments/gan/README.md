# GAN Research Experiment (Phase 1)

## 状态：已冻结（2026-04-30）

- **冻结日期**：2026 年 4 月 30 日  
- **冻结原因**：在 `experiments/gan/` 内已完成 CTGAN 训练—生成—闸门—质量检查的迭代；X-only 虽可将物理闸门通过率提升至 120/120，但 **`memorization_risk` 仍为 high**（`synthetic_internal_repeat.near_duplicate_ratio` 约 0.84，模式塌缩明显），且 **`distribution_shift_risk` 仍为 high**（关键材料列相对真实集偏移显著）。**通过闸门/后处理带来的 throughputs 提升，不等于 synthetic 具备主结论对照所需的可信度。**  
- **当前结论（须遵守）**：**synthetic 样本不得并入主训练集、不得进入主 OOF、不得用于主结论对照**；本目录及 `outputs/` 下相关 CSV/JSON **仅作历史记录与复盘**，默认视为研究废弃路径，直至有新的数据与假设并另行立项。  
- **范围**：不进入第三阶段对照实验；不实现 `assemble_experiment_sets.py`、`train_eval_runner.py`、`report_builder.py`、`scripts/run_gan_experiment.py`（按冻结决议保持不扩展）。

第一版仅针对 `compressive_true` / `flexural_true`，不覆盖 `crack_width` / `crack_density` / `cracking_risk`。本目录只用于 `lab_strength_residual` 的研究性数据增强实验，不改主训练、主评估、主 OOF、主推理链路。

## Scope and Boundary

- 仅在 `experiments/gan/` 下开发与运行。
- 只读取现有真实数据，不新增假数据，不人工改写标签。
- 不修改主链路文件：`train_model.py`、`predictor.py`、`app.py`、`visualizer.py`、`features.py`、`evaluate.py`、`shap_analysis.py`、`src/lab_formula_gb.py`。
- 本阶段仅落地：
  - `config/gan_experiment.yaml`
  - `src/load_real_data.py`
  - `src/protocol_filter.py`
  - `src/physical_gate.py`

## Main Evaluation Set (Frozen Definition in V1)

第一版主评估集定义固定如下（必须同时满足）：

- `source_domain = fiber`
- `data_tier = A`
- `eligible_for_main_oof = 1`
- `needs_manual_review != 1`

说明：主指标只在该真实样本子集上计算。synthetic 样本不得进入主 OOF 与主指标计算。

## Data and Label Rules

- GAN 训练输入只能来自真实、可追溯、协议闭合样本。
- 允许区分 `source_domain = ordinary` 与 `source_domain = fiber`。
- Tier A 为默认训练来源；Tier B 仅可选；Tier C 不允许进入 GAN 训练。
- 禁止人工插值、平滑、趋势补点来构造 `compressive_true` / `flexural_true`。

## Gate Rules (Implemented in Phase 1)

- 协议闸门（`protocol_filter.py`）：
  - 必要列存在性检查
  - 关键字段缺失检查
  - 枚举/离散字段合法性检查（可配置）
  - 主评估集规则字段可解析性检查
- 物理闸门（`physical_gate.py`）：
  - `compressive_true > 0`
  - `flexural_true > 0`
  - `flexural_true < compressive_true`
  - `w_b_ratio` 合理范围
  - 材料用量非负
  - `fiber_content` 合理范围
  - `aspect_ratio` 合理范围

未通过样本必须剔除并记录原因，供后续报告汇总。

## What Is Not Included Yet

本阶段不包含以下内容（下一阶段再接）：

- CTGAN 训练与采样
- synthetic 固定打标
- memorization check
- 三组实验集组装
- 训练评估 runner 与报告汇总
