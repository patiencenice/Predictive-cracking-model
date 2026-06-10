# 主开裂模型权重与离线指标一致性自检

**检查时间：** 2026-05-22  
**类型：** 只读自检（未重训、未改代码、未写回训练数据）  
**机器路径：** `D:\cursor\code\SteelFiberCrackPredictor`

---

## 1. 检查范围

| 类别 | 路径 |
|------|------|
| 权重 | `models/crack_regressor.pkl`、`crack_density_regressor.pkl`、`crack_classifier.pkl`、`feature_scaler.pkl` |
| 指标 | `models/training_metrics.json`、`models/cv_metrics.json`、`models/feature_importance.json` |
| 指标副本 | `outputs/training_metrics.json` |
| 加载逻辑 | `src/predictor.py`、`src/model_bootstrap.py` |
| 训练数据行数 | `data/training_data.csv`（只读计数） |
| 只读预测 | `PREDICTION_INPUT_DEFAULTS` → `validate_and_transform` → `predict_all` |

---

## 2. `models/` 文件存在性

| 文件 | 存在 | 大小 | 修改时间 |
|------|------|------|----------|
| `crack_regressor.pkl` | 是 | 384,418 B | 2026-04-03 20:07:44 |
| `crack_density_regressor.pkl` | 是 | 444,483 B | 2026-04-03 20:07:44 |
| `crack_classifier.pkl` | 是 | 838,327 B | 2026-04-03 20:07:44 |
| `feature_scaler.pkl` | 是 | 1,895 B | 2026-04-03 20:07:44 |
| `training_metrics.json` | 是 | 460 B | 2026-04-03 20:07:44 |
| `cv_metrics.json` | 是 | 3,592 B | 2026-04-03 19:33:39 |
| `feature_importance.json` | 是 | 3,537 B | 2026-04-03 20:07:44 |

**结论：** 本次检查机器上 **四个必需 pkl 均存在**；与先前“仓库快照无 pkl”不同，以 **本机 `models/` 实际内容** 为准。

---

## 3. Predictor 加载路径与 bootstrap 风险

**加载顺序（`SteelFiberCrackPredictor.__init__`）：**

1. `ensure_default_models(model_dir)`
2. `_load_models()` → `joblib.load` 四个 pkl

**`ensure_default_models()` 触发条件：**

- 任一必需 pkl 缺失，或
- `feature_scaler.pkl` 特征维数 ≠ `len(FEATURE_COLUMNS)`

**触发后行为：**

- 调用 `_synthetic_dataset(n=2800)` 生成 **合成演示数据**
- 调用 `fit_models_save(...)` 写入 pkl，并 **覆盖** `training_metrics.json` / `feature_importance.json`

**本次检查时刻：** `ensure_default_models_would_run = false`（pkl 齐全且 scaler 维数一致）。

**风险（新环境 / 删 pkl 后）：** 首次打开 App 会 **静默用合成数据训练**，页面预测与历史「120 行真实 CSV」指标 **可能脱节**。

---

## 4. `training_metrics.json` 与 pkl 是否同源

### 4.1 JSON 内元数据

| 字段 | 是否存在 |
|------|----------|
| `data_path` | **否** |
| `trained_at` / 版本号 | **否** |
| 模型哈希 | **否** |

**无法从 JSON  alone 做 cryptographic 级同源证明。**

### 4.2 间接证据（弱确认）

| 证据 | 结果 |
|------|------|
| `n_train` / `n_test` | **96 / 24** → 与 `training_data.csv` **120 行**、默认 `test_size=0.2` **一致** |
| 合成 bootstrap 默认 2800 行 | 约 `n_train≈2240` → **与当前 96/24 不一致** |
| pkl + `training_metrics.json` + `feature_importance.json` 修改时间 | **均为 2026-04-03 20:07:44**（同一批次） |
| `models/` vs `outputs/training_metrics.json` | **内容完全一致** |
| `cv_metrics.json` 时间 | **19:33:39**（更早）→ **另一次** `cross_validate` 运行，**不保证**与当前 pkl 同批 |

**同源判定：** `weak_yes`（时间戳 + 划分规模 + 非合成规模）  
**不一致风险：** `medium`（缺 provenance 字段；缺 pkl 时会 bootstrap；CV 指标与 hold-out 指标不同次运行）

---

## 5. 只读预测自检（默认侧栏输入）

**输入：** `PREDICTION_INPUT_DEFAULTS`（与 App 默认一致）

| 项目 | 结果 |
|------|------|
| `validate_and_transform` | 通过 |
| `intermediate.crack_density_source` | **`regressor`** |
| 是否 fallback | **否** |
| 裂缝宽度来源 | `crack_regressor.pkl`（回归输出，裁剪后） |
| 开裂风险来源 | `crack_classifier.pkl`（`predict_proba` + 类权重） |
| 示例输出 | 缝宽 ≈ 0.117 mm；密度 ≈ 0.65 条/m²；P ≈ 0.81；**高风险** |

**说明：** 主模型 **未暴露** `crack_width_source` / `cracking_risk_source` 字段；上表按代码路径推断。

**本次自检是否触发 bootstrap：** **否**（检查前后 pkl 时间戳未变）。

---

## 6. 页面离线指标是否可能与实际预测脱节

| 场景 | 风险 |
|------|------|
| **本机（pkl 齐全，本次）** | **低**：推理权重与 `training_metrics.json` 弱同源；密度走回归器非 fallback |
| **新克隆 / 删 pkl / 维数变更** | **高**：`ensure_default_models` → 合成训练 → 指标被覆盖 |
| **展示 `cv_metrics.json` 当作“当前模型”** | **中**：CV 为 19:33 批次，与 20:07 pkl 批次 **未必同一权重** |

**是否建议继续展示 hold-out `training_metrics`：**  
**可以**，但须标注「离线 hold-out / 历史训练批次 / 非当前输入实时误差」。

**是否建议展示 `cv_metrics`：**  
**可以作研发参考**，不应与「当前加载权重」默认同源。

---

## 7. 建议的下一步（P0 结论）

| 优先级 | 建议 |
|--------|------|
| **现在** | **继续数据治理（P0）**：填 `crack_governance_fill_template.csv`；在 `tier_A > 0` 且分组可审计前 **不重训**（与阶段文档一致） |
| **本机** | **不必为对齐指标立即重训**：pkl 与 `training_metrics` 已弱同源；本次预测未 fallback |
| **部署/协作** | **固定权重版本**：将四 pkl + `training_metrics.json` 作为一组产物归档（建议未来训练写入 `data_path`/`trained_at`） |
| **缺 pkl 环境** | **先** `py -m src.train_model --csv data/training_data.csv` **再** 对外展示离线指标（本轮自检 **未执行** 重训） |

---

## 8. 附：hold-out 指标摘要（`models/training_metrics.json`）

| 任务 | test R² / 准确率 |
|------|------------------|
| 裂缝宽度 | R² ≈ 0.489 |
| 裂缝密度 | R² ≈ **-0.077** |
| 开裂风险 | accuracy ≈ **0.50** |

---

*本报告由只读脚本生成，对应 `outputs/model_integrity/model_loading_integrity_report.json`。*
