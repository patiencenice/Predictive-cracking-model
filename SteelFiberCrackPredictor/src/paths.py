"""项目根目录与常用绝对路径（避免从其它工作目录启动时找不到 models/config）。"""

from __future__ import annotations

from pathlib import Path

# src/paths.py -> 上级目录为项目根
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = PROJECT_ROOT / "models"
CONFIG_YAML: Path = PROJECT_ROOT / "config" / "model_config.yaml"
# 训练与评估导出的图表、报表、完整 Pipeline 等（与 models/ 下推理用拆分 pkl 互补）
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
