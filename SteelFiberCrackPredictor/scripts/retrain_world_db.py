# -*- coding: utf-8 -*-
"""一键：世界数据库 → 文献合并 → 主开裂/力学/温度应力重训 → 治理诊断。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    py = sys.executable
    run(
        [
            py,
            "run_literature_pipeline.py",
            "--user-csv",
            "data/training_data.csv",
            "--raw-input",
            "data/literature/example_raw_extracted.csv",
            "--auto-annotate-user",
        ]
    )
    run([py, "-m", "src.train_model", "--csv", "outputs/training_data_merged.csv"])
    run(
        [
            py,
            "train_lab_strength_residual.py",
            "--data",
            "data/lab_strength_training_merged.csv",
            "--out",
            "outputs/lab_strength",
            "--save-models",
        ]
    )
    run([py, "train_thermal_stress_residual.py", "--granularity", "point", "--save-models"])
    run([py, "-m", "src.cross_validate", "--csv", "outputs/training_data_merged.csv", "--folds", "5"])
    merged = ROOT / "outputs" / "training_data_merged.csv"
    target = ROOT / "data" / "training_data.csv"
    target.write_bytes(merged.read_bytes())
    run([py, "scripts/diagnose_crack_training_governance.py"])
    print("\n完成。模型: models/  指标: models/training_metrics.json")


if __name__ == "__main__":
    main()
