"""生成 data/lab_strength_training.example.csv（公式基线 + 噪声真值，便于跑通训练脚本）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.lab_strength_residual.lab_mix_features import (
    LAB_MIX_EXTRA_FEATURE_NAMES,
    LAB_STRENGTH_FEATURE_COLUMNS,
    lab_mix_extra_row_vector,
)
from src.lab_experiment import LOADING_COMPRESSION, LOADING_FLEXURAL, SPECIMEN_TYPES
from src.lab_formula_gb import (
    compressive_formula_pred_mpa,
    flexural_formula_pred_mpa,
)


def main() -> None:
    base = pd.read_csv(_ROOT / "data" / "training_data.example.csv", nrows=24)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(len(base)):
        r = base.iloc[i].copy()
        spec = SPECIMEN_TYPES[0]
        fcu = float(r["cube_strength_mpa"])
        fc, _ = compressive_formula_pred_mpa(
            spec,
            cube_edge_mm=150.0,
            prism_b_mm=150.0,
            prism_h_mm=150.0,
            prism_l_mm=300.0,
            loading_compression=LOADING_COMPRESSION[0],
            cube_strength_mpa=fcu,
        )
        ff, _ = flexural_formula_pred_mpa(
            spec,
            beam_b_mm=100.0,
            beam_h_mm=100.0,
            beam_span_mm=400.0,
            loading_flexural=LOADING_FLEXURAL[0],
            cube_strength_mpa=fcu,
            fiber_content_pct=float(r["fiber_content"]),
        )
        r["compressive_true"] = fc + float(rng.normal(0, 1.2))
        r["flexural_true"] = ff + float(rng.normal(0, 0.06))
        r["source_group"] = f"G{i % 5}"
        rows.append(r)

    out = pd.DataFrame(rows)

    for i in range(len(out)):
        mv = lab_mix_extra_row_vector(out.iloc[i])
        for name, val in zip(LAB_MIX_EXTRA_FEATURE_NAMES, mv):
            out.loc[out.index[i], name] = val

    cols = [
        c
        for c in out.columns
        if c in LAB_STRENGTH_FEATURE_COLUMNS
        or c in ("compressive_true", "flexural_true", "source_group")
    ]
    out = out[cols]
    out_path = _ROOT / "data" / "lab_strength_training.example.csv"
    out.to_csv(out_path, index=False)
    print("wrote", out_path, out.shape)


if __name__ == "__main__":
    main()
