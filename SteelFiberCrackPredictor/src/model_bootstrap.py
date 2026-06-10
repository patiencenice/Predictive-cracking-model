"""
在无现成 pkl 时，用合成数据训练一套可运行的演示模型（便于直接启动 Web）。
真实工程请用 data/training_data.csv 运行 train_model.py 覆盖 models/。

当 FEATURE_COLUMNS 维度与已有 scaler 不一致时会自动删除旧 pkl 并重新训练。
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    STRENGTH_GRADE_ORDER,
    STRENGTH_GRADE_TO_MPA,
)
from src.paths import CONFIG_YAML
from src.train_utils import fit_models_save, load_model_config

_MODEL_FILES = (
    "crack_regressor.pkl",
    "crack_density_regressor.pkl",
    "crack_classifier.pkl",
    "feature_scaler.pkl",
)


def _scaler_feature_count(model_dir: Path) -> int | None:
    p = model_dir / "feature_scaler.pkl"
    if not p.exists():
        return None
    try:
        sc = joblib.load(p)
        n = getattr(sc, "n_features_in_", None)
        return int(n) if n is not None else None
    except Exception:
        return None


def _remove_stale_models(model_dir: Path) -> None:
    for name in _MODEL_FILES:
        fp = model_dir / name
        if fp.exists():
            fp.unlink()


def _tensile_for_material(material_enc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """按材质采样更贴近工程区间的抗拉强度 (MPa)。"""
    n = material_enc.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        m = int(material_enc[i])
        if m == 0:  # 钢纤维
            out[i] = rng.uniform(800, 2200)
        elif m == 1:  # 玄武岩
            out[i] = rng.uniform(1200, 2800)
        elif m == 2:  # 聚丙烯
            out[i] = rng.uniform(300, 750)
        else:  # 玻璃纤维
            out[i] = rng.uniform(1000, 2600)
    return out


def _synthetic_dataset(n: int = 2800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fiber_material_enc = rng.integers(0, 4, n)
    fiber_content = rng.uniform(0.5, 3.0, n)
    aspect_ratio = rng.uniform(30, 100, n)
    tensile_strength = _tensile_for_material(fiber_material_enc, rng)
    fiber_type_enc = rng.integers(0, 3, n)
    w_b_ratio = rng.uniform(0.3, 0.5, n)
    n_grades = len(STRENGTH_GRADE_ORDER)
    strength_grade_enc = rng.integers(0, n_grades, n)
    cube_strength_mpa = np.array(
        [STRENGTH_GRADE_TO_MPA[STRENGTH_GRADE_ORDER[int(i)]] for i in strength_grade_enc],
        dtype=np.float64,
    )
    concrete_type_enc = rng.integers(0, 4, n)
    cement_content = rng.uniform(300, 550, n)
    sand_ratio = rng.uniform(30, 55, n)
    sand_type_enc = rng.integers(0, 2, n)
    sand_content = rng.uniform(560, 920, n)
    stone_content = rng.uniform(950, 1280, n)
    fly_ash = rng.uniform(0, 120, n)
    slag_powder = rng.uniform(0, 100, n)
    slag_grade_enc = rng.integers(0, 3, n)
    binder_min = cement_content + fly_ash + slag_powder
    binder_content = binder_min + rng.uniform(0, 40, n)
    admixture_enc = rng.integers(0, 3, n)
    mixing_water = w_b_ratio * binder_content * rng.uniform(0.92, 1.08, n)
    admixture_dosage = np.where(
        admixture_enc == 0,
        rng.uniform(0.0, 0.35, n),
        rng.uniform(1.0, 8.5, n),
    )
    curing_days = rng.integers(1, 91, n)
    temperature = rng.uniform(5, 40, n)
    humidity = rng.uniform(30, 100, n)
    casting_method_enc = rng.integers(0, 2, n)
    fiber_content_x_aspect_ratio = fiber_content * aspect_ratio

    # 材质对裂缝的简化影响（演示启发式）：钢/玄武岩桥接更好；PP 以阻裂为主易显宽缝；玻纤脆性略增敏感
    mat_w = np.zeros(n, dtype=np.float64)
    mat_w[fiber_material_enc == 0] = 0.0
    mat_w[fiber_material_enc == 1] = -0.025
    mat_w[fiber_material_enc == 2] = 0.055
    mat_w[fiber_material_enc == 3] = 0.03

    mat_d = np.zeros(n, dtype=np.float64)
    mat_d[fiber_material_enc == 2] = 0.35
    mat_d[fiber_material_enc == 3] = 0.15
    mat_d[fiber_material_enc == 1] = -0.12

    ct_w = np.zeros(n, dtype=np.float64)
    ct_w[concrete_type_enc == 2] = -0.018
    ct_w[concrete_type_enc == 3] = 0.028
    ct_d = np.zeros(n, dtype=np.float64)
    ct_d[concrete_type_enc == 3] = 0.22
    ct_d[concrete_type_enc == 1] = -0.08

    X = pd.DataFrame(
        {
            "fiber_content": fiber_content,
            "aspect_ratio": aspect_ratio,
            "tensile_strength": tensile_strength,
            "fiber_type_enc": fiber_type_enc.astype(np.float64),
            "fiber_material_enc": fiber_material_enc.astype(np.float64),
            "cube_strength_mpa": cube_strength_mpa,
            "strength_grade_enc": strength_grade_enc.astype(np.float64),
            "concrete_type_enc": concrete_type_enc.astype(np.float64),
            "binder_content": binder_content,
            "cement_content": cement_content,
            "fly_ash": fly_ash,
            "slag_grade_enc": slag_grade_enc.astype(np.float64),
            "slag_powder": slag_powder,
            "mixing_water": mixing_water,
            "w_b_ratio": w_b_ratio,
            "sand_type_enc": sand_type_enc.astype(np.float64),
            "sand_content": sand_content,
            "sand_ratio": sand_ratio,
            "stone_content": stone_content,
            "admixture_enc": admixture_enc.astype(np.float64),
            "admixture_dosage": admixture_dosage,
            "curing_days": curing_days.astype(np.float64),
            "temperature": temperature,
            "humidity": humidity,
            "casting_method_enc": casting_method_enc.astype(np.float64),
            "fiber_content_x_aspect_ratio": fiber_content_x_aspect_ratio,
        }
    )[FEATURE_COLUMNS]

    scm_total = fly_ash + slag_powder
    width_lin = (
        0.12
        + 0.55 * (w_b_ratio - 0.35)
        - 0.032 * fiber_content
        - 0.00012 * tensile_strength / 1000.0
        - 0.0011 * (cube_strength_mpa - 35.0)
        - 0.00035 * (binder_content - 420.0) / 100.0
        - 0.00025 * (scm_total / 100.0)
        - 0.00012 * slag_grade_enc
        + 0.006 * sand_type_enc
        - 0.00018 * (sand_content - 720.0) / 100.0
        - 0.0002 * (stone_content - 1100.0) / 100.0
        + 0.00006 * (mixing_water - w_b_ratio * binder_content) / 10.0
        - 0.0004 * admixture_dosage
        + 0.001 * (temperature - 20)
        - 0.0008 * (humidity - 65)
        + 0.02 * (1 - casting_method_enc)
        + mat_w
        + ct_w
        + rng.normal(0, 0.04, n)
    )
    crack_width = np.clip(width_lin, 0.02, 1.2)

    density_lin = (
        1.2
        + 2.5 * (w_b_ratio - 0.35)
        - 0.22 * fiber_content
        - 0.007 * (cube_strength_mpa - 35.0) / 10.0
        + 0.001 * (scm_total / 50.0)
        - 0.01 * np.log1p(curing_days)
        + mat_d
        + ct_d
        + rng.normal(0, 0.33, n)
    )
    crack_density = np.clip(density_lin, 0.1, 8.0)

    risk_score = (
        0.78 * (crack_width / 0.5)
        + 0.22 * (crack_density / 4.0)
        - 0.14 * fiber_content
        - 0.018 * (cube_strength_mpa - 35.0) / 25.0
        - 0.06 * (fiber_material_enc == 0)
        - 0.04 * (fiber_material_enc == 1)
        + 0.05 * (fiber_material_enc == 2)
        + rng.normal(0, 0.14, n)
    )
    # 原固定阈值在合成公式下几乎总落在「低」档，多分类无法训练；按分位数拆成低/中/高三档（仅演示数据）
    order = np.argsort(risk_score)
    tert = np.empty(n, dtype=np.int32)
    tert[order] = np.minimum((np.arange(n, dtype=np.int32) * 3) // max(n, 1), 2)
    cracking_risk = tert

    out = X.copy()
    out["crack_width"] = crack_width
    out["crack_density"] = crack_density
    out["cracking_risk"] = cracking_risk.astype(int)
    return out


def ensure_default_models(model_dir: str) -> None:
    """若缺少关键 pkl，或特征维数与当前 FEATURE_COLUMNS 不一致，则训练并保存演示模型。"""
    model_path = Path(model_dir)
    n_expected = len(FEATURE_COLUMNS)
    scaler_n = _scaler_feature_count(model_path)
    need_ok = all((model_path / name).exists() for name in _MODEL_FILES)
    if need_ok and scaler_n == n_expected:
        return

    if need_ok and scaler_n != n_expected:
        _remove_stale_models(model_path)

    if not all((model_path / name).exists() for name in _MODEL_FILES):
        if model_path.exists():
            _remove_stale_models(model_path)
        model_path.mkdir(parents=True, exist_ok=True)

    df = _synthetic_dataset()
    target_cols = ["crack_width", "crack_density", "cracking_risk"]
    X = df.drop(columns=target_cols)
    y_w = df["crack_width"]
    y_d = df["crack_density"]
    y_r = df["cracking_risk"]

    cfg = load_model_config(CONFIG_YAML)
    fit_models_save(
        X,
        y_w,
        y_d,
        y_r,
        model_path,
        cfg,
        verbose=False,
    )
