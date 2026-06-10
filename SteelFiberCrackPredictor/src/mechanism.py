"""
机理分析：特征重要性（训练落盘）、部分依赖（PDP）、局部单变量响应（ICE 风格）。
背景样本来自 data/training_data.example.csv；缺失时用当前输入加小扰动合成近似背景。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

from src.features import FEATURE_COLUMNS
from src.paths import PROJECT_ROOT

# 图中展示用中文简称（键为 FEATURE_COLUMNS 英文名）
FEATURE_LABEL_CN: dict[str, str] = {
    "fiber_content": "纤维掺量(%)",
    "aspect_ratio": "长径比",
    "tensile_strength": "纤维抗拉(MPa)",
    "fiber_type_enc": "纤维外形(编码)",
    "fiber_material_enc": "纤维材质(编码)",
    "cube_strength_mpa": "立方体强度(MPa)",
    "strength_grade_enc": "强度等级(编码)",
    "concrete_type_enc": "混凝土类型(编码)",
    "binder_content": "胶材(kg/m³)",
    "cement_content": "水泥(kg/m³)",
    "fly_ash": "粉煤灰(kg/m³)",
    "slag_grade_enc": "矿粉等级(编码)",
    "slag_powder": "矿粉(kg/m³)",
    "mixing_water": "用水量(kg/m³)",
    "w_b_ratio": "水胶比",
    "sand_type_enc": "砂类型(编码)",
    "sand_content": "砂用量(kg/m³)",
    "sand_ratio": "砂率(%)",
    "stone_content": "石用量(kg/m³)",
    "admixture_enc": "外加剂(编码)",
    "admixture_dosage": "外加剂用量(kg/m³)",
    "curing_days": "养护龄期(d)",
    "temperature": "环境温度(℃)",
    "humidity": "环境湿度(%)",
    "casting_method_enc": "浇筑方式(编码)",
    "fiber_content_x_aspect_ratio": "掺量×长径比",
}


def feature_label(name: str) -> str:
    return FEATURE_LABEL_CN.get(name, name)


def load_feature_importance_json(models_dir: Path) -> dict[str, Any] | None:
    p = models_dir / "feature_importance.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_training_metrics_json(models_dir: Path) -> dict[str, Any] | None:
    p = models_dir / "training_metrics.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_cv_metrics_json(models_dir: Path) -> dict[str, Any] | None:
    """由 `py -m src.cross_validate` 生成的 K 折交叉验证报表。"""
    p = models_dir / "cv_metrics.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_background_scaled(
    scaler, max_rows: int = 280
) -> np.ndarray | None:
    """与训练一致的标准化空间背景样本，用于 PDP / SHAP。"""
    csv = PROJECT_ROOT / "data" / "training_data.example.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    miss = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if miss:
        return None
    df = df[FEATURE_COLUMNS].astype(np.float64)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) == 0:
        return None
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    return scaler.transform(df)


def synthetic_background_around(
    x_single_scaled: np.ndarray, n: int = 200, seed: int = 42
) -> np.ndarray:
    """无 CSV 时：围绕当前点加小高斯扰动生成近似背景（仅用于 PDP 数值稳定）。"""
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x_single_scaled, dtype=np.float64).ravel()
    d = len(x0)
    scale = 0.12 * (np.abs(x0) + 0.4)
    noise = rng.normal(0.0, 1.0, (n, d)) * scale
    bg = x0.reshape(1, -1) + noise
    return bg


def top_feature_names(imp_map: dict[str, float], n: int = 12) -> list[str]:
    items = sorted(imp_map.items(), key=lambda kv: -float(kv[1]))
    return [k for k, _ in items[:n]]


def top_feature_importance_rows(
    imp_map: dict[str, float], n: int = 10
) -> list[tuple[str, str, float]]:
    """报告用：按重要性降序返回 (中文标签, 英文名, 归一化重要性)。"""
    if not imp_map:
        return []
    parsed: list[tuple[str, float]] = []
    for k, v in imp_map.items():
        try:
            parsed.append((str(k), abs(float(v))))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return []
    parsed.sort(key=lambda kv: -kv[1])
    take = parsed[:n]
    s = sum(v for _, v in take) or 1.0
    return [(feature_label(k), k, v / s) for k, v in take]


def partial_dependence_1d(
    model,
    X_bg: np.ndarray,
    feature_idx: int,
    *,
    grid_resolution: int = 24,
    is_classifier: bool = False,
    class_index: int = 0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """单特征平均部分依赖（标准化特征空间）。分类器用 predict_proba 的指定类别概率。"""
    gr = min(int(grid_resolution), 48)
    try:
        if is_classifier:
            res = partial_dependence(
                model,
                X_bg,
                features=[feature_idx],
                kind="average",
                grid_resolution=gr,
                response_method="predict_proba",
            )
        else:
            res = partial_dependence(
                model,
                X_bg,
                features=[feature_idx],
                kind="average",
                grid_resolution=gr,
            )
        grid_vals = res["grid_values"][0]
        avg = res["average"]
        if is_classifier:
            ci = int(np.clip(class_index, 0, avg.shape[0] - 1))
            curve = np.asarray(avg[ci], dtype=np.float64).ravel()
        else:
            curve = np.asarray(avg[0], dtype=np.float64).ravel()
        return np.asarray(grid_vals, dtype=np.float64), curve
    except Exception:
        return None, None


def _scan_grid(
    x_single_scaled: np.ndarray,
    X_bg: np.ndarray | None,
    j: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.asarray(x_single_scaled, dtype=np.float64).ravel()
    x = np.tile(x0, (n, 1))
    if X_bg is not None and X_bg.shape[0] >= 5:
        lo = float(np.percentile(X_bg[:, j], 3))
        hi = float(np.percentile(X_bg[:, j], 97))
    else:
        v = float(x0[j])
        span = 0.2 * (abs(v) + 0.5)
        lo, hi = v - span, v + span
    if hi <= lo + 1e-12:
        hi = lo + 1e-6
    grid = np.linspace(lo, hi, n)
    for i, v in enumerate(grid):
        x[i, j] = v
    return grid, x


def local_ice_curve(
    model,
    x_single_scaled: np.ndarray,
    X_bg: np.ndarray | None,
    j: int,
    *,
    n: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """回归模型：沿单特征扫描预测输出。"""
    grid, x = _scan_grid(x_single_scaled, X_bg, j, n)
    y = model.predict(x)
    return grid, np.asarray(y, dtype=np.float64)


def local_ice_curve_proba(
    clf,
    x_single_scaled: np.ndarray,
    X_bg: np.ndarray | None,
    j: int,
    class_index: int,
    *,
    n: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """分类模型：沿单特征扫描「当前预测类」的概率。"""
    grid, x = _scan_grid(x_single_scaled, X_bg, j, n)
    proba = clf.predict_proba(x)
    ci = int(np.clip(class_index, 0, proba.shape[1] - 1))
    y = proba[:, ci]
    return grid, np.asarray(y, dtype=np.float64)


def shap_risk_bar(
    clf,
    X_bg: np.ndarray,
    x_single_scaled: np.ndarray,
    pred_class: int,
    top_k: int = 14,
) -> tuple[list[str], np.ndarray] | None:
    """开裂风险分类：对预测类别给出 SHAP 贡献条形图数据（绝对值取 Top-K）。"""
    try:
        import shap
    except ImportError:
        return None

    x1 = np.asarray(x_single_scaled, dtype=np.float64)
    if x1.ndim == 1:
        x1 = x1.reshape(1, -1)

    try:
        explainer = shap.TreeExplainer(clf, X_bg)
        sv = explainer.shap_values(x1)
    except Exception:
        try:
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(x1)
        except Exception:
            return None

    sv = np.asarray(sv)
    if sv.ndim == 3:
        # (n_samples, n_features, n_classes)
        ci = int(np.clip(pred_class, 0, sv.shape[2] - 1))
        vec = sv[0, :, ci].astype(np.float64)
    elif isinstance(sv, list) or (sv.ndim == 2 and sv.shape[0] == len(FEATURE_COLUMNS)):
        if isinstance(sv, list):
            ci = int(np.clip(pred_class, 0, len(sv) - 1))
            vec = np.asarray(sv[ci], dtype=np.float64).ravel()
        else:
            vec = sv.ravel()
    else:
        vec = sv.reshape(-1)[: len(FEATURE_COLUMNS)]

    if vec.size != len(FEATURE_COLUMNS):
        return None

    order = np.argsort(-np.abs(vec))[:top_k]
    names = [FEATURE_COLUMNS[i] for i in order]
    values = vec[order]
    return names, values
