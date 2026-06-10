from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS
from src.lab_experiment import LOADING_COMPRESSION, LOADING_FLEXURAL, SPECIMEN_TYPES
from src.lab_formula_gb import compressive_formula_pred_mpa, flexural_formula_pred_mpa
from src.lab_strength_residual.lab_mix_features import (
    LAB_MIX_EXTRA_FEATURE_NAMES,
    LAB_STRENGTH_FEATURE_COLUMNS,
    LAB_STRENGTH_FEATURE_COLUMNS_NO_WATER_REDUCER,
    ensure_lab_mix_extra_columns_in_dataframe,
    lab_mix_extra_row_vector,
    summarize_water_reducer_features,
)
from src.lab_strength_residual.water_reducer_learnability import (
    build_water_reducer_learnability_report,
)

# 丢行诊断：明细条数上限（控制 JSON 体积）
DROPPED_ROW_DIAGNOSTIC_CAP = 100
_BAD_VALUE_REPR_MAX = 80


def _cell_repr_for_drop(v: Any, max_len: int = _BAD_VALUE_REPR_MAX) -> str:
    """将单元格值转为短字符串，便于 CSV 修复时对照。"""
    try:
        if pd.isna(v) and not isinstance(v, (str, bytes)):
            return "<NaN>"
        s = repr(v)
    except Exception:
        s = "<unrepr>"
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _drop_meta(
    reason: str,
    *,
    bad_column: str | None = None,
    bad_value_repr: str | None = None,
    detail: str | None = None,
    missing_columns: list[str] | None = None,
) -> dict[str, Any]:
    """单行丢行诊断字典（仅含可序列化字段）。"""
    m: dict[str, Any] = {"reason": reason}
    if bad_column is not None:
        m["bad_column"] = bad_column
    if bad_value_repr is not None:
        m["bad_value_repr"] = bad_value_repr
    if detail is not None:
        m["detail"] = detail[:200] if len(detail) > 200 else detail
    if missing_columns:
        m["missing_columns"] = list(missing_columns)
    return m


LAB_DEFAULTS: dict[str, Any] = {
    "lab_specimen": SPECIMEN_TYPES[0],
    "lab_cube_edge_mm": 150.0,
    "lab_prism_b_mm": 150.0,
    "lab_prism_h_mm": 150.0,
    "lab_prism_l_mm": 300.0,
    "lab_beam_b_mm": 100.0,
    "lab_beam_h_mm": 100.0,
    "lab_beam_span_mm": 400.0,
    "lab_loading_compression": LOADING_COMPRESSION[0],
    "lab_loading_flexural": LOADING_FLEXURAL[0],
}


def _row_get(row: pd.Series, key: str, default: Any) -> Any:
    if key not in row.index or pd.isna(row[key]):
        return default
    return row[key]


def _row_needs_manual_review(row: pd.Series) -> bool:
    """为 1/true/yes 时该行不进入公式基线残差训练（数据仍保留在 CSV）。"""
    if "needs_manual_review" not in row.index:
        return False
    v = row["needs_manual_review"]
    if pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return True
    try:
        return float(v) >= 0.5
    except (TypeError, ValueError):
        return False


MANUAL_REVIEW_DEFAULT_NOTE = (
    "字段语义未闭合：fcu/试件协议未与原文对齐前不入公式基线；"
    "补齐 cube_strength_mpa、lab_specimen、立方边长或棱柱尺寸、lab_loading_compression，"
    "并确认 compressive_true 与公式同口径后，将 needs_manual_review 置 0。"
)


def _collect_manual_review_block(df: pd.DataFrame) -> dict[str, Any]:
    """全表扫描 needs_manual_review=1，供报告与 dropped_rows_diagnosis 扩展（不臆造字段）。"""
    samples: list[dict[str, Any]] = []
    groups: set[str] = set()
    for i in range(len(df)):
        row = df.iloc[i]
        if not _row_needs_manual_review(row):
            continue
        sg_s: str | None = None
        if "source_group" in row.index:
            vv = row["source_group"]
            if pd.notna(vv):
                sg_s = str(vv).strip()
                if sg_s:
                    groups.add(sg_s)
        note = MANUAL_REVIEW_DEFAULT_NOTE
        if "manual_review_note" in row.index and pd.notna(row["manual_review_note"]):
            t = str(row["manual_review_note"]).strip()
            if t:
                note = t
        samples.append(
            {
                "iloc": int(i),
                "source_group": sg_s,
                "reason": "needs_manual_review",
                "note": note,
            }
        )
    return {
        "manual_review_rows_count": len(samples),
        "manual_review_source_groups": sorted(groups),
        "manual_review_samples": samples,
    }


def row_compressive_formula_prediction(row: pd.Series) -> float:
    """
    单行抗压公式基线（MPa）。
    仅使用抗压路径参数：试件类型、立方体/棱柱尺寸、抗压加载方式、fcu,k；**不读取**抗折加载方式。
    """
    spec = str(_row_get(row, "lab_specimen", LAB_DEFAULTS["lab_specimen"]))
    cube_edge = float(_row_get(row, "lab_cube_edge_mm", LAB_DEFAULTS["lab_cube_edge_mm"]))
    pb = float(_row_get(row, "lab_prism_b_mm", LAB_DEFAULTS["lab_prism_b_mm"]))
    ph = float(_row_get(row, "lab_prism_h_mm", LAB_DEFAULTS["lab_prism_h_mm"]))
    pl = float(_row_get(row, "lab_prism_l_mm", LAB_DEFAULTS["lab_prism_l_mm"]))
    load_c = str(
        _row_get(row, "lab_loading_compression", LAB_DEFAULTS["lab_loading_compression"])
    )
    fcu = float(row["cube_strength_mpa"])
    fc, _ = compressive_formula_pred_mpa(
        spec,
        cube_edge_mm=cube_edge,
        prism_b_mm=pb,
        prism_h_mm=ph,
        prism_l_mm=pl,
        loading_compression=load_c,
        cube_strength_mpa=fcu,
    )
    return float(fc)


def row_flexural_formula_prediction(row: pd.Series) -> float:
    """
    单行抗折公式基线（MPa）。
    使用抗折路径：试件类型、梁尺寸、抗折加载方式、fcu,k、纤维掺量。
    """
    spec = str(_row_get(row, "lab_specimen", LAB_DEFAULTS["lab_specimen"]))
    bb = float(_row_get(row, "lab_beam_b_mm", LAB_DEFAULTS["lab_beam_b_mm"]))
    bh = float(_row_get(row, "lab_beam_h_mm", LAB_DEFAULTS["lab_beam_h_mm"]))
    bspan = float(_row_get(row, "lab_beam_span_mm", LAB_DEFAULTS["lab_beam_span_mm"]))
    load_f = str(
        _row_get(row, "lab_loading_flexural", LAB_DEFAULTS["lab_loading_flexural"])
    )
    fcu = float(row["cube_strength_mpa"])
    fiber_pct = float(row["fiber_content"])
    ff, _ = flexural_formula_pred_mpa(
        spec,
        beam_b_mm=bb,
        beam_h_mm=bh,
        beam_span_mm=bspan,
        loading_flexural=load_f,
        cube_strength_mpa=fcu,
        fiber_content_pct=fiber_pct,
    )
    return float(ff)


def row_formula_predictions(row: pd.Series) -> tuple[float, float]:
    """单行：抗压 + 抗折公式基线（MPa）；抗压侧与抗折加载方式解耦。"""
    return row_compressive_formula_prediction(row), row_flexural_formula_prediction(row)


def augment_formula_columns(df: pd.DataFrame) -> pd.DataFrame:
    """新增 compressive_formula_pred / flexural_formula_pred；不删原有列。"""
    out = df.copy()
    cfs, ffs = [], []
    for _, row in out.iterrows():
        fc, ff = row_formula_predictions(row)
        cfs.append(fc)
        ffs.append(ff)
    out["compressive_formula_pred"] = cfs
    out["flexural_formula_pred"] = ffs
    return out


def add_residual_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """在已有 compressive_true / flexural_true 时写入残差标签列（供导出训练数据）。"""
    out = augment_formula_columns(df)
    if "compressive_true" in out.columns:
        out["residual_compressive"] = out["compressive_true"] - out[
            "compressive_formula_pred"
        ]
    if "flexural_true" in out.columns:
        out["residual_flexural"] = out["flexural_true"] - out["flexural_formula_pred"]
    return out


def row_source_domain_value(row: pd.Series) -> float:
    """
    来源域：0=本地/合成示例，1=文献等外部表。
    若 CSV 已有 source_domain 列且非空则直接用；否则用 source_group 启发式（^G\\d+$ 视为本地）。
    """
    if "source_domain" in row.index and pd.notna(row["source_domain"]):
        try:
            return float(row["source_domain"])
        except (TypeError, ValueError):
            pass
    sg = str(_row_get(row, "source_group", "")).strip()
    if re.fullmatch(r"G\d+", sg):
        return 0.0
    return 1.0


def row_training_sample_weight(
    row: pd.Series,
    *,
    weight_local: float = 1.0,
    weight_literature: float = 0.7,
    weight_fiber_missing_mult: float = 0.8,
) -> float:
    """训练用样本权重：本地 1.0、文献 0.7；fiber_type 缺失占位时再乘 0.8。"""
    dom = row_source_domain_value(row)
    w = float(weight_local) if dom < 0.5 else float(weight_literature)
    _, ft_miss = _fiber_type_enc_float_and_flag(row)
    if ft_miss >= 0.5:
        w *= float(weight_fiber_missing_mult)
    return float(w)


def _fiber_type_enc_float_and_flag(row: pd.Series) -> tuple[float, float]:
    """
    fiber_type_enc：缺失可通行。
    缺失、空串、NaN、或非有限数值 → 特征位置用 -1.0，fiber_type_missing_flag=1.0；否则为原值与 0.0。
    """
    if "fiber_type_enc" not in row.index:
        return -1.0, 1.0
    val = row["fiber_type_enc"]
    if isinstance(val, str) and val.strip() == "":
        return -1.0, 1.0
    if pd.isna(val):
        return -1.0, 1.0
    try:
        f = float(val)
    except (TypeError, ValueError):
        return -1.0, 1.0
    if not math.isfinite(f):
        return -1.0, 1.0
    return f, 0.0


def _non_fiber_feature_columns_all_finite(row: pd.Series) -> bool:
    """除 fiber_type_enc 外，主 FEATURE_COLUMNS 均须可转为有限浮点（与旧版严格性一致）。"""
    for c in FEATURE_COLUMNS:
        if c == "fiber_type_enc":
            continue
        try:
            v = float(row[c])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(v):
            return False
    return True


def _fiber_type_enc_strict_bad_for_legacy(row: pd.Series) -> bool:
    """旧逻辑下 fiber_type_enc 是否会导致非有限（即旧版会因此丢行）。"""
    _, miss = _fiber_type_enc_float_and_flag(row)
    return miss >= 0.5


def _count_rows_after_label_filter(df: pd.DataFrame) -> int:
    """抗压/抗折标签均可解析为有限浮点的行数。"""
    if "compressive_true" not in df.columns or "flexural_true" not in df.columns:
        return 0
    n = 0
    for i in range(len(df)):
        row = df.iloc[i]
        try:
            yc = float(row["compressive_true"])
            yf = float(row["flexural_true"])
            if math.isfinite(yc) and math.isfinite(yf):
                n += 1
        except (KeyError, TypeError, ValueError):
            continue
    return n


def _count_patch_equivalent_rows(df: pd.DataFrame) -> int:
    """
    等价于「旧版仅因 fiber_type_enc 缺失/非法而丢行、现补丁可保留」的行数：
    非纤维特征全有限、fiber_type 按旧规则为坏值、且新逻辑下单行可入矩阵。
    """
    n = 0
    for i in range(len(df)):
        row = df.iloc[i]
        if not _non_fiber_feature_columns_all_finite(row):
            continue
        if not _fiber_type_enc_strict_bad_for_legacy(row):
            continue
        if _one_row_feature_vector(row, task="compressive", append_source_domain=True) is None:
            continue
        n += 1
    return n


def _one_row_feature_vector_with_reason(
    row: pd.Series,
    *,
    task: str,
    append_source_domain: bool = True,
    include_lab_water_reducer_features: bool = True,
) -> tuple[tuple[np.ndarray, float, float, float] | None, dict[str, Any] | None]:
    """
    与 _one_row_feature_vector 等价；失败时第二元组为列级诊断 dict（reason / bad_column / …）。
    fiber_type_enc 缺失仍入模，不记为丢行原因。
    """
    if task not in ("compressive", "flexural"):
        raise ValueError(task)

    if _row_needs_manual_review(row):
        return None, _drop_meta(
            "needs_manual_review",
            bad_column="needs_manual_review",
            detail="字段语义未闭合（如 fcu,k/试件协议），暂缓进入公式基线训练样本",
        )

    formula_slot = (
        "compressive_formula_pred"
        if task == "compressive"
        else "flexural_formula_pred"
    )

    if "compressive_true" not in row.index:
        return None, _drop_meta(
            "missing_required_columns",
            missing_columns=["compressive_true"],
            detail="column absent",
        )
    try:
        yc = float(row["compressive_true"])
    except (TypeError, ValueError) as e:
        return None, _drop_meta(
            "strict_feature_nonfinite",
            bad_column="compressive_true",
            bad_value_repr=_cell_repr_for_drop(row.get("compressive_true")),
            detail=str(e),
        )

    if "flexural_true" not in row.index:
        return None, _drop_meta(
            "missing_required_columns",
            missing_columns=["flexural_true"],
            detail="column absent",
        )
    try:
        yf = float(row["flexural_true"])
    except (TypeError, ValueError) as e:
        return None, _drop_meta(
            "strict_feature_nonfinite",
            bad_column="flexural_true",
            bad_value_repr=_cell_repr_for_drop(row.get("flexural_true")),
            detail=str(e),
        )

    try:
        if task == "compressive":
            if not math.isfinite(float(yc)):
                return None, _drop_meta(
                    "nonfinite_y_true",
                    bad_column="compressive_true",
                    bad_value_repr=_cell_repr_for_drop(yc),
                    detail="compressive_true not finite",
                )
            formula = float(row_compressive_formula_prediction(row))
            y_true = float(yc)
        else:
            if not math.isfinite(float(yf)):
                return None, _drop_meta(
                    "nonfinite_y_true",
                    bad_column="flexural_true",
                    bad_value_repr=_cell_repr_for_drop(yf),
                    detail="flexural_true not finite",
                )
            formula = float(row_flexural_formula_prediction(row))
            y_true = float(yf)
    except KeyError as e:
        miss = [str(a) for a in e.args if isinstance(a, str)]
        return None, _drop_meta(
            "missing_required_columns",
            missing_columns=miss or ["<KeyError>"],
            detail=str(e)[:200],
        )
    except (TypeError, ValueError) as e:
        return None, _drop_meta(
            "exception_other",
            bad_column=formula_slot,
            detail=f"formula path: {e}"[:200],
        )

    if not math.isfinite(formula):
        return None, _drop_meta(
            "nonfinite_formula_pred",
            bad_column=formula_slot,
            bad_value_repr=_cell_repr_for_drop(formula),
            detail="formula baseline not finite",
        )

    y_resid = y_true - formula
    if not math.isfinite(y_resid):
        return None, _drop_meta(
            "nonfinite_y_residual",
            bad_column="__y_residual__",
            bad_value_repr=_cell_repr_for_drop(y_resid),
            detail="y_true - formula not finite",
        )

    ft_enc, ft_miss = _fiber_type_enc_float_and_flag(row)
    mix_map = dict(zip(LAB_MIX_EXTRA_FEATURE_NAMES, lab_mix_extra_row_vector(row)))
    vec: list[float] = []
    vec_slot_names: list[str] = []
    cols = (
        LAB_STRENGTH_FEATURE_COLUMNS
        if include_lab_water_reducer_features
        else LAB_STRENGTH_FEATURE_COLUMNS_NO_WATER_REDUCER
    )

    for c in cols:
        vec_slot_names.append(c)
        if c == "fiber_type_enc":
            vec.append(ft_enc)
        elif c in mix_map:
            try:
                vec.append(float(mix_map[c]))
            except (TypeError, ValueError) as e:
                return None, _drop_meta(
                    "strict_feature_nonfinite",
                    bad_column=c,
                    bad_value_repr=_cell_repr_for_drop(mix_map.get(c)),
                    detail=str(e),
                )
        else:
            if c not in row.index:
                return None, _drop_meta(
                    "missing_required_columns",
                    missing_columns=[c],
                    detail="column absent in row",
                )
            try:
                raw = row[c]
                vec.append(float(raw))
            except (TypeError, ValueError) as e:
                return None, _drop_meta(
                    "strict_feature_nonfinite",
                    bad_column=c,
                    bad_value_repr=_cell_repr_for_drop(raw),
                    detail=str(e),
                )

    vec_slot_names.append(formula_slot)
    vec.append(float(formula))

    vec_slot_names.append("fiber_factor")
    if "fiber_content" not in row.index or "aspect_ratio" not in row.index:
        miss = [
            x for x in ("fiber_content", "aspect_ratio") if x not in row.index
        ]
        return None, _drop_meta(
            "missing_required_columns",
            missing_columns=miss,
            detail="fiber_factor needs fiber_content and aspect_ratio",
        )
    try:
        fc_raw = row["fiber_content"]
        ar_raw = row["aspect_ratio"]
        prod = float(fc_raw) * float(ar_raw)
    except (TypeError, ValueError) as e:
        return None, _drop_meta(
            "strict_feature_nonfinite",
            bad_column="fiber_factor",
            bad_value_repr=(
                f"fiber_content={_cell_repr_for_drop(row.get('fiber_content'))} "
                f"aspect_ratio={_cell_repr_for_drop(row.get('aspect_ratio'))}"
            ),
            detail=str(e),
        )
    vec.append(prod)

    vec_slot_names.append("fiber_type_missing_flag")
    vec.append(ft_miss)

    if append_source_domain:
        vec_slot_names.append("source_domain")
        try:
            vec.append(float(row_source_domain_value(row)))
        except (TypeError, ValueError) as e:
            return None, _drop_meta(
                "strict_feature_nonfinite",
                bad_column="source_domain",
                detail=str(e),
            )

    x = np.array(vec, dtype=np.float64)
    if not np.isfinite(x).all():
        for idx in range(int(x.shape[0])):
            if not math.isfinite(float(x[idx])):
                bn = (
                    vec_slot_names[idx]
                    if idx < len(vec_slot_names)
                    else f"index_{idx}"
                )
                return None, _drop_meta(
                    "vector_nonfinite",
                    bad_column=bn,
                    bad_value_repr=_cell_repr_for_drop(
                        vec[idx] if idx < len(vec) else x[idx]
                    ),
                    detail="assembled vector has NaN/Inf",
                )

    if not math.isfinite(y_resid):
        return None, _drop_meta(
            "nonfinite_y_residual",
            bad_column="__y_residual__",
            bad_value_repr=_cell_repr_for_drop(y_resid),
            detail="post-vector y_residual check",
        )

    return (x, float(y_resid), float(y_true), float(formula)), None


def _one_row_feature_vector(
    row: pd.Series,
    *,
    task: str,
    append_source_domain: bool = True,
    include_lab_water_reducer_features: bool = True,
) -> tuple[np.ndarray, float, float, float] | None:
    """
    单行特征向量、残差标签、真值、公式基线；异常或非法数值返回 None。
    fiber_type_enc 单独放宽：缺失时 -1.0 + missing_flag；主列仍严格有限；减水剂扩展由 lab_mix_features 生成。
    拼接顺序：LAB_STRENGTH_FEATURE_COLUMNS → formula → fiber_factor → fiber_type_missing_flag
    [→ source_domain，若 append_source_domain]。
    """
    tup, _meta = _one_row_feature_vector_with_reason(
        row,
        task=task,
        append_source_domain=append_source_domain,
        include_lab_water_reducer_features=include_lab_water_reducer_features,
    )
    return tup if tup is not None else None


def build_xy_matrices(
    df: pd.DataFrame,
    *,
    task: str,
    append_source_domain: bool = True,
    weight_local: float = 1.0,
    weight_literature: float = 0.7,
    weight_fiber_missing_mult: float = 0.8,
    include_lab_water_reducer_features: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """
    task: 'compressive' | 'flexural'
    X: LAB_STRENGTH_FEATURE_COLUMNS（或不含减水剂六列）+ 公式基线 + fiber_factor + fiber_type_missing_flag [+ source_domain]
    y_resid: 残差 = true - formula_pred
    返回 (... , sample_weights, dataset_stats)；sample_weights 与行对齐，供加权拟合/指标。
    groups 为 object 数组（来自 source_group），无该列时为 None；
    row_indices 为与 X 行对齐的原始 df 行号（iloc，整数）。
    """
    if task not in ("compressive", "flexural"):
        raise ValueError(task)
    need = {"compressive_true", "flexural_true"} <= set(df.columns)
    if not need:
        raise ValueError("数据需含 compressive_true 与 flexural_true 列")

    # 减水剂扩展六列：若 CSV 缺列则按 lab_mix_features 统一占位补齐（不臆造真实减水率）
    df_work = df
    if any(c not in df.columns for c in LAB_MIX_EXTRA_FEATURE_NAMES):
        df_work = ensure_lab_mix_extra_columns_in_dataframe(df)

    manual_review_block = _collect_manual_review_block(df_work)
    water_reducer_feature_summary = summarize_water_reducer_features(df_work)
    water_reducer_learnability = build_water_reducer_learnability_report(df_work)

    formula_col = (
        "compressive_formula_pred"
        if task == "compressive"
        else "flexural_formula_pred"
    )
    cols = (
        LAB_STRENGTH_FEATURE_COLUMNS
        if include_lab_water_reducer_features
        else LAB_STRENGTH_FEATURE_COLUMNS_NO_WATER_REDUCER
    )
    feat_names = list(cols) + [
        formula_col,
        "fiber_factor",
        "fiber_type_missing_flag",
    ]
    if append_source_domain:
        feat_names.append("source_domain")

    n_rows_csv = int(len(df_work))
    n_rows_after_task_filter = _count_rows_after_label_filter(df_work)
    n_rows_dropped_due_to_missing_before_patch_equivalent = (
        _count_patch_equivalent_rows(df_work)
    )

    xs: list[np.ndarray] = []
    yrs: list[float] = []
    yts: list[float] = []
    fs: list[float] = []
    gs: list[Any] = []
    ridx: list[int] = []
    has_group = "source_group" in df_work.columns
    fiber_type_missing_rows_used = 0
    sw_list: list[float] = []
    dropped_counts: Counter[str] = Counter()
    reason_by_column: Counter[str] = Counter()
    dropped_samples: list[dict[str, Any]] = []

    for i in range(len(df_work)):
        row = df_work.iloc[i]
        one, drop_meta = _one_row_feature_vector_with_reason(
            row,
            task=task,
            append_source_domain=append_source_domain,
            include_lab_water_reducer_features=include_lab_water_reducer_features,
        )
        if one is None:
            meta = drop_meta if isinstance(drop_meta, dict) else {}
            rc = str(meta.get("reason") or "exception_other")
            dropped_counts[rc] += 1
            bc = meta.get("bad_column")
            if isinstance(bc, str) and bc:
                reason_by_column[f"{rc}|{bc}"] += 1
            for mc in meta.get("missing_columns") or []:
                if isinstance(mc, str) and mc:
                    reason_by_column[f"{rc}|missing:{mc}"] += 1
            if (
                rc != "needs_manual_review"
                and len(dropped_samples) < DROPPED_ROW_DIAGNOSTIC_CAP
            ):
                sg_val: str | None
                if "source_group" in row.index:
                    vv = row["source_group"]
                    sg_val = None if pd.isna(vv) else str(vv)
                else:
                    sg_val = None
                sample: dict[str, Any] = {
                    "iloc": int(i),
                    "task": task,
                    "source_group": sg_val,
                }
                for k in (
                    "reason",
                    "bad_column",
                    "bad_value_repr",
                    "detail",
                    "missing_columns",
                ):
                    if k in meta:
                        sample[k] = meta[k]
                dropped_samples.append(sample)
            continue
        x, yr, yt, f = one
        _, ft_miss = _fiber_type_enc_float_and_flag(row)
        if ft_miss >= 0.5:
            fiber_type_missing_rows_used += 1
        xs.append(x)
        sw_list.append(
            row_training_sample_weight(
                row,
                weight_local=weight_local,
                weight_literature=weight_literature,
                weight_fiber_missing_mult=weight_fiber_missing_mult,
            )
        )
        yrs.append(yr)
        yts.append(yt)
        fs.append(f)
        ridx.append(i)
        if has_group:
            v = row["source_group"]
            gs.append("NA" if pd.isna(v) else str(v))

    if not xs:
        raise ValueError(f"{task}: 无有效样本行（检查数值列与标签列）。")

    X = np.vstack(xs)
    y_resid = np.array(yrs, dtype=np.float64)
    y_true = np.array(yts, dtype=np.float64)
    formula = np.array(fs, dtype=np.float64)
    row_indices = np.array(ridx, dtype=np.int64)
    groups_arr = np.array(gs, dtype=object) if has_group and len(gs) == len(xs) else None
    n_dropped_task = int(sum(dropped_counts.values()))
    dropped_diagnosis: dict[str, Any] = {
        "n_skipped_total": n_dropped_task,
        "summary": dict(sorted(dropped_counts.items())),
        "reason_by_column": dict(sorted(reason_by_column.items())),
        "samples": dropped_samples,
        "max_samples_stored": DROPPED_ROW_DIAGNOSTIC_CAP,
        "manual_review": {
            "manual_review_rows_count": manual_review_block[
                "manual_review_rows_count"
            ],
            "manual_review_source_groups": manual_review_block[
                "manual_review_source_groups"
            ],
            "manual_review_samples": manual_review_block["manual_review_samples"],
        },
    }
    dataset_stats: dict[str, Any] = {
        "n_rows_csv": n_rows_csv,
        "n_rows_after_task_filter": n_rows_after_task_filter,
        "n_rows_dropped_due_to_missing_before_patch_equivalent": int(
            n_rows_dropped_due_to_missing_before_patch_equivalent
        ),
        "n_rows_final_used": int(X.shape[0]),
        "fiber_type_missing_rows_used": int(fiber_type_missing_rows_used),
        "dropped_rows_diagnosis": dropped_diagnosis,
        "manual_review_rows_count": manual_review_block["manual_review_rows_count"],
        "manual_review_source_groups": manual_review_block[
            "manual_review_source_groups"
        ],
        "manual_review_samples": manual_review_block["manual_review_samples"],
        "water_reducer_feature_summary": water_reducer_feature_summary,
        "water_reducer_learnability": water_reducer_learnability,
        "include_lab_water_reducer_features": include_lab_water_reducer_features,
    }
    sample_weights = np.array(sw_list, dtype=np.float64)
    return (
        X,
        y_resid,
        y_true,
        formula,
        feat_names,
        groups_arr,
        row_indices,
        sample_weights,
        dataset_stats,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err**2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def regression_metrics_weighted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, float]:
    """加权 MAE / RMSE / R²（sum(w)≈0 时退回非加权）。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64)
    s = float(np.sum(w))
    if s <= 1e-12:
        return regression_metrics(y_true, y_pred)
    err = y_pred - y_true
    mae = float(np.sum(w * np.abs(err)) / s)
    rmse = float(math.sqrt(np.sum(w * (err**2)) / s))
    y_bar = float(np.sum(w * y_true) / s)
    ss_res = float(np.sum(w * ((y_true - y_pred) ** 2)))
    ss_tot = float(np.sum(w * ((y_true - y_bar) ** 2)))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def merge_metrics_unweighted_weighted(
    uw: dict[str, float], wt: dict[str, float]
) -> dict[str, float]:
    """合并非加权与加权指标到同一 dict（便于 JSON 序列化）。"""
    return {
        "mae": uw["mae"],
        "rmse": uw["rmse"],
        "r2": uw["r2"],
        "mae_weighted": wt["mae"],
        "rmse_weighted": wt["rmse"],
        "r2_weighted": wt["r2"],
    }
