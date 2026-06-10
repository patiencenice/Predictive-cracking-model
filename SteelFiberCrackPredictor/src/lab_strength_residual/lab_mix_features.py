"""
lab_strength 训练矩阵专用的配合比扩展特征（与主开裂模型 FEATURE_COLUMNS 解耦）。

减水剂相关：缺失不编造数值，用 -1.0 占位 + *_missing_flag。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS

# 减水剂类型编码（训练 CSV 可用中文或直接用 *_enc 列）
# 语义（与 water_reducer_type_missing_flag 联立）：
#   0–3：类型已知；flag=0
#   4：明确无减水剂；flag=0（仅当 admixture_enc==0 或 CSV 显式 4 且无外加剂矛盾时采用）
#   -1 + flag=1：类型未知（含「未知」文本、列缺失、或与「用了外加剂」矛盾的旧码 4）
# 「未知」不得再与「无」共用同一整数，避免树模型把「明确无」与「类型未知」混为一类。
WATER_REDUCER_TYPE_MAP: dict[str, int] = {
    "聚羧酸系减水剂": 0,
    "聚羧酸减水剂": 0,
    "萘系减水剂": 1,
    "萘系": 1,
    "脂肪族减水剂": 2,
    "缓凝型减水剂": 3,
    "无": 4,
}

# 网页侧栏与校验用（均为 WATER_REDUCER_TYPE_MAP 的键）
WATER_REDUCER_TYPE_UI_OPTIONS: tuple[str, ...] = (
    "无",
    "聚羧酸系减水剂",
    "萘系减水剂",
    "脂肪族减水剂",
    "缓凝型减水剂",
)

LAB_MIX_EXTRA_FEATURE_NAMES: tuple[str, ...] = (
    "water_reducer_type_enc",
    "water_reducer_type_missing_flag",
    "water_reduction_rate_pct",
    "water_reduction_rate_missing_flag",
    "adjusted_w_b_ratio",
    "adjusted_w_b_ratio_missing_flag",
)

# lab_strength_residual 训练用完整特征列顺序（主 FEATURE_COLUMNS + 减水剂扩展）
LAB_STRENGTH_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    list(FEATURE_COLUMNS) + list(LAB_MIX_EXTRA_FEATURE_NAMES)
)

# 消融对照：不包含减水剂扩展六列（顺序与主 FEATURE_COLUMNS 一致，不改 features.py）
LAB_STRENGTH_FEATURE_COLUMNS_NO_WATER_REDUCER: tuple[str, ...] = tuple(FEATURE_COLUMNS)


def _row_admixture_enc_is_no_admixture(row: pd.Series) -> bool:
    """admixture_enc==0 表示配合比侧栏「无」外加剂，按约定视为未使用减水剂（非未知）。"""
    if "admixture_enc" not in row.index or pd.isna(row["admixture_enc"]):
        return False
    try:
        return abs(float(row["admixture_enc"])) < 1e-9
    except (TypeError, ValueError):
        return False


def _water_reducer_type_enc_and_flag(row: pd.Series) -> tuple[float, float]:
    """
    返回 (water_reducer_type_enc, water_reducer_type_missing_flag)。

    - **已知类型**：enc ∈ {0,1,2,3}，missing_flag=0。
    - **明确无减水剂**：enc=4，missing_flag=0（来自文本「无」、CSV 显式 enc=4、
      或 admixture_enc==0 且类型信息仍缺失时闭合为无）。
    - **类型未知**：missing_flag=1，enc=-1（含文本「未知」、非法码、列缺失、以及
      CSV/历史上误用的 enc=4 在「已用外加剂」时的保守处理）。
    """
    no_adm = _row_admixture_enc_is_no_admixture(row)
    enc: float = -1.0
    type_miss = 1.0

    if "water_reducer_type_enc" in row.index and pd.notna(row["water_reducer_type_enc"]):
        try:
            raw = float(row["water_reducer_type_enc"])
            if not math.isfinite(raw):
                enc, type_miss = -1.0, 1.0
            elif raw in (0.0, 1.0, 2.0, 3.0):
                enc, type_miss = raw, 0.0
            elif abs(raw - 4.0) < 1e-9:
                # 显式 enc=4：明确「无减水剂类型」，与外加剂大类是否选「减水剂」解耦（用户可显式选类型=无）
                enc, type_miss = 4.0, 0.0
            else:
                enc, type_miss = -1.0, 1.0
        except (TypeError, ValueError):
            enc, type_miss = -1.0, 1.0
    elif "water_reducer_type" in row.index and pd.notna(row["water_reducer_type"]):
        s = str(row["water_reducer_type"]).strip()
        if s and s in WATER_REDUCER_TYPE_MAP:
            mapped = float(WATER_REDUCER_TYPE_MAP[s])
            if abs(mapped - 4.0) < 1e-9:
                enc, type_miss = 4.0, 0.0
            else:
                enc, type_miss = mapped, 0.0
        else:
            enc, type_miss = -1.0, 1.0

    if no_adm and type_miss >= 0.5:
        enc, type_miss = 4.0, 0.0

    return enc, type_miss


def lab_mix_extra_row_vector(row: pd.Series) -> list[float]:
    """
    返回与 LAB_MIX_EXTRA_FEATURE_NAMES 对齐的 6 个浮点特征。

    减水率语义（与 merge/prepare/dataset 一致）：
    - **A 未使用减水剂**：`admixture_enc==0` 且（减水率列缺失/NaN，或 CSV 占位 -1）→ rate=0、rate_miss=0，
      adjusted_w_b_ratio = w_b_ratio（折算率 0%）。
    - **B 使用减水剂但减水率未知**：减水率列缺失/NaN 或无法解析为非负数，且非 A → rate=-1、rate_miss=1，adjusted=-1。
    - **C 减水率已知**：列上为非负有限数（含真实 0%）→ rate 原值、rate_miss=0；adjusted 仅在此且 w_b 可用时计算。

    water_reduction_rate_pct 为百分数（如 25 表示 25%）。**真实 0%** 与 **未知 -1** 由 rate_miss 区分。
    """
    enc, type_miss = _water_reducer_type_enc_and_flag(row)

    # 减水剂类型 enc=4（明确「无」）：A 语义，强制 rate=0、adjusted=w_b，不采信减水率列中的噪声
    if type_miss < 0.5 and abs(enc - 4.0) < 1e-9:
        rate = 0.0
        rate_miss = 0.0
        adj = -1.0
        adj_miss = 1.0
        if "w_b_ratio" in row.index and pd.notna(row["w_b_ratio"]):
            try:
                wb = float(row["w_b_ratio"])
                if math.isfinite(wb) and wb > 0:
                    adj = float(wb)
                    adj_miss = 0.0
            except (TypeError, ValueError):
                pass
        return [enc, type_miss, rate, rate_miss, adj, adj_miss]

    # ----- water_reduction_rate_pct -----（A / B / C 见函数文档）
    rate = -1.0
    rate_miss = 1.0
    if "water_reduction_rate_pct" in row.index and pd.notna(row["water_reduction_rate_pct"]):
        try:
            r = float(row["water_reduction_rate_pct"])
            if math.isfinite(r) and r >= 0.0:
                rate = r
                rate_miss = 0.0
            elif math.isfinite(r) and abs(r + 1.0) < 1e-9 and _row_admixture_enc_is_no_admixture(row):
                # CSV -1 占位 + 明确无外加剂 → 按 A 处理，避免「未用减水剂却标 unknown」
                rate = 0.0
                rate_miss = 0.0
        except (TypeError, ValueError):
            pass
    if rate_miss >= 0.5 and _row_admixture_enc_is_no_admixture(row):
        rate = 0.0
        rate_miss = 0.0

    # ----- adjusted_w_b_ratio -----
    adj = -1.0
    adj_miss = 1.0
    if rate_miss < 0.5 and "w_b_ratio" in row.index and pd.notna(row["w_b_ratio"]):
        try:
            wb = float(row["w_b_ratio"])
            if math.isfinite(wb) and wb > 0:
                adj = float(wb * (1.0 - rate / 100.0))
                adj_miss = 0.0
        except (TypeError, ValueError):
            pass

    return [
        enc,
        type_miss,
        rate,
        rate_miss,
        adj,
        adj_miss,
    ]


def ensure_lab_mix_extra_columns_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    为整张表写入 LAB_MIX_EXTRA 六列，与 lab_mix_extra_row_vector 逐行一致。

    中文说明（统一缺失策略）：
    - 类型未知时：type_enc=-1、type_missing_flag=1；明确无减水剂时 enc=4、flag=0；
    - 减水率未知（且非「无外加剂」）：water_reduction_rate_pct=-1、rate_missing_flag=1；
    - 明确未用减水剂（admixture_enc==0）：rate=0、rate_missing_flag=0；
    - 减水率未知时 **不** 用 w_b 冒充 adjusted：adjusted_w_b_ratio=-1、adjusted_missing_flag=1（与 lab_mix_extra_row_vector 一致，不臆造折算后水胶比）。
    """
    out = df.copy()
    n = len(out)
    if n == 0:
        for name in LAB_MIX_EXTRA_FEATURE_NAMES:
            out[name] = pd.Series(dtype="float64")
        return out
    vals = np.zeros((n, len(LAB_MIX_EXTRA_FEATURE_NAMES)), dtype=np.float64)
    for i in range(n):
        vals[i, :] = np.asarray(
            lab_mix_extra_row_vector(out.iloc[i]), dtype=np.float64
        )
    for j, name in enumerate(LAB_MIX_EXTRA_FEATURE_NAMES):
        out[name] = vals[:, j]
    return out


def summarize_water_reducer_features(df: pd.DataFrame) -> dict[str, Any]:
    """
    按与 lab_mix_extra_row_vector 一致语义统计减水剂六列（用于报告，不参与拟合）。
    """
    n = int(len(df))
    n_known = n_unknown = n_zero = 0
    n_adj_ok = n_adj_miss = 0
    n_type_none = n_type_unknown = n_type_known = 0
    by_sg: dict[str, dict[str, int]] = defaultdict(
        lambda: {"known": 0, "unknown": 0, "zero": 0}
    )
    by_sg_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"none": 0, "unknown": 0, "known": 0}
    )
    for i in range(n):
        row = df.iloc[i]
        vec = lab_mix_extra_row_vector(row)
        te, tm, r, rm, _adj, am = vec
        sg = ""
        if "source_group" in row.index and pd.notna(row["source_group"]):
            sg = str(row["source_group"]).strip()
        if tm < 0.5:
            if abs(te - 4.0) < 1e-9:
                n_type_none += 1
                if sg:
                    by_sg_type[sg]["none"] += 1
            elif te in (0.0, 1.0, 2.0, 3.0):
                n_type_known += 1
                if sg:
                    by_sg_type[sg]["known"] += 1
            else:
                n_type_unknown += 1
                if sg:
                    by_sg_type[sg]["unknown"] += 1
        else:
            n_type_unknown += 1
            if sg:
                by_sg_type[sg]["unknown"] += 1
        if rm >= 0.5:
            n_unknown += 1
            if sg:
                by_sg[sg]["unknown"] += 1
        else:
            n_known += 1
            if sg:
                by_sg[sg]["known"] += 1
            if abs(float(r)) < 1e-12:
                n_zero += 1
                if sg:
                    by_sg[sg]["zero"] += 1
        if am >= 0.5:
            n_adj_miss += 1
        else:
            n_adj_ok += 1

    all_u: list[str] = []
    some_k: list[str] = []
    for k, v in sorted(by_sg.items()):
        if v["unknown"] > 0 and v["known"] == 0 and v["zero"] == 0:
            all_u.append(k)
        if v["known"] > 0 or v["zero"] > 0:
            some_k.append(k)

    return {
        "n_rows_csv": n,
        "n_rows_water_reducer_type_none": n_type_none,
        "n_rows_water_reducer_type_unknown": n_type_unknown,
        "n_rows_water_reducer_type_known": n_type_known,
        "water_reducer_type_by_source_group": {k: dict(v) for k, v in by_sg_type.items()},
        "n_rows_water_reducer_known": n_known,
        "n_rows_water_reducer_unknown": n_unknown,
        "n_rows_water_reducer_zero": n_zero,
        "n_rows_adjusted_w_b_ratio_available": n_adj_ok,
        "n_rows_adjusted_w_b_ratio_missing": n_adj_miss,
        "water_reducer_by_source_group": {k: dict(v) for k, v in by_sg.items()},
        "source_groups_all_water_reducer_unknown": all_u,
        "source_groups_with_some_water_reducer_known": some_k,
    }


def user_inputs_to_lab_mix_source_series(user_inputs: dict) -> pd.Series:
    """
    将网页/API 合并后的 user_inputs 转为单行 Series，供 lab_mix_extra_row_vector 消费。
    与训练 CSV 语义对齐：外加剂非「减水剂」时闭合成减水剂类型「无」；减水率未知写 -1（由 missing_flag 表达）。
    """
    from src.features import user_inputs_to_feature_frame

    df = user_inputs_to_feature_frame(user_inputs)
    s = df.iloc[0].copy()
    adm = str(user_inputs.get("admixture", "无")).strip()
    if adm != "减水剂":
        s["water_reducer_type"] = "无"
        return s

    wt = str(user_inputs.get("water_reducer_type", "无")).strip()
    if wt not in WATER_REDUCER_TYPE_MAP:
        wt = "无"
    s["water_reducer_type"] = wt

    if wt == "无":
        return s

    if bool(user_inputs.get("water_reduction_rate_unknown")):
        s["water_reduction_rate_pct"] = -1.0
    else:
        s["water_reduction_rate_pct"] = float(user_inputs.get("water_reduction_rate_pct", 0.0))
    return s


def lab_mix_extra_dict_from_user_inputs(user_inputs: dict) -> dict[str, float]:
    """与 LAB_MIX_EXTRA_FEATURE_NAMES 对齐的六列浮点字典（与训练矩阵 lab_mix_extra_row_vector 一致）。"""
    row = user_inputs_to_lab_mix_source_series(user_inputs)
    vec = lab_mix_extra_row_vector(row)
    return dict(zip(LAB_MIX_EXTRA_FEATURE_NAMES, vec))
