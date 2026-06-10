"""
试验板块：由试件尺寸、加载方式，结合当前配合比强度等级与纤维掺量，
估算抗压强度与抗折强度（信息性、非替代标准试验）。

抗压公式基线：GB/T 50081-2019 立方体尺寸折算；棱柱体采用常用 0.76·fcu,k 比值等（详见 src.lab_formula_gb）。
抗折公式基线：GB 50010-2010 表 4.1.3-2 ftk 与信息性弯拉系数（详见 src.lab_formula_gb）。

实际报告应以带测力设备的标准试验与 GB/T 50081 等现行条文为准。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.lab_formula_gb import (
    compressive_formula_pred_mpa,
    flexural_formula_pred_mpa,
)


LOADING_COMPRESSION = (
    "匀速位移控制（加载速率按规范）",
    "匀速力控制",
    "分级加载（GB/T 50081 常用）",
)

LOADING_FLEXURAL = (
    "三点弯曲（单点加载）",
    "四点弯曲（等弯矩段）",
)

SPECIMEN_TYPES = (
    "立方体（边长可变）",
    "棱柱体（轴心抗压）",
    "梁式试件（抗折）",
)


@dataclass
class LabEstimateResult:
    """试验估算结果：保留兼容字段 compressive_mpa / flexural_mpa，与公式基线一致。"""

    compressive_mpa: float
    flexural_mpa: float
    compressive_formula_pred: float
    flexural_formula_pred: float
    notes: list[str]
    detail: dict[str, Any]


def estimate_compressive_strength(
    specimen_type: str,
    *,
    cube_edge_mm: float,
    prism_b_mm: float,
    prism_h_mm: float,
    prism_l_mm: float,
    loading_compression: str,
    cube_strength_mpa: float,
) -> tuple[float, dict[str, Any]]:
    """抗压公式基线（MPa）；不消费任何抗折试验参数。"""
    return compressive_formula_pred_mpa(
        specimen_type,
        cube_edge_mm=cube_edge_mm,
        prism_b_mm=prism_b_mm,
        prism_h_mm=prism_h_mm,
        prism_l_mm=prism_l_mm,
        loading_compression=loading_compression,
        cube_strength_mpa=cube_strength_mpa,
    )


def estimate_flexural_strength(
    specimen_type: str,
    *,
    beam_b_mm: float,
    beam_h_mm: float,
    beam_span_mm: float,
    loading_flexural: str,
    cube_strength_mpa: float,
    fiber_content_pct: float,
) -> tuple[float, dict[str, Any]]:
    """抗折公式基线（MPa）；需显式提供抗折加载方式。"""
    return flexural_formula_pred_mpa(
        specimen_type,
        beam_b_mm=beam_b_mm,
        beam_h_mm=beam_h_mm,
        beam_span_mm=beam_span_mm,
        loading_flexural=loading_flexural,
        cube_strength_mpa=cube_strength_mpa,
        fiber_content_pct=fiber_content_pct,
    )


def estimate_strengths(
    specimen_type: str,
    *,
    cube_edge_mm: float,
    prism_b_mm: float,
    prism_h_mm: float,
    prism_l_mm: float,
    beam_b_mm: float,
    beam_h_mm: float,
    beam_span_mm: float,
    loading_compression: str,
    loading_flexural: str | None = None,
    cube_strength_mpa: float,
    fiber_content_pct: float,
    compute_flexural: bool = True,
) -> LabEstimateResult:
    """
    估算抗压强度、抗折强度（MPa）。

    compute_flexural=False：不调用抗折公式，抗折结果为 math.nan；抗压与抗折加载方式完全无关。
    compute_flexural=True 且 loading_flexural 为 None：抗折侧采用默认「三点弯曲」；
    **抗压结果与此参数无关**（内部走 estimate_compressive_strength）。

    cube_strength_mpa: 来自强度等级的立方体抗压强度标准值 fcu,k（与表单 cube_strength_mpa 一致，150 mm 基准）。
    fiber_content_pct: 纤维体积掺量 %。
    """
    notes: list[str] = []
    detail: dict[str, Any] = {}

    fc_f, d_c = estimate_compressive_strength(
        specimen_type,
        cube_edge_mm=cube_edge_mm,
        prism_b_mm=prism_b_mm,
        prism_h_mm=prism_h_mm,
        prism_l_mm=prism_l_mm,
        loading_compression=loading_compression,
        cube_strength_mpa=cube_strength_mpa,
    )

    detail["抗压_公式层"] = d_c
    detail["compressive_formula_pred_MPa"] = round(fc_f, 4)

    ff_f = float("nan")
    if not compute_flexural:
        notes.append("本次为仅抗压估算：未计算抗折公式基线。")
        detail["flexural_formula_pred_MPa"] = None
        detail["抗折_公式层"] = None
    else:
        lf = (
            loading_flexural
            if loading_flexural is not None
            else str(LOADING_FLEXURAL[0])
        )
        if loading_flexural is None:
            notes.append(
                "抗折加载方式未指定：已采用默认「三点弯曲（单点加载）」仅用于抗折公式；抗压未使用该字段。"
            )

        ff_f, d_f = estimate_flexural_strength(
            specimen_type,
            beam_b_mm=beam_b_mm,
            beam_h_mm=beam_h_mm,
            beam_span_mm=beam_span_mm,
            loading_flexural=lf,
            cube_strength_mpa=cube_strength_mpa,
            fiber_content_pct=fiber_content_pct,
        )
        detail["抗折_公式层"] = d_f
        detail["flexural_formula_pred_MPa"] = round(ff_f, 4)

        if lf == "四点弯曲（等弯矩段）":
            notes.append(
                "抗折：四点与三点弯矩分布不同，本模块对公式基线施加 0.96 信息性系数（见 detail）。"
            )

    if loading_compression == "分级加载（GB/T 50081 常用）":
        notes.append("抗压：分级加载与 GB/T 50081 常用流程一致，加载方式系数取 1.0。")
    elif loading_compression == "匀速力控制":
        notes.append("抗压：匀速力控制相对分级加载作约 1% 经验折减（信息性）。")
    else:
        notes.append("抗压：位移控制下速率效应已简化，结果仍为信息性估计。")

    ff_out = float(ff_f) if compute_flexural else float("nan")
    return LabEstimateResult(
        compressive_mpa=float(fc_f),
        flexural_mpa=ff_out,
        compressive_formula_pred=float(fc_f),
        flexural_formula_pred=ff_out,
        notes=notes,
        detail=detail,
    )
