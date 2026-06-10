"""
国标/规范导向的强度公式基线（信息性，用于「试验估算」与残差学习标签）。

单位与符号
----------
- 强度：MPa（N/mm²）
- 尺寸：mm
- 龄期：本模块以「与侧栏强度等级对应的 fcu,k」为基准，**不单独输入龄期**；若需 28 d 外推，应在数据层增加 `curing_days` 修正后再接入残差模型。

适用条件（摘要）
----------------
1) 立方体抗压（尺寸折算）  
   依据 **GB/T 50081-2019** 对非标准立方体试件抗压强度向 **边长 150 mm 标准立方体** 的常用折算系数：  
   100 mm 试件 ×0.95；200 mm 试件 ×1.05（将非标准试件结果折算为标准试件强度）。  
   本模块在已知 **fcu,k（按 150 mm 标准立方体定义）** 时，反推**在该边长试件上可预期的试验强度量级**：  
   - 150 mm：取 fcu,k；  
   - 100 mm：fcu,k / 0.95；  
   - 200 mm：fcu,k / 1.05；  
   其它边长在 [100,200] 内对 **1/系数** 线性插值（端点同表列值）。

2) 棱柱体轴心抗压  
   标准试验以棱柱体轴压强度为试验值；无试验荷载时，采用 **棱柱体与 150 mm 立方体抗压强度常用比值约 0.76**（150×150×300 量级，工程文献/条文说明常用值，**非 GB/T 50081 直接给出的显式算式**），再乘以本模块保留的 **高宽比/长细比几何修正**（与旧版一致、仅作形状影响，见 `prism_shape_factor`）。

3) 抗折（弯拉/断裂模量级）  
   **GB 50010-2010（2015 年版）表 4.1.3-2**：轴心抗拉强度标准值 ftk，对 **≤C60** 取 ftk = 0.26·fcu^(2/3)（fcu 为立方体抗压强度标准值，MPa）。  
   **>C60** 时按表 4.1.3-2 在 (C60,2.74)…(C80,3.11) 间线性插值。  
   标准抗折试验破坏应力由 **GB/T 50081** 以荷载与几何给出；在无峰值荷载时，本模块将 **弯拉强度（断裂模量级）** 取为 **κ·ftk**（κ=1.43，弯拉高于轴拉的常用经验中值，**条文未给统一换算，属信息性衔接**），再叠加载方式、纤维掺量、梁截面形状等**与旧版一致的小幅工程修正**（在 detail 中注明）。

参数作用域（抗压 / 抗折解耦）
--------------------------
- **仅影响抗压公式** `compressive_formula_pred_mpa`：试件类型（立方体/棱柱体/梁式占位）、立方体边长或棱柱体 b×h×L、**抗压加载方式** `loading_compression`、立方体强度代表值 `cube_strength_mpa`。  
  **不读取**梁跨、**不读取**抗折加载方式。
- **仅影响抗折公式** `flexural_formula_pred_mpa`：试件类型、**梁 b×h×跨度**、**抗折加载方式** `loading_flexural`（三点/四点）、`cube_strength_mpa`、纤维体积掺量。  
  **不读取**抗压加载方式、棱柱体尺寸（非梁式试件时梁参数仍可传入但不参与梁截面修正分支）。

抗折由荷载反算（供以后扩展）
--------------------------
三点弯曲（跨中单点加载，跨中断裂）常用形式：ff = F·L / (b·h²)，其中 F 为破坏荷载（N），L 为下支座跨度（mm），b、h 为截面宽、高（mm）。结果 MPa。见 GB/T 50081-2019 抗折强度试验相关规定。
"""

from __future__ import annotations

import math
from typing import Any


# GB/T 50081-2019 非标准立方体 → 150 mm 标准立方体抗压强度的常用折算系数（标准试件强度 = 实测 × 系数）
_GB_CUBE_EDGE_TO_ALPHA: dict[float, float] = {100.0: 0.95, 150.0: 1.0, 200.0: 1.05}


def _interp_cube_alpha(edge_mm: float) -> float:
    """边长 edge_mm 立方体：标准(150)强度 = 实测强度 × alpha(edge)。alpha 在 100~200 上线性插值。"""
    e = float(edge_mm)
    keys = sorted(_GB_CUBE_EDGE_TO_ALPHA.keys())
    if e <= keys[0]:
        return float(_GB_CUBE_EDGE_TO_ALPHA[keys[0]])
    if e >= keys[-1]:
        return float(_GB_CUBE_EDGE_TO_ALPHA[keys[-1]])
    for i in range(len(keys) - 1):
        a0, a1 = keys[i], keys[i + 1]
        if a0 <= e <= a1:
            f0, f1 = _GB_CUBE_EDGE_TO_ALPHA[a0], _GB_CUBE_EDGE_TO_ALPHA[a1]
            t = (e - a0) / (a1 - a0)
            return float(f0 + t * (f1 - f0))
    return 1.0


def predicted_cube_test_strength_from_fcu_k_mpa(fcu_k_mpa: float, edge_mm: float) -> float:
    """
    已知 150 mm 标准立方体定义的 fcu,k（MPa），估算在边长为 edge_mm 的立方体上
    「可预期的试验强度」水平（同一材料、信息性）：fcu,k / alpha(edge)。
    """
    alpha = _interp_cube_alpha(edge_mm)
    return float(max(0.1, fcu_k_mpa / max(alpha, 1e-9)))


def prism_shape_factor(b_mm: float, h_mm: float, l_mm: float) -> float:
    """棱柱体相对 150×150×300 的简化形状修正（与历史版本一致，信息性）。"""
    b, h, L = max(b_mm, 1.0), max(h_mm, 1.0), max(l_mm, 1.0)
    hb = h / b
    lh = L / h
    k = 0.97
    if abs(hb - 2.0) > 0.35:
        k *= 0.99
    if lh > 3.8:
        k *= 0.98
    return float(max(0.90, min(1.05, k)))


def compressive_formula_pred_mpa(
    specimen_type: str,
    *,
    cube_edge_mm: float,
    prism_b_mm: float,
    prism_h_mm: float,
    prism_l_mm: float,
    loading_compression: str,
    cube_strength_mpa: float,
) -> tuple[float, dict[str, Any]]:
    """
    抗压强度公式基线（MPa）。不删除调用方任何原始字段；本函数仅消费副本。
    """
    fcu_k = max(float(cube_strength_mpa), 1.0)
    detail: dict[str, Any] = {
        "基准_fcu_k_150_MPa": round(fcu_k, 4),
        "依据_抗压": "GB/T 50081-2019 立方体尺寸折算系数；棱柱体轴压采用 0.76·fcu,k 常用比值 + 形状修正",
    }

    fc_formula = fcu_k
    if specimen_type == "立方体（边长可变）":
        edge = float(cube_edge_mm)
        alpha = _interp_cube_alpha(edge)
        fc_formula = predicted_cube_test_strength_from_fcu_k_mpa(fcu_k, edge)
        detail["立方体边长_mm"] = edge
        detail["GB尺寸折算系数_alpha_实测折标准"] = round(alpha, 4)
        detail["说明"] = (
            "fcu,k 按 150 mm 标准立方体定义；"
            f"边长 {edge:g} mm 时取预期试验强度 ≈ fcu,k/α = {fcu_k:.3f}/{alpha:.4f}。"
        )
    elif specimen_type == "棱柱体（轴心抗压）":
        k_geom = prism_shape_factor(prism_b_mm, prism_h_mm, prism_l_mm)
        fc_formula = 0.76 * fcu_k * k_geom
        detail["棱柱体_b_h_L_mm"] = [prism_b_mm, prism_h_mm, prism_l_mm]
        detail["棱柱体系数_0_76"] = 0.76
        detail["棱柱体形状修正"] = round(k_geom, 4)
    else:
        # 梁式试件：抗压仍给与等级相关的材料轴压代表值量级（信息性）
        fc_formula = 0.98 * fcu_k
        detail["说明"] = "梁式试件以抗折为主；抗压公式基线取 0.98·fcu,k（信息性）。"

    # GB/T 50081 推荐分级加载；其它方式仅作微小演示性区分（与旧版一致）
    if loading_compression == "分级加载（GB/T 50081 常用）":
        k_load = 1.0
    elif loading_compression == "匀速力控制":
        k_load = 0.99
    else:
        k_load = 0.995
    fc_formula *= k_load
    detail["抗压加载方式系数"] = k_load

    return float(max(0.1, fc_formula)), detail


def ftk_standard_mpa_from_fcu_k(fcu_k_mpa: float) -> tuple[float, dict[str, Any]]:
    """GB 50010-2010 表 4.1.3-2 轴心抗拉强度标准值 ftk（MPa），fcu,k 为立方体抗压强度标准值。"""
    fcu = max(float(fcu_k_mpa), 1.0)
    d: dict[str, Any] = {"fcu_k_MPa": round(fcu, 4), "依据_ftk": "GB 50010-2010 表 4.1.3-2"}
    if fcu <= 60.0:
        ftk = 0.26 * (fcu ** (2.0 / 3.0))
        d["公式"] = "ftk = 0.26 * fcu^(2/3)  （fcu,k ≤ 60 MPa）"
    else:
        pts = [(60.0, 2.74), (65.0, 2.85), (70.0, 2.96), (75.0, 3.07), (80.0, 3.11)]
        if fcu >= 80.0:
            ftk = 3.11 + 0.02 * (fcu - 80.0) / 10.0
            d["说明"] = "fcu,k>80 时表外线性外推（信息性），建议以试验或规范专门条文为准。"
        else:
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                if x0 <= fcu <= x1:
                    t = (fcu - x0) / (x1 - x0)
                    ftk = y0 + t * (y1 - y0)
                    break
            else:
                ftk = 2.74
        d["公式"] = "表 4.1.3-2 分段线性（C60~C80）"
    return float(max(0.05, ftk)), d


def flexural_from_peak_load_three_point_mpa(
    F_n: float, L_mm: float, b_mm: float, h_mm: float
) -> float:
    """GB/T 50081 三点弯由破坏荷载得到的抗折强度（MPa）：ff = F*L/(b*h^2)。"""
    F, L, b, h = float(F_n), float(L_mm), float(b_mm), float(h_mm)
    if b <= 0 or h <= 0:
        return 0.0
    return float(F * L / (b * h * h))


def flexural_formula_pred_mpa(
    specimen_type: str,
    *,
    beam_b_mm: float,
    beam_h_mm: float,
    beam_span_mm: float,
    loading_flexural: str,
    cube_strength_mpa: float,
    fiber_content_pct: float,
) -> tuple[float, dict[str, Any]]:
    """
    抗折强度公式基线（MPa）：ftk（GB50010）→ 弯拉量级 κ·ftk，再纤维与加载形式微调。
    """
    fcu_k = max(float(cube_strength_mpa), 1.0)
    ftk, d_ftk = ftk_standard_mpa_from_fcu_k(fcu_k)
    kappa = 1.43
    f_rupture = kappa * ftk
    detail: dict[str, Any] = {
        **d_ftk,
        "kappa_弯拉_轴拉": kappa,
        "说明_kappa": "弯拉（断裂模量级）与轴拉标准值比值取条文说明常用区间中值 1.43（信息性）",
    }

    v = max(0.0, min(3.0, float(fiber_content_pct)))
    fiber_boost = 1.0 + 0.04 * (v - 0.5)
    fiber_boost = float(max(1.0, min(1.12, fiber_boost)))
    f_rupture *= fiber_boost
    detail["纤维体积掺量_百分数"] = v
    detail["纤维弯拉增强系数_工程经验"] = round(fiber_boost, 4)

    if loading_flexural == "四点弯曲（等弯矩段）":
        k_load = 0.96
        detail["加载形式系数"] = k_load
    else:
        k_load = 1.0
        detail["加载形式系数"] = k_load
    f_rupture *= k_load

    if specimen_type == "梁式试件（抗折）":
        bh = beam_h_mm / max(beam_b_mm, 1.0)
        k_beam = 1.0 - 0.01 * min(max(bh - 1.0, 0.0), 3.0)
        k_beam = max(0.92, k_beam)
        f_rupture *= k_beam
        detail["梁_b_h_span_mm"] = [beam_b_mm, beam_h_mm, beam_span_mm]
        detail["梁截面高宽比修正_信息性"] = round(k_beam, 4)

    return float(max(0.05, f_rupture)), detail
