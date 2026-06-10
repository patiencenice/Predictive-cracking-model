"""标签口径：为 crack_width、crack_density、cracking_risk 建立可审计的定义 ID，并生成 label_definition.md。

注意：不同论文「最大裂缝宽度 / 平均宽度 / 表面裂缝宽度」不可混为同一标签；通过 definition_id 区分。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# 裂缝宽度定义 ID（与数据行 crack_width_definition_id 对应）
CRACK_WIDTH_DEFINITIONS: dict[str, dict[str, Any]] = {
    "CW_MAX_SURFACE_MM": {
        "title": "构件表面实测最大裂缝宽度",
        "description": "典型为读数显微镜/裂缝测宽仪在表面测得的最大值，单位 mm。",
        "exclude_mix_with": ["CW_MEAN_MM", "CW_INNER_MM"],
    },
    "CW_MEAN_MM": {
        "title": "多条裂缝宽度平均值",
        "description": "对若干条裂缝宽度取算术平均；与最大值不是同一随机变量。",
        "exclude_mix_with": ["CW_MAX_SURFACE_MM"],
    },
    "CW_INNER_MM": {
        "title": "内部或剖开观测宽度",
        "description": "与表面最大宽度可能系统偏差；禁止与表面最大宽度混训。",
        "exclude_mix_with": ["CW_MAX_SURFACE_MM"],
    },
    "CW_UNSPECIFIED": {
        "title": "文献未说明测量口径",
        "description": "仅用于占位；质量分应降低，建议人工复核后改为上述之一。",
        "exclude_mix_with": [],
    },
}

CRACK_DENSITY_DEFINITIONS: dict[str, dict[str, Any]] = {
    "CD_COUNT_PER_M2": {
        "title": "单位面积裂缝条数",
        "description": "条/m²，测区面积与是否含微裂缝需在论文中一致。",
    },
    "CD_UNSPECIFIED": {
        "title": "口径不明",
        "description": "不建议与明确定义混训。",
    },
}

# 开裂风险 0/1/2：与线上一致
CRACKING_RISK_RULES = """
分级规则（与训练脚本 cracking_risk 一致）：
- 0：低风险
- 1：中风险
- 2：高风险

文献若为「低/中/高」文字，映射为 0/1/2；若为连续指标，需单独建立映射表（本管线不强行转换）。
"""


def assign_default_definition_ids(df: pd.DataFrame) -> pd.DataFrame:
    """缺省 definition_id 时填 UNSPECIFIED，便于质量模块降权。"""
    out = df.copy()
    if "crack_width_definition_id" not in out.columns:
        out["crack_width_definition_id"] = "CW_UNSPECIFIED"
    else:
        out["crack_width_definition_id"] = out["crack_width_definition_id"].fillna(
            "CW_UNSPECIFIED"
        )
    if "crack_density_definition_id" not in out.columns:
        out["crack_density_definition_id"] = "CD_UNSPECIFIED"
    else:
        out["crack_density_definition_id"] = out[
            "crack_density_definition_id"
        ].fillna("CD_UNSPECIFIED")
    return out


def write_label_definition_md(path: Path) -> Path:
    """写出 outputs/label_definition.md（静态规则 + 定义字典摘要）。"""
    lines: list[str] = [
        "# 标签定义（文献管线）",
        "",
        "## crack_width_definition_id",
        "",
    ]
    for k, v in CRACK_WIDTH_DEFINITIONS.items():
        lines.append(f"### `{k}`")
        lines.append(f"- {v['title']}")
        lines.append(f"- {v['description']}")
        lines.append("")
    lines.append("## crack_density_definition_id")
    lines.append("")
    for k, v in CRACK_DENSITY_DEFINITIONS.items():
        lines.append(f"### `{k}`")
        lines.append(f"- {v['title']}")
        lines.append(f"- {v['description']}")
        lines.append("")
    lines.append("## cracking_risk")
    lines.append(CRACKING_RISK_RULES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def validate_single_definition_family(df: pd.DataFrame, col: str = "crack_width_definition_id") -> list[str]:
    """
    若同一训练集中混用互斥的 crack_width 定义，返回警告列表（不自动删行，由用户决策）。
    """
    warns: list[str] = []
    if col not in df.columns or len(df) == 0:
        return warns
    ids = df[col].dropna().astype(str).unique().tolist()
    for a, da in CRACK_WIDTH_DEFINITIONS.items():
        ex = da.get("exclude_mix_with") or []
        for b in ex:
            if a in ids and b in ids:
                warns.append(f"同时存在互斥宽度定义 {a} 与 {b}，请勿混训或按子集分开建模。")
    return warns
