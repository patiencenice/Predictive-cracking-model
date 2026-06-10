# -*- coding: utf-8 -*-
"""校验 GB/T 50082 第 8/9 章填表 CSV 的必填列与 method_id。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.standards.gbt50082_rules import list_method_ids

CH08_METHODS = {
    "shrinkage_noncontact",
    "shrinkage_contact",
    "shrinkage_corrugated_pipe",
}
CH09_METHODS = {"early_cracking_plate"}


def validate(path: Path) -> list[str]:
    issues: list[str] = []
    if not path.is_file():
        return [f"文件不存在: {path}"]
    df = pd.read_csv(path)
    if df.empty:
        print("  （仅表头，填表后可再校验）")
        return []

    registered = set(list_method_ids())
    if "standard_method_id" not in df.columns:
        issues.append("缺少列 standard_method_id")
        return issues

    for i, row in df.iterrows():
        mid = str(row.get("standard_method_id", "") or "").strip()
        if not mid:
            issues.append(f"第 {i + 2} 行: standard_method_id 为空")
        elif mid not in registered:
            issues.append(f"第 {i + 2} 行: 未注册的 method_id={mid!r}")
        if not str(row.get("source_group", "") or "").strip():
            issues.append(f"第 {i + 2} 行: 建议填写 source_group（批次溯源）")
        if mid in CH09_METHODS:
            if not str(row.get("crack_width_definition_id", "") or "").strip():
                issues.append(f"第 {i + 2} 行: 早裂试验建议填写 crack_width_definition_id")
        if mid in CH08_METHODS:
            has_meas = any(
                pd.notna(row.get(c))
                for c in ("length_change_mm", "shrinkage_strain_ue")
            )
            if not has_meas:
                issues.append(f"第 {i + 2} 行: 收缩试验缺少 length_change_mm 或 shrinkage_strain_ue")

    return issues


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:]] or [
        ROOT / "data" / "gbt50082" / "template_ch08_shrinkage.csv",
        ROOT / "data" / "gbt50082" / "template_ch09_early_cracking.csv",
    ]
    ok = True
    for p in paths:
        print(f"\n=== {p} ===")
        issues = validate(p)
        if not issues:
            print("通过（或仅表头占位）")
        else:
            for x in issues:
                print(" -", x)
            if any("缺少" in x or "未注册" in x or "为空" for x in issues):
                ok = False
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
