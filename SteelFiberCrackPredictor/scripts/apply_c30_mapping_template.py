"""
将 c30_mapping_fill_template.csv 中已确认的 xls 路径写回 c30_group_source_map.json。

用法：
  py scripts/apply_c30_mapping_template.py --dry-run
  py scripts/apply_c30_mapping_template.py

仅处理 confirmed_xls_path 非空行；不触发导入。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = _ROOT / "outputs" / "thermal_stress" / "c30_mapping_fill_template.csv"
MAP_JSON = _ROOT / "data" / "thermal_stress" / "c30_group_source_map.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.template.exists():
        print(f"未找到模板：{args.template}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.template)
    if "sample_id" not in df.columns or "confirmed_xls_path" not in df.columns:
        print("模板需含 sample_id、confirmed_xls_path 列", file=sys.stderr)
        sys.exit(1)

    base = json.loads(MAP_JSON.read_text(encoding="utf-8")) if MAP_JSON.exists() else {}
    updated = 0
    for _, row in df.iterrows():
        sid = str(row["sample_id"]).strip()
        path = str(row.get("confirmed_xls_path", "") or "").strip()
        if not sid or sid.startswith("_") or not path or path.lower() == "nan":
            continue
        full = _ROOT / path
        if not full.exists():
            print(f"[SKIP] {sid}：路径不存在 {path}", file=sys.stderr)
            continue
        base[sid] = path.replace("\\", "/")
        updated += 1

    if args.dry_run:
        print(json.dumps({k: v for k, v in base.items() if not k.startswith("_")}, ensure_ascii=False, indent=2))
        print(f"\n[dry-run] 将更新 {updated} 条映射")
        return

    MAP_JSON.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {MAP_JSON}（{updated} 条 confirmed 映射）")


if __name__ == "__main__":
    main()
