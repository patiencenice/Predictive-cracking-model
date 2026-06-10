"""
C30 温度应力 raw 数据分析 + 文献库对照 + 映射建议（只读）。

用法：
  py scripts/analyze_c30_thermal_mapping.py

产物：
  outputs/thermal_stress/c30_mapping_analysis.json
  outputs/thermal_stress/c30_mapping_analysis.md
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RAW_BASE = _ROOT / "data" / "thermal_stress" / "raw" / "温度应力"
MAP_JSON = _ROOT / "data" / "thermal_stress" / "c30_group_source_map.json"
LIT_DIR = _ROOT / "data" / "literature"
OUT_JSON = _ROOT / "outputs" / "thermal_stress" / "c30_mapping_analysis.json"
OUT_MD = _ROOT / "outputs" / "thermal_stress" / "c30_mapping_analysis.md"

WB_VALUES = (0.36, 0.44, 0.48)
RESTRAINTS = ("R0", "R50", "R100")


def _sample_id(wb: float, rc: str) -> str:
    return f"C30_wb{str(wb).replace('.', 'p')}_{rc.upper()}"


def _parse_group(name: str) -> tuple[float | None, str | None]:
    wb = None
    for v in WB_VALUES:
        if re.search(rf"(?<!\d){re.escape(str(v))}(?!\d)", name):
            wb = v
            break
    m = re.search(r"R(0|50|100)\b", name, re.I)
    rc = f"R{m.group(1)}" if m else None
    return wb, rc


def _fingerprint_xls(path: Path) -> dict:
    import xlrd

    sh = xlrd.open_workbook(path).sheet_by_name("Sheet4")
    n = sh.nrows
    if n < 2:
        return {"path": str(path.relative_to(_ROOT)), "error": "empty Sheet4"}
    stress = [
        float(sh.cell_value(r, 2))
        for r in range(1, n)
        if isinstance(sh.cell_value(r, 2), (int, float))
    ]
    temp = [
        float(sh.cell_value(r, 1))
        for r in range(1, n)
        if isinstance(sh.cell_value(r, 1), (int, float))
    ]
    times = [
        float(sh.cell_value(r, 0))
        for r in range(1, n)
        if isinstance(sh.cell_value(r, 0), (int, float))
    ]
    return {
        "path": str(path.relative_to(_ROOT)).replace("\\", "/"),
        "nrows": n,
        "duration_h": round(max(times), 3) if times else None,
        "temp_max_c": round(max(temp), 4) if temp else None,
        "temp_min_c": round(min(temp), 4) if temp else None,
        "axial_stress_min_mpa": round(min(stress), 4) if stress else None,
        "axial_stress_max_mpa": round(max(stress), 4) if stress else None,
    }


def _literature_wb_summary() -> dict:
    """文献库中与 w/b 相关的间接信息（非温度应力试验）。"""
    import pandas as pd

    rows: list[dict] = []
    for p in [
        LIT_DIR / "example_raw_extracted.csv",
        _ROOT / "data" / "training_data.csv",
    ]:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "w_b_ratio" not in df.columns:
            continue
        ser = pd.to_numeric(df["w_b_ratio"], errors="coerce").dropna()
        rows.append(
            {
                "source": str(p.relative_to(_ROOT)),
                "n_rows": int(len(df)),
                "w_b_min": round(float(ser.min()), 4),
                "w_b_max": round(float(ser.max()), 4),
                "w_b_median": round(float(ser.median()), 4),
                "note": "裂缝/主训练表 w_b_ratio，非温度应力原始试验",
            }
        )
    return {
        "thermal_stress_in_literature_db": False,
        "literature_purpose": "crack_width 抽取（paper_01~04），无 R0/R50/R100 温度应力矩阵",
        "indirect_wb_references": rows,
        "bridge_note": (
            "C30 温度应力 3×3 矩阵为独立实验 ground truth；"
            "与 data/literature 无直接行级映射，仅 w_b_ratio 量纲可与主训练表对照。"
        ),
    }


def _unique_xls_sources() -> list[dict]:
    seen: dict[str, dict] = {}
    for p in sorted(RAW_BASE.rglob("*.xls")):
        if any(x in p.name for x in ("C50", "C70", "表头")):
            continue
        fp = _fingerprint_xls(p)
        key = (
            fp.get("nrows"),
            fp.get("temp_max_c"),
            fp.get("axial_stress_min_mpa"),
            fp.get("duration_h"),
        )
        if key not in seen:
            seen[key] = fp
        else:
            fp["duplicate_of"] = seen[key]["path"]
        fp["folder_hint"] = p.parent.name
        if "C30温度应力" in str(p.parent):
            fp["folder_hint"] = "C30温度应力根目录"
        seen.setdefault(key, fp)
    return list(seen.values())


def _hypotheses(unique_xls: list[dict]) -> list[dict]:
    """弱假设：目录名/时长/应力指纹 → w/b（需人工确认，不可自动导入）。"""
    by_path = {x["path"]: x for x in unique_xls}
    ideas = []
    # 金隅C30 / 2016-05-23
    for path, wb, conf, reason in (
        (
            "data/thermal_stress/raw/温度应力/C30温度应力/C30/金隅C30/2016-05-23-1153.xls",
            0.48,
            "low",
            "金隅C30 子目录独立试验；时长较短(≈69h)；可能与较高 w/b 档对应（待实验记录确认）",
        ),
        (
            "data/thermal_stress/raw/温度应力/C30温度应力/2016-05-12-1122.xls",
            0.44,
            "low",
            "C30温度应力 根目录与 基准C30 同指纹长试验(≈120h)；可能为中档 w/b（待确认）",
        ),
        (
            "data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/C30基准72.xls",
            0.36,
            "low",
            "文件名含「72」、时长≈73h，为 2016-05-12 同试验截断版；可能为低 w/b（待确认）",
        ),
        (
            "data/thermal_stress/raw/温度应力/C30.xls",
            None,
            "low",
            "根目录独立长试验(≈96h)，T_max 更高、拉应力更大；可能为标定/对照试件，未必在 3×3 矩阵内",
        ),
    ):
        if path in by_path:
            ideas.append(
                {
                    "xls": path,
                    "suggested_w_b_ratio": wb,
                    "confidence": conf,
                    "reason": reason,
                    "fingerprints": by_path[path],
                }
            )
    return ideas


def run() -> dict:
    bmp_files = sorted(RAW_BASE.rglob("*.bmp"))
    bmp_matrix = []
    for p in bmp_files:
        wb, rc = _parse_group(p.name)
        bmp_matrix.append(
            {
                "file": str(p.relative_to(RAW_BASE)).replace("\\", "/"),
                "sample_id": _sample_id(wb, rc) if wb and rc else None,
                "w_b_ratio": wb,
                "restraint_code": rc,
            }
        )

    unique_xls = _unique_xls_sources()
    hypotheses = _hypotheses(unique_xls)

    missing = []
    for wb in WB_VALUES:
        for rc in RESTRAINTS:
            sid = _sample_id(wb, rc)
            stem_xls = RAW_BASE / "C30温度应力" / f"{wb}{rc}.xls"
            mapped = MAP_JSON.exists() and json.loads(MAP_JSON.read_text(encoding="utf-8")).get(sid, "")
            missing.append(
                {
                    "sample_id": sid,
                    "w_b_ratio": wb,
                    "restraint_code": rc,
                    "stem_match_xls_exists": stem_xls.exists(),
                    "explicit_map_configured": bool(mapped),
                    "status": "ready" if stem_xls.exists() or mapped else "pending_manual_mapping",
                }
            )

    report = {
        "analyzed_at_iso": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "raw_folder": str(RAW_BASE.relative_to(_ROOT)).replace("\\", "/"),
        "summary": {
            "bmp_count": len(bmp_files),
            "unique_c30_xls_time_series": len(unique_xls),
            "expected_groups": 9,
            "gap_note": (
                f"9 张 bmp 各不相同，但独立 C30 时序 xls 仅 {len(unique_xls)} 条指纹；"
                "R0/R50/R100 三档约束无法从现有 xls 文件名区分，"
                "文献库亦无温度应力矩阵可对表。"
            ),
        },
        "literature_bridge": _literature_wb_summary(),
        "bmp_matrix": bmp_matrix,
        "unique_xls_catalog": unique_xls,
        "w_b_hypotheses_low_confidence": hypotheses,
        "group_status": missing,
        "recommended_next_steps": [
            "对照实验记录/Origin 工程，确认每个 w/b 对应哪一份日期 xls",
            "确认 R0/R50/R100 是否各有独立 xls（当前目录可能缺失 6 份）",
            "在 c30_group_source_map.json 填写 9 行路径后运行 import_c30_temperature_stress.py",
            "或把 xls 重命名为 0.36R0.xls 等与 bmp 同名（方式 A）",
        ],
        "do_not_auto_import": True,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_render_md(report), encoding="utf-8")
    return report


def _render_md(r: dict) -> str:
    s = r["summary"]
    lines = [
        "# C30 温度应力 raw 数据分析（含文献库对照）",
        "",
        f"**分析时间（UTC）：** {r['analyzed_at_iso']}",
        "",
        "## 1. 结论摘要",
        "",
        f"- **bmp：** {s['bmp_count']} 张（3×3 矩阵，文件名可解析 w/b 与 R 档）",
        f"- **独立 C30 时序 xls 指纹：** {s['unique_c30_xls_time_series']} 条（≠ 9）",
        f"- **文献库：** 无温度应力试验矩阵；`data/literature` 仅服务 crack_width 抽取",
        f"- **{s['gap_note']}**",
        "",
        "## 2. 文献库对照（间接）",
        "",
        r["literature_bridge"]["bridge_note"],
        "",
    ]
    for row in r["literature_bridge"].get("indirect_wb_references", []):
        lines.append(
            f"- `{row['source']}`：w/b ∈ [{row['w_b_min']}, {row['w_b_max']}]，"
            f"中位数 {row['w_b_median']}（{row['note']}）"
        )
    lines.extend(["", "## 3. 独立 xls 指纹目录", ""])
    for x in r["unique_xls_catalog"]:
        lines.append(
            f"- `{x['path']}`：{x.get('nrows')} 行，"
            f"时长 {x.get('duration_h')} h，T_max {x.get('temp_max_c')} ℃，"
            f"σ [{x.get('axial_stress_min_mpa')}, {x.get('axial_stress_max_mpa')}] MPa"
        )
    lines.extend(["", "## 4. w/b 弱假设（不可自动导入）", ""])
    for h in r["w_b_hypotheses_low_confidence"]:
        lines.append(
            f"- w/b≈{h.get('suggested_w_b_ratio')} ← `{h['xls']}` "
            f"（置信度 **{h['confidence']}**）：{h['reason']}"
        )
    lines.extend(["", "## 5. 九组状态", ""])
    for g in r["group_status"]:
        lines.append(
            f"- **{g['sample_id']}**：stem 同名 xls={'有' if g['stem_match_xls_exists'] else '无'}，"
            f"显式映射={'有' if g['explicit_map_configured'] else '无'} → **{g['status']}**"
        )
    lines.extend(["", "## 6. 建议下一步", ""])
    for step in r["recommended_next_steps"]:
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    rep = run()
    print(json.dumps(rep["summary"], ensure_ascii=False, indent=2))
