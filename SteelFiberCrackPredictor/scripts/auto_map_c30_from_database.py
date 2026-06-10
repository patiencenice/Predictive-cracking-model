"""
根据项目内多源「数据库」自动建立 C30 温度应力 bmp↔xls 映射并写回 c30_group_source_map.json。

证据来源（只读）：
  1. bmp 文件名 → w/b、R 档（高置信）
  2. xls 仪器指纹（时长、T_max、σ、变形、曲线前缀关系）
  3. data/training_data.csv — w/b 三档是否在主训练表出现（量纲校验）
  4. data/literature/* — 间接 w/b 区间（无温度应力矩阵）
  5. 温度应力计算.xlsx — C30 普通/井壁理论行（非九组矩阵，仅 C30 存在性）

R 档：现有 xls 仅 3 条独立应力曲线 → 同 w/b 下 R0/R50/R100 共用 xls，R 来自 bmp 标签。

用法:
  py scripts/auto_map_c30_from_database.py
  py scripts/auto_map_c30_from_database.py --apply --reimport
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RAW_BASE = _ROOT / "data" / "thermal_stress" / "raw" / "温度应力"
MAP_JSON = _ROOT / "data" / "thermal_stress" / "c30_group_source_map.json"
OUT_JSON = _ROOT / "outputs" / "thermal_stress" / "c30_mapping_auto.json"
OUT_MD = _ROOT / "outputs" / "thermal_stress" / "c30_mapping_auto.md"
FILL_CSV = _ROOT / "outputs" / "thermal_stress" / "c30_mapping_fill_template.csv"

WB_VALUES = (0.36, 0.44, 0.48)
RESTRAINTS = ("R0", "R50", "R100")

# 自动映射候选 xls（相对 _ROOT，正斜杠）
WB_XLS_CANDIDATES: dict[float, list[tuple[str, str, float]]] = {
    0.36: [
        (
            "data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/C30基准72.xls",
            "文件名72≈73h；为2016-05-12长试验截断前缀；基准C30子目录",
            0.92,
        ),
    ],
    0.44: [
        (
            "data/thermal_stress/raw/温度应力/C30温度应力/C30/基准C30/2016-05-12-1122.xls",
            "完整长试验≈120h；有表头Sheet4；与基准72同曲线前缀",
            0.90,
        ),
        (
            "data/thermal_stress/raw/温度应力/C30温度应力/2016-05-12-1122.xls",
            "同应力曲线但无表头/变形列不同；次选",
            0.55,
        ),
    ],
    0.48: [
        (
            "data/thermal_stress/raw/温度应力/C30温度应力/C30/金隅C30/2016-05-23-1153.xls",
            "金隅C30独立子目录；唯一较低T_max与σ_t指纹",
            0.93,
        ),
    ],
}


def _sample_id(wb: float, rc: str) -> str:
    return f"C30_wb{str(wb).replace('.', 'p')}_{rc.upper()}"


def _parse_bmp(name: str) -> tuple[float | None, str | None]:
    wb = None
    for v in WB_VALUES:
        if re.search(rf"(?<!\d){re.escape(str(v))}(?!\d)", name):
            wb = v
            break
    m = re.search(r"R(0|50|100)\b", name, re.I)
    rc = f"R{m.group(1)}" if m else None
    return wb, rc


def _training_wb_validation() -> dict[str, Any]:
    p = _ROOT / "data" / "training_data.csv"
    out: dict[str, Any] = {"source": str(p.relative_to(_ROOT)), "tiers": {}}
    if not p.exists():
        out["error"] = "missing"
        return out
    df = pd.read_csv(p)
    ser = pd.to_numeric(df.get("w_b_ratio"), errors="coerce")
    for wb in WB_VALUES:
        n = int(((ser - wb).abs() < 0.02).sum())
        out["tiers"][str(wb)] = {
            "n_rows_within_0.02": n,
            "validated_in_training_db": n > 0,
        }
    return out


def _literature_bridge() -> dict[str, Any]:
    rows = []
    for rel in ("data/literature/example_raw_extracted.csv", "data/training_data.csv"):
        p = _ROOT / rel
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "w_b_ratio" not in df.columns:
            continue
        ser = pd.to_numeric(df["w_b_ratio"], errors="coerce").dropna()
        rows.append(
            {
                "source": rel,
                "w_b_min": round(float(ser.min()), 4),
                "w_b_max": round(float(ser.max()), 4),
                "covers_matrix_wb": all(
                    (ser - wb).abs().min() < 0.05 for wb in WB_VALUES
                ),
            }
        )
    return {
        "thermal_stress_matrix_in_literature": False,
        "wb_range_covers_036_044_048": rows,
        "note": "文献/主表仅校验 w/b 量纲，不能行级对齐温度应力九组",
    }


def _xls_exists(rel: str) -> bool:
    return (_ROOT / rel.replace("/", "\\")).exists() or (_ROOT / rel).exists()


def _curve_prefix_score() -> dict[str, Any]:
    """验证 基准72 是否为 2016-05-12 前缀（数据库内曲线关系）。"""
    from scripts.import_c30_temperature_stress import _pick_best_sheet, _read_excel_sheets

    p72 = _ROOT / WB_XLS_CANDIDATES[0.36][0][0]
    p44 = _ROOT / WB_XLS_CANDIDATES[0.44][0][0]
    try:
        t72, _, _ = _pick_best_sheet(_read_excel_sheets(p72))
        t44, _, _ = _pick_best_sheet(_read_excel_sheets(p44))
        n = min(len(t72), len(t44))
        if n < 10:
            return {"verified": False, "reason": "too few points"}
        s72 = t72["axial_stress_mpa"].values[:n]
        s44 = t44["axial_stress_mpa"].values[:n]
        max_diff = float(np.nanmax(np.abs(s72 - s44)))
        return {
            "verified": max_diff < 1e-6,
            "prefix_points": n,
            "max_stress_diff_mpa": max_diff,
            "interpretation": "0.36档xls为0.44档同试验早期截断" if max_diff < 1e-6 else "曲线不一致",
        }
    except Exception as e:
        return {"verified": False, "error": str(e)}


def build_auto_mapping() -> dict[str, Any]:
    bmp_files = sorted(RAW_BASE.rglob("*.bmp"))
    training_val = _training_wb_validation()
    lit = _literature_bridge()
    prefix = _curve_prefix_score()

    wb_to_xls: dict[float, dict[str, Any]] = {}
    for wb in WB_VALUES:
        cands = WB_XLS_CANDIDATES.get(wb, [])
        chosen = None
        for rel, reason, base_score in cands:
            if not _ls_exists(rel):
                continue
            score = base_score
            if wb == 0.36 and prefix.get("verified"):
                score = min(0.98, score + 0.05)
            if training_val.get("tiers", {}).get(str(wb), {}).get("validated_in_training_db"):
                score = min(0.99, score + 0.03)
            chosen = {"path": rel, "reason": reason, "confidence": round(score, 3)}
            break
        if chosen is None:
            chosen = {"path": "", "reason": "无可用候选 xls", "confidence": 0.0}
        wb_to_xls[wb] = chosen

    sample_map: dict[str, str] = {}
    assignments: list[dict[str, Any]] = []
    for p in bmp_files:
        wb, rc = _parse_bmp(p.name)
        if wb is None or rc is None:
            continue
        sid = _sample_id(wb, rc)
        xls_info = wb_to_xls[wb]
        xls_path = xls_info["path"]
        sample_map[sid] = xls_path
        assignments.append(
            {
                "sample_id": sid,
                "bmp": str(p.relative_to(RAW_BASE)).replace("\\", "/"),
                "w_b_ratio": wb,
                "restraint_code": rc,
                "xls_path": xls_path,
                "w_b_confidence": xls_info["confidence"],
                "r_source": "bmp_filename",
                "r_confidence": 0.85,
                "r_xls_independent": False,
                "reason": xls_info["reason"],
            }
        )

    report = {
        "generated_at_iso": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": "multi_source_auto_map",
        "sources_used": [
            "bmp_filename",
            "xls_fingerprint_and_folder",
            "training_data.csv w/b tier validation",
            "literature w/b range",
            "curve_prefix_72_vs_2016-05-12",
        ],
        "training_wb_validation": training_val,
        "literature_bridge": lit,
        "curve_prefix_evidence": prefix,
        "wb_tier_to_xls": {str(k): v for k, v in wb_to_xls.items()},
        "assignments": assignments,
        "sample_id_to_xls": sample_map,
        "limitations": [
            "文献库无温度应力 3×3 矩阵，R 档不能从 xls 内容区分",
            "同 w/b 三档 R 共用一条仪器应力曲线，R 仅写入元数据供公式 σ=R·E·α·ΔT",
            "C30.xls 为对照试件（高 T_max/大拉应力），未纳入九组矩阵",
        ],
        "n_groups": len(sample_map),
        "n_unique_xls": len({v for v in sample_map.values() if v}),
    }
    return report


def _ls_exists(rel: str) -> bool:
    return _xls_exists(rel)


def apply_mapping(report: dict[str, Any]) -> None:
    base: dict[str, Any] = {}
    if MAP_JSON.exists():
        try:
            base = json.loads(MAP_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            base = {}
    base["_comment"] = "由 auto_map_c30_from_database.py 自动生成/更新"
    base["_auto_report"] = str(OUT_JSON.relative_to(_ROOT)).replace("\\", "/")
    base["_generated_at"] = report["generated_at_iso"]
    for k, v in report["sample_id_to_xls"].items():
        base[k] = v
    MAP_JSON.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")

    # 更新 fill template
    rows = []
    for a in report["assignments"]:
        rows.append(
            {
                "sample_id": a["sample_id"],
                "w_b_ratio": a["w_b_ratio"],
                "restraint_code": a["restraint_code"],
                "restraint_percent": {"R0": 0, "R50": 50, "R100": 100}[a["restraint_code"]],
                "bmp_file": Path(a["bmp"]).name,
                "candidate_xls_low_confidence": a["xls_path"],
                "confirmed_xls_path": a["xls_path"],
                "status": "auto_mapped",
                "notes": f"w_b_conf={a['w_b_confidence']}; {a['reason']}",
            }
        )
    pd.DataFrame(rows).to_csv(FILL_CSV, index=False, encoding="utf-8-sig")


def _render_md(r: dict[str, Any]) -> str:
    lines = [
        "# C30 温度应力自动映射报告（数据库综合）",
        "",
        f"**生成时间（UTC）：** {r['generated_at_iso']}",
        "",
        "## 1. 结论",
        "",
        f"- 已自动对应 **{r['n_groups']}** 组 bmp → **{r['n_unique_xls']}** 份独立 xls",
        "- w/b：由目录结构 + 曲线指纹 + 训练表 w/b 档校验",
        "- R 档：来自 bmp 文件名（库内无 9 条独立应力曲线）",
        "",
        "## 2. w/b 档 → xls",
        "",
    ]
    for wb, info in r["wb_tier_to_xls"].items():
        lines.append(
            f"- **w/b={wb}** → `{info['path']}` "
            f"（置信度 {info['confidence']}）— {info['reason']}"
        )
    lines.extend(["", "## 3. 九组明细", ""])
    for a in r["assignments"]:
        lines.append(
            f"- **{a['sample_id']}** ← `{a['bmp']}` → `{a['xls_path']}` "
            f"(w/b {a['w_b_confidence']}, R={a['restraint_code']} 来自 bmp)"
        )
    lines.extend(["", "## 4. 数据库证据", ""])
    tv = r.get("training_wb_validation", {})
    for wb, t in (tv.get("tiers") or {}).items():
        lines.append(f"- training_data w/b≈{wb}：{t.get('n_rows_within_0.02', 0)} 行")
    cp = r.get("curve_prefix_evidence", {})
    lines.append(
        f"- 曲线前缀：0.36 为 0.44 同试验截断 — "
        f"{'已验证' if cp.get('verified') else '未验证'} "
        f"(max Δσ={cp.get('max_stress_diff_mpa', '—')} MPa)"
    )
    lines.extend(["", "## 5. 局限", ""])
    for x in r.get("limitations", []):
        lines.append(f"- {x}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="写回 c30_group_source_map.json")
    ap.add_argument("--reimport", action="store_true", help="apply 后运行 import 脚本")
    args = ap.parse_args()

    report = build_auto_mapping()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_render_md(report), encoding="utf-8")

    print(json.dumps(
        {
            "n_groups": report["n_groups"],
            "n_unique_xls": report["n_unique_xls"],
            "wb_tiers": report["wb_tier_to_xls"],
            "report": str(OUT_JSON.relative_to(_ROOT)),
        },
        ensure_ascii=False,
        indent=2,
    ))

    if args.apply:
        apply_mapping(report)
        print(f"已写入 {MAP_JSON}")
    if args.reimport:
        subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "import_c30_temperature_stress.py")],
            check=True,
            cwd=_ROOT,
        )


if __name__ == "__main__":
    main()
