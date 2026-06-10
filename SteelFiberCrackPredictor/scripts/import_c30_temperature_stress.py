"""
C30 温度应力试验数据导入（独立验证集，不并入主训练表）。

读取 data/thermal_stress/raw/ 下 Excel 原始数据，生成时间序列与摘要 CSV 及诊断报告。
优先 Excel，不从 bmp 抠数。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RAW_DIR = _ROOT / "data" / "thermal_stress" / "raw"
MAP_JSON = _ROOT / "data" / "thermal_stress" / "c30_group_source_map.json"
OUT_TS = _ROOT / "data" / "thermal_stress" / "c30_temperature_stress_timeseries.csv"
OUT_SUM = _ROOT / "data" / "thermal_stress" / "c30_temperature_stress_summary.csv"
OUT_JSON = _ROOT / "data" / "thermal_stress" / "c30_field_mapping.json"
REPORT_JSON = _ROOT / "outputs" / "thermal_stress" / "c30_temperature_stress_import_report.json"
REPORT_MD = _ROOT / "outputs" / "thermal_stress" / "c30_temperature_stress_import_report.md"

STRENGTH_GRADE = "C30"
WB_VALUES = (0.36, 0.44, 0.48)
RESTRAINT_MAP = {"R0": 0, "R50": 50, "R100": 100}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "time_h": (
        "时间",
        "time",
        "time_h",
        "time(h",
        "t/h",
        "hour",
        "hours",
        "t(h",
        "t (h",
        "elapsed",
    ),
    "specimen_temperature_c": (
        "试件温度",
        "temperature",
        "temp",
        "温度",
        "specimen",
        "t/℃",
        "t(c",
        "t (c",
        "℃",
    ),
    "axial_stress_mpa": (
        "轴向应力",
        "stress",
        "应力",
        "axial",
        "mpa",
        "σ",
        "sigma",
        "force",
    ),
    "deformation_um": (
        "累积变位",
        "deformation",
        "变形",
        "displacement",
        "disp",
        "um",
        "μm",
        "cumulative",
        "累积",
        "累计",
        "strain",
    ),
}


def _parse_group_from_name(name: str) -> tuple[float | None, str | None, int | None]:
    stem = Path(name).stem
    wb: float | None = None
    for v in WB_VALUES:
        if re.search(rf"(?<!\d){re.escape(str(v))}(?!\d)", stem):
            wb = v
            break
    r_code: str | None = None
    r_pct: int | None = None
    m = re.search(r"R\s*(0|50|100)\b", stem, flags=re.IGNORECASE)
    if m:
        r_code = f"R{m.group(1)}"
        r_pct = RESTRAINT_MAP.get(r_code.upper())
    return wb, r_code, r_pct


def _sample_id(wb: float, r_code: str) -> str:
    wb_s = str(wb).replace(".", "p")
    return f"C30_wb{wb_s}_{r_code.upper()}"


def _normalize_col(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _map_columns(cols: list[str]) -> tuple[dict[str, str], list[str]]:
    mapped: dict[str, str] = {}
    used: set[str] = set()
    norm_cols = {c: _normalize_col(c) for c in cols}

    for std, aliases in COLUMN_ALIASES.items():
        best_orig: str | None = None
        best_rank = 10_000
        for rank, alias in enumerate(aliases):
            for orig, norm in norm_cols.items():
                if orig in used or alias not in norm:
                    continue
                if std == "specimen_temperature_c" and "time" in norm and "温度" not in norm and "temp" not in norm:
                    continue
                if std == "time_h" and "temp" in norm and "time" not in norm and "时间" not in norm:
                    continue
                if std == "axial_stress_mpa" and "环向" in norm and "轴向" not in norm:
                    continue
                if rank < best_rank:
                    best_rank = rank
                    best_orig = orig
        if best_orig:
            mapped[std] = best_orig
            used.add(best_orig)

    return mapped, [c for c in cols if c not in used]


def _read_excel_sheets(path: Path) -> dict[str, pd.DataFrame]:
    last_err: Exception | None = None
    raw = None
    for eng in ("xlrd", "openpyxl", None):
        try:
            kw: dict[str, Any] = {"sheet_name": None, "header": None}
            if eng:
                kw["engine"] = eng
            raw = pd.read_excel(path, **kw)
            break
        except Exception as e:
            last_err = e
    if raw is None:
        raise RuntimeError(f"无法读取 {path}: {last_err}")

    out: dict[str, pd.DataFrame] = {}
    if not isinstance(raw, dict):
        raw = {"Sheet1": raw}

    for sheet, df in raw.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            out[str(sheet)] = df
            continue
        header_row = _detect_header_row(df)
        if header_row is not None:
            hdr = df.iloc[header_row].astype(str).tolist()
            body = df.iloc[header_row + 1 :].copy()
            body.columns = hdr
            out[str(sheet)] = body.reset_index(drop=True)
        else:
            df.columns = [f"col_{i}" for i in range(len(df.columns))]
            out[str(sheet)] = df.reset_index(drop=True)
    return out


def _detect_header_row(df: pd.DataFrame, max_scan: int = 8) -> int | None:
    best_row: int | None = None
    best_score = 0
    for i in range(min(max_scan, len(df))):
        row = df.iloc[i].astype(str).str.lower()
        score = sum(
            1
            for cell in row
            for aliases in COLUMN_ALIASES.values()
            if any(a in cell for a in aliases)
        )
        if score > best_score:
            best_score = score
            best_row = i
    return best_row if best_score >= 2 else None


def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")


def _extract_timeseries_from_sheet(df: pd.DataFrame, *, sheet: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {
        "sheet": sheet,
        "n_rows_raw": int(len(df)),
        "columns_raw": [str(c) for c in df.columns],
        "column_map": {},
        "unrecognized_columns": [],
        "notes": [],
    }
    if df.empty:
        meta["notes"].append("空 sheet")
        return pd.DataFrame(), meta

    col_map, unrecognized = _map_columns([str(c) for c in df.columns])
    meta["column_map"] = col_map
    meta["unrecognized_columns"] = unrecognized
    if "time_h" not in col_map and len(df.columns) >= 3:
        # 部分 C30 仪器导出 Sheet4 无表头：列0=时间，列1=温度，列2=轴向应力
        cols = list(df.columns)
        col_map = {
            "time_h": cols[0],
            "specimen_temperature_c": cols[1],
            "axial_stress_mpa": cols[2],
        }
        if len(cols) >= 4:
            col_map["deformation_um"] = cols[3]
        meta["column_map"] = col_map
        meta["notes"].append("无表头 Sheet4：按列序 positional 映射")
    if "time_h" not in col_map:
        meta["notes"].append("未识别 time_h 列")
        return pd.DataFrame(), meta

    out = pd.DataFrame({"time_h": _to_numeric_series(df[col_map["time_h"]])})
    for std in ("specimen_temperature_c", "axial_stress_mpa", "deformation_um"):
        if std in col_map:
            out[std] = _to_numeric_series(df[col_map[std]])

    out = out.dropna(subset=["time_h"])
    out = out[np.isfinite(out["time_h"].to_numpy())]
    meta["n_rows_extracted"] = int(len(out))
    if len(out) == 0:
        meta["notes"].append("映射后无有效时间点")
    return out, meta


def _pick_best_sheet(sheets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any], str]:
    best_df = pd.DataFrame()
    best_meta: dict[str, Any] = {}
    best_name = ""
    best_n = -1
    sheet_metas: list[dict[str, Any]] = []
    for name, df in sheets.items():
        ts, meta = _extract_timeseries_from_sheet(df, sheet=name)
        sheet_metas.append(meta)
        n = meta.get("n_rows_extracted", 0) or 0
        if n > best_n:
            best_n = n
            best_df = ts
            best_meta = meta
            best_name = name
    best_meta["all_sheets"] = sheet_metas
    best_meta["selected_sheet"] = best_name
    return best_df, best_meta, best_name


def _summarize_group(ts: pd.DataFrame) -> dict[str, Any]:
    s: dict[str, Any] = {}
    if ts.empty:
        s["curve_quality_note"] = "无时间序列数据"
        return s

    t = ts["time_h"].to_numpy(dtype=float)
    note_parts: list[str] = []

    if "specimen_temperature_c" in ts.columns:
        temp = ts["specimen_temperature_c"].to_numpy(dtype=float)
        mask = np.isfinite(temp)
        if mask.any():
            idx = int(np.nanargmax(temp))
            s["max_temperature_c"] = float(temp[idx])
            s["time_at_max_temperature_h"] = float(t[idx])
            if len(t) >= 2:
                dt = np.diff(t)
                dT = np.diff(temp)
                valid = np.isfinite(dt) & (dt > 0)
                if valid.any():
                    rate = np.abs(dT[valid] / dt[valid])
                    rise = dT[valid] >= 0
                    cool = dT[valid] < 0
                    if rise.any():
                        s["temperature_rise_rate_est"] = float(np.nanmax(rate[rise]))
                    if cool.any():
                        s["cooling_rate_est"] = float(np.nanmax(rate[cool]))
        else:
            note_parts.append("温度列无有效数值")

    if "axial_stress_mpa" in ts.columns:
        stress = ts["axial_stress_mpa"].to_numpy(dtype=float)
        mask = np.isfinite(stress)
        if mask.any():
            s["max_compressive_stress_mpa"] = float(np.nanmax(stress))
            s["time_at_max_compressive_stress_h"] = float(t[int(np.nanargmax(stress))])
            s["max_tensile_stress_mpa"] = float(np.nanmin(stress))
            s["time_at_max_tensile_stress_h"] = float(t[int(np.nanargmin(stress))])
            s["final_stress_mpa"] = float(stress[np.where(mask)[0][-1]])
            note_parts.append(
                "应力符号约定：max_compressive=列最大值，max_tensile=列最小值；若设备符号相反请人工复核"
            )
        else:
            note_parts.append("应力列无有效数值")

    if "deformation_um" in ts.columns:
        defo = ts["deformation_um"].to_numpy(dtype=float)
        mask = np.isfinite(defo)
        if mask.any():
            s["max_deformation_um"] = float(np.nanmax(defo))
            s["min_deformation_um"] = float(np.nanmin(defo))
            s["final_deformation_um"] = float(defo[np.where(mask)[0][-1]])

    if len(t) >= 2 and np.all(np.diff(t[np.isfinite(t)]) >= 0):
        note_parts.append("时间序列单调递增")
    else:
        note_parts.append("时间序列非严格单调，建议人工复核")

    s["curve_quality_note"] = "；".join(note_parts) if note_parts else "ok"
    return s


def _is_c30_candidate(path: Path) -> bool:
    name = path.name.upper()
    if re.search(r"\bC50\b|\bC70\b", name):
        return False
    if "表头" in path.name:
        return False
    return True


def _find_raw_files(raw_dir: Path) -> tuple[list[Path], list[Path], list[Path]]:
    if not raw_dir.exists():
        return [], [], []
    excel: list[Path] = []
    images: list[Path] = []
    other: list[Path] = []
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in (".xls", ".xlsx") and _is_c30_candidate(p):
            excel.append(p)
        elif ext == ".bmp":
            images.append(p)
        elif p.name.lower() != "readme.md":
            other.append(p)
    return excel, images, other


def _load_group_map() -> dict[str, str]:
    if not MAP_JSON.exists():
        return {}
    try:
        data = json.loads(MAP_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in data.items():
        if str(k).startswith("_"):
            continue
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out[str(k)] = s
    return out


def _resolve_map_path(rel_or_abs: str) -> Path | None:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = _ROOT / p
    return p.resolve() if p.exists() else None


def _find_bmp_for_group(wb: float, r_code: str, bmp_files: list[Path]) -> Path | None:
    for bmp in bmp_files:
        bwb, br, _ = _parse_group_from_name(bmp.name)
        if bwb == wb and br and br.upper() == r_code.upper():
            return bmp
    return None


def _resolve_xls_for_group(
    sample_id: str,
    bmp: Path | None,
    group_map: dict[str, str],
) -> tuple[Path | None, str | None]:
    """
    方式 A：bmp 同目录下同名 stem（0.36R0.xls）。
    方式 B：c30_group_source_map.json 显式路径（仅当 A 未命中）。
    不再按 xls 文件名猜测 w/b×R。
    返回 (path, mode) mode in stem_match | explicit_map | None。
    """
    if bmp is not None:
        for ext in (".xls", ".xlsx"):
            cand = bmp.parent / f"{bmp.stem}{ext}"
            if cand.exists():
                return cand.resolve(), "stem_match"

    mapped = group_map.get(sample_id)
    if mapped:
        p = _resolve_map_path(mapped)
        if p is not None:
            return p, "explicit_map"
        return None, "explicit_map_missing_file"

    return None, None


def _audit_excel(xpath: Path) -> dict[str, Any]:
    rel = str(xpath.relative_to(_ROOT))
    wb, r_code, r_pct = _parse_group_from_name(xpath.name)
    entry: dict[str, Any] = {
        "source_file": rel,
        "parsed_w_b_ratio": wb,
        "parsed_restraint_code": r_code,
        "parsed_restraint_percent": r_pct,
        "read_ok": False,
        "sheets": [],
        "error": None,
    }
    try:
        sheets = _read_excel_sheets(xpath)
        for sname, sdf in sheets.items():
            entry["sheets"].append(
                {
                    "name": sname,
                    "n_rows": int(len(sdf)),
                    "columns": [str(c) for c in sdf.columns],
                }
            )
        ts, sheet_meta, selected = _pick_best_sheet(sheets)
        entry["read_ok"] = True
        entry["selected_sheet"] = selected
        entry["column_map"] = sheet_meta.get("column_map")
        entry["unrecognized_columns"] = sheet_meta.get("unrecognized_columns")
        entry["n_timeseries_rows"] = int(len(ts))
        entry["_timeseries"] = ts
    except Exception as e:
        entry["error"] = str(e)
        entry["_timeseries"] = pd.DataFrame()
    return entry


def _expected_sample_ids() -> list[tuple[str, float, str, int]]:
    rows: list[tuple[str, float, str, int]] = []
    for wb in WB_VALUES:
        for rc in ("R0", "R50", "R100"):
            rows.append((_sample_id(wb, rc), wb, rc, RESTRAINT_MAP[rc]))
    return rows


def run_import(raw_dir: Path) -> dict[str, Any]:
    excel_files, bmp_files, other_files = _find_raw_files(raw_dir)
    group_map = _load_group_map()
    report: dict[str, Any] = {
        "import_type": "c30_temperature_stress_validation",
        "imported_at_iso": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "raw_dir": str(raw_dir.relative_to(_ROOT) if raw_dir.is_relative_to(_ROOT) else raw_dir),
        "strength_grade": STRENGTH_GRADE,
        "expected_groups": 9,
        "mapping_file": str(MAP_JSON.relative_to(_ROOT)) if MAP_JSON.exists() else None,
        "mapping_entries_loaded": len(group_map),
        "files_found": {
            "excel": [str(p.relative_to(_ROOT)) for p in excel_files],
            "bmp": [str(p.relative_to(_ROOT)) for p in bmp_files],
            "other": [str(p.relative_to(_ROOT)) for p in other_files],
        },
        "excel_audit": [],
        "groups": [],
        "missing_groups": [],
        "unmapped_excel_files": [],
        "groups_needing_manual_review": [],
        "field_mapping_reference": {k: list(v) for k, v in COLUMN_ALIASES.items()},
        "notes": [
            "restraint_percent 为试验约束档位（0/50/100），不等同于 Phase1 restraint_factor_R。",
            "本导入不写入 training_data.csv，不生成 crack 标签。",
            "方式 A：bmp 同目录同名 stem（如 0.36R0.xls ↔ 0.36R0.bmp）。",
            "方式 B：data/thermal_stress/c30_group_source_map.json 显式映射（A 未命中时使用）。",
            "方式 A 下同一 xls 不可重复分配给多个分组；方式 B 允许映射文件显式指向同一 xls。",
            "未建立映射的分组写入 missing_groups，不强行导入。",
        ],
    }

    all_ts_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    imported_ids: set[str] = set()
    stem_match_used: dict[str, str] = {}

    excel_cache: dict[str, dict[str, Any]] = {}
    audited_rels: set[str] = set()

    def _get_excel_entry(xls_path: Path) -> dict[str, Any]:
        rel = str(xls_path.relative_to(_ROOT))
        if rel not in excel_cache:
            entry = _audit_excel(xls_path)
            excel_cache[rel] = entry
            if rel not in audited_rels:
                report["excel_audit"].append(
                    {k: v for k, v in entry.items() if not k.startswith("_")}
                )
                audited_rels.add(rel)
        return excel_cache[rel]

    for xpath in excel_files:
        _get_excel_entry(xpath)

    for sid, wb, r_code, r_pct in _expected_sample_ids():
        bmp = _find_bmp_for_group(wb, r_code, bmp_files)
        xls_path, mode = _resolve_xls_for_group(sid, bmp, group_map)
        img_name = bmp.name if bmp else ""

        if xls_path is None:
            reason = "缺少同名 Excel 且映射文件未配置"
            if mode == "explicit_map_missing_file":
                reason = f"映射文件已配置但路径不存在：{group_map.get(sid)}"
            report["missing_groups"].append(
                {
                    "sample_id": sid,
                    "w_b_ratio": wb,
                    "restraint_code": r_code,
                    "source_image": img_name or None,
                    "reason": reason,
                }
            )
            continue

        rel = str(xls_path.relative_to(_ROOT))

        if mode == "stem_match":
            if rel in stem_match_used and stem_match_used[rel] != sid:
                report["missing_groups"].append(
                    {
                        "sample_id": sid,
                        "w_b_ratio": wb,
                        "restraint_code": r_code,
                        "source_image": img_name or None,
                        "source_file": rel,
                        "reason": (
                            f"同名匹配冲突：{rel} 已用于 {stem_match_used[rel]}；"
                            "请改用 c30_group_source_map.json 显式分配或提供独立 xls"
                        ),
                    }
                )
                report["groups_needing_manual_review"].append(
                    {"sample_id": sid, "source_file": rel, "reason": "stem_match 重复占用"}
                )
                continue
            stem_match_used[rel] = sid

        entry = _get_excel_entry(xls_path)
        ts: pd.DataFrame = entry.get("_timeseries", pd.DataFrame())
        if not entry.get("read_ok"):
            report["missing_groups"].append(
                {
                    "sample_id": sid,
                    "source_file": rel,
                    "mapping_mode": mode,
                    "reason": f"Excel 读取失败：{entry.get('error')}",
                }
            )
            continue

        if len(ts) == 0:
            report["missing_groups"].append(
                {
                    "sample_id": sid,
                    "source_file": rel,
                    "mapping_mode": mode,
                    "reason": "Excel 已读但未提取到时间序列",
                }
            )
            report["groups_needing_manual_review"].append(
                {"sample_id": sid, "reason": "空时间序列"}
            )
            continue

        imported_ids.add(sid)
        for _, row in ts.iterrows():
            all_ts_rows.append(
                {
                    "sample_id": sid,
                    "strength_grade": STRENGTH_GRADE,
                    "w_b_ratio": wb,
                    "restraint_code": r_code,
                    "restraint_percent": r_pct,
                    "time_h": row.get("time_h"),
                    "specimen_temperature_c": row.get("specimen_temperature_c"),
                    "axial_stress_mpa": row.get("axial_stress_mpa"),
                    "deformation_um": row.get("deformation_um"),
                    "source_file": rel,
                    "source_image": img_name,
                    "mapping_mode": mode,
                }
            )

        summ = _summarize_group(ts)
        summary_rows.append(
            {
                "sample_id": sid,
                "strength_grade": STRENGTH_GRADE,
                "w_b_ratio": wb,
                "restraint_code": r_code,
                "restraint_percent": r_pct,
                **summ,
                "source_file": rel,
                "source_image": img_name,
                "mapping_mode": mode,
            }
        )
        report["groups"].append(
            {
                "sample_id": sid,
                "w_b_ratio": wb,
                "restraint_code": r_code,
                "restraint_percent": r_pct,
                "mapping_mode": mode,
                "n_timeseries_points": int(len(ts)),
                "source_file": rel,
                "source_image": img_name or None,
                **{k: summ.get(k) for k in (
                    "max_temperature_c",
                    "max_tensile_stress_mpa",
                    "max_compressive_stress_mpa",
                    "final_stress_mpa",
                )},
            }
        )

    used_rels = {g["source_file"] for g in report["groups"]}
    for xpath in excel_files:
        rel = str(xpath.relative_to(_ROOT))
        if rel not in used_rels:
            report["unmapped_excel_files"].append(rel)

    ts_cols = [
        "sample_id", "strength_grade", "w_b_ratio", "restraint_code", "restraint_percent",
        "time_h", "specimen_temperature_c", "axial_stress_mpa", "deformation_um",
        "source_file", "source_image", "mapping_mode",
    ]
    sum_cols = [
        "sample_id", "strength_grade", "w_b_ratio", "restraint_code", "restraint_percent",
        "max_temperature_c", "time_at_max_temperature_h",
        "max_tensile_stress_mpa", "time_at_max_tensile_stress_h",
        "max_compressive_stress_mpa", "time_at_max_compressive_stress_h",
        "final_stress_mpa", "max_deformation_um", "min_deformation_um", "final_deformation_um",
        "temperature_rise_rate_est", "cooling_rate_est", "curve_quality_note",
        "source_file", "source_image", "mapping_mode",
    ]

    ts_df = pd.DataFrame(all_ts_rows, columns=ts_cols)
    sum_df = pd.DataFrame(summary_rows, columns=sum_cols)

    OUT_TS.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    ts_df.to_csv(OUT_TS, index=False, encoding="utf-8-sig")
    sum_df.to_csv(OUT_SUM, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(
        json.dumps(
            {"standard_columns": list(COLUMN_ALIASES.keys()), "aliases": {k: list(v) for k, v in COLUMN_ALIASES.items()}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report["outputs"] = {
        "timeseries_csv": str(OUT_TS.relative_to(_ROOT)),
        "summary_csv": str(OUT_SUM.relative_to(_ROOT)),
        "field_mapping_json": str(OUT_JSON.relative_to(_ROOT)),
        "group_source_map_json": str(MAP_JSON.relative_to(_ROOT)),
        "n_timeseries_rows": int(len(ts_df)),
        "n_summary_rows": int(len(sum_df)),
        "n_groups_imported": len(summary_rows),
        "n_missing_groups": len(report["missing_groups"]),
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(_render_md(report), encoding="utf-8")
    return report


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# C30 温度应力试验数据导入报告",
        "",
        f"**导入时间（UTC）：** {report.get('imported_at_iso', '—')}",
        f"**原始目录：** `{report.get('raw_dir', '—')}`",
        "",
        "## 1. 成功读取的文件",
        "",
    ]
    ff = report.get("files_found") or {}
    for kind in ("excel", "bmp", "other"):
        items = ff.get(kind) or []
        lines.append(f"### {kind.upper()}（{len(items)}）")
        lines.extend([f"- `{x}`" for x in items] if items else ["- *未发现*"])
        lines.append("")

    lines.extend(["## 2. Excel sheet 与列审计", ""])
    for ex in report.get("excel_audit") or []:
        lines.append(f"### `{ex.get('source_file')}`")
        lines.append(f"- 读取：{'成功' if ex.get('read_ok') else '失败'}")
        if ex.get("error"):
            lines.append(f"- 错误：`{ex['error']}`")
        lines.append(f"- 解析分组：w/b={ex.get('parsed_w_b_ratio')}，{ex.get('parsed_restraint_code')}")
        for sh in ex.get("sheets") or []:
            lines.append(f"- Sheet `{sh.get('name')}`：{sh.get('n_rows')} 行")
        if ex.get("column_map"):
            lines.append(f"- 字段映射：{json.dumps(ex['column_map'], ensure_ascii=False)}")
        lines.append("")

    lines.extend(["## 3. 分组映射（3×3）", ""])
    for g in report.get("groups") or []:
        lines.append(
            f"- **{g.get('sample_id')}** [{g.get('mapping_mode')}]："
            f"{g.get('n_timeseries_points')} 点；"
            f"T_max={g.get('max_temperature_c')}；σ_t_max={g.get('max_tensile_stress_mpa')}"
        )
    if not report.get("groups"):
        lines.append("- *尚无成功导入的分组*")
    lines.append("")

    lines.extend(["## 4. 缺失分组（missing_groups）", ""])
    for item in report.get("missing_groups") or []:
        lines.append(f"- {item}")
    if not report.get("missing_groups"):
        lines.append("- 无")
    lines.append("")

    lines.extend(["## 5. 需人工复核", ""])
    for item in report.get("groups_needing_manual_review") or []:
        lines.append(f"- {item}")
    lines.append("")

    lines.extend(["## 6. 未映射 Excel", ""])
    um = report.get("unmapped_excel_files") or []
    lines.extend([f"- `{x}`" for x in um] if um else ["- 无"])
    lines.append("")

    out = report.get("outputs") or {}
    lines.extend([
            "## 7. 输出产物",
            "",
            f"- `{out.get('timeseries_csv')}`（{out.get('n_timeseries_rows', 0)} 行）",
            f"- `{out.get('summary_csv')}`（{out.get('n_summary_rows', 0)} 行）",
            f"- `{out.get('group_source_map_json')}`",
            f"- `{REPORT_JSON.relative_to(_ROOT)}`",
            f"- 已导入 {out.get('n_groups_imported', 0)} 组，缺失 {out.get('n_missing_groups', 0)} 组",
            "",
            "## 8. 边界说明",
            "",
            "- 未并入 `training_data.csv`；未修改主模型训练/推理。",
            "- `restraint_percent` ≠ `restraint_factor_R`（后续单独映射）。",
            "- bmp 仅用于核对与展示，未作为主数据源抠数。",
            "- **方式 A**：同名 stem 自动匹配；**方式 B**：`c30_group_source_map.json`。",
            "- 方式 A 禁止同一 xls 重复分配；方式 B 允许映射文件显式共用同一 xls。",
            "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="导入 C30 温度应力试验 Excel 至独立验证 CSV")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = ap.parse_args()
    rep = run_import(args.raw_dir.resolve())
    print(json.dumps(rep.get("outputs", {}), ensure_ascii=False, indent=2))
    if rep.get("outputs", {}).get("n_groups_imported", 0) == 0:
        print(
            "\n[WARN] 未导入任何试验组。"
            "方式 A：bmp 同目录放置同名 xls（如 0.36R0.xls）。"
            "方式 B：填写 data/thermal_stress/c30_group_source_map.json 后重新运行。",
            file=sys.stderr,
        )
    elif rep.get("outputs", {}).get("n_missing_groups", 0) > 0:
        print(
            f"\n[INFO] 部分分组未导入，见 missing_groups（{rep['outputs']['n_missing_groups']} 组）。",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
