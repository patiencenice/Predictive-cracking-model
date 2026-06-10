"""
将人工填写的治理模板按 row_index 合并回主开裂训练 CSV。

仅更新 sidecar 治理列；不修改特征列、标签列；不自动填写任何默认值。

用法（在 SteelFiberCrackPredictor 目录下）:
  py scripts/apply_crack_governance_fill_template.py --dry-run
  py scripts/apply_crack_governance_fill_template.py
  py scripts/apply_crack_governance_fill_template.py --template outputs/crack_governance/crack_governance_fill_template.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_init_path = Path(__file__).resolve().parent / "init_crack_governance_columns.py"
_spec = importlib.util.spec_from_file_location(
    "init_crack_governance_columns", _init_path
)
_init_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_init_mod)
CRACK_GOVERNANCE_COLUMN_NAMES: tuple[str, ...] = _init_mod.CRACK_GOVERNANCE_COLUMN_NAMES

DEFAULT_TARGET = _ROOT / "data" / "training_data.csv"
DEFAULT_TEMPLATE = _ROOT / "outputs" / "crack_governance" / "crack_governance_fill_template.csv"
DIAGNOSE_SCRIPT = _ROOT / "scripts" / "diagnose_crack_training_governance.py"

VALID_DATA_TIERS = frozenset(
    {
        "A_lab_native",
        "B_literature_verified",
        "C_literature_extracted",
    }
)
VALID_CRACK_WIDTH_DEFINITION_IDS = frozenset(
    {
        "CW_MAX_SURFACE_MM",
        "CW_MEAN_MM",
        "CW_INNER_MM",
        "CW_UNSPECIFIED",
    }
)

# 模板中的对照列：只读，禁止写回主表
FORBIDDEN_WRITE_COLUMNS = frozenset(
    {
        "row_index",
        "strength_grade",
        "fiber_type",
        "fiber_content",
        "aspect_ratio",
        "w_b_ratio",
        "cube_strength_mpa",
        "curing_days",
        "crack_width",
        "crack_density",
        "cracking_risk",
        "fill_note",
        "reviewer",
        "review_date",
    }
)


def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


def _cell_filled(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return _norm_str(v) != ""


def _validate_row_index(template: pd.DataFrame, n_target: int) -> list[str]:
    problems: list[str] = []
    if "row_index" not in template.columns:
        problems.append("模板缺少列 row_index")
        return problems
    ri = pd.to_numeric(template["row_index"], errors="coerce")
    if ri.isna().any():
        problems.append("row_index 存在无法解析为整数的行")
    ri_int = ri.dropna().astype(int)
    if len(ri_int) != len(template):
        problems.append("row_index 有效行数与模板行数不一致")
    if ri_int.duplicated().any():
        dups = ri_int[ri_int.duplicated()].unique().tolist()
        problems.append(f"row_index 重复: {dups[:20]}")
    expected = set(range(n_target))
    got = set(ri_int.tolist())
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        if missing:
            problems.append(f"row_index 缺少: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        if extra:
            problems.append(f"row_index 多余或越界: {extra[:20]}{'...' if len(extra) > 20 else ''}")
    if len(template) != n_target:
        problems.append(
            f"模板行数 {len(template)} 与主表行数 {n_target} 不一致"
        )
    return problems


def _validate_governance_cell(col: str, raw: Any) -> str | None:
    """返回错误信息；通过则返回 None。空值不校验（表示不更新）。"""
    if not _cell_filled(raw):
        return None
    if col == "data_tier":
        v = _norm_str(raw)
        if v not in VALID_DATA_TIERS:
            return f"非法 data_tier: {v!r}"
        return None
    if col == "crack_width_definition_id":
        v = _norm_str(raw)
        if v not in VALID_CRACK_WIDTH_DEFINITION_IDS:
            return f"非法 crack_width_definition_id: {v!r}"
        return None
    if col == "needs_manual_review":
        try:
            x = float(raw)
        except (TypeError, ValueError):
            return f"非法 needs_manual_review: {raw!r}"
        if x not in (0.0, 1.0):
            return f"needs_manual_review 须为 0 或 1，当前: {raw!r}"
        return None
    return None


def _coerce_governance_value(col: str, raw: Any) -> Any:
    if col == "needs_manual_review":
        return int(float(raw))
    return _norm_str(raw)


def _target_gov_value(target: pd.DataFrame, row_i: int, col: str) -> str:
    if col not in target.columns:
        return ""
    return _norm_str(target.at[row_i, col])


def plan_merge(
    target: pd.DataFrame,
    template: pd.DataFrame,
) -> tuple[dict[str, Any], list[str]]:
    """
    生成合并计划与校验错误列表。
    仅当模板单元格非空且与主表不同时计入更新。
    """
    problems: list[str] = []
    n = len(target)

    for c in CRACK_GOVERNANCE_COLUMN_NAMES:
        if c not in target.columns:
            problems.append(f"主表缺少治理列: {c}")
    if problems:
        return {}, problems

    problems.extend(_validate_row_index(template, n))
    if problems:
        return {}, problems

    tpl = template.set_index(
        pd.to_numeric(template["row_index"], errors="coerce").astype(int),
        drop=False,
    )

    illegal_data_tier: list[dict[str, Any]] = []
    illegal_def_id: list[dict[str, Any]] = []
    illegal_review: list[dict[str, Any]] = []

    updates: list[dict[str, Any]] = []
    cols_touched: set[str] = set()
    rows_touched: set[int] = set()

    for row_i in range(n):
        if row_i not in tpl.index:
            problems.append(f"模板缺少 row_index={row_i}")
            continue
        trow = tpl.loc[row_i]
        if isinstance(trow, pd.DataFrame):
            trow = trow.iloc[0]

        for col in CRACK_GOVERNANCE_COLUMN_NAMES:
            raw = trow[col] if col in trow.index else None
            err = _validate_governance_cell(col, raw)
            if err:
                rec = {"row_index": row_i, "column": col, "value": raw, "error": err}
                if col == "data_tier":
                    illegal_data_tier.append(rec)
                elif col == "crack_width_definition_id":
                    illegal_def_id.append(rec)
                elif col == "needs_manual_review":
                    illegal_review.append(rec)
                problems.append(f"row {row_i} {err}")
                continue
            if not _cell_filled(raw):
                continue
            new_v = _coerce_governance_value(col, raw)
            old_s = _target_gov_value(target, row_i, col)
            new_s = _norm_str(new_v) if col != "needs_manual_review" else str(int(new_v))
            if old_s == new_s:
                continue
            updates.append(
                {
                    "row_index": row_i,
                    "column": col,
                    "old": old_s,
                    "new": new_s,
                }
            )
            cols_touched.add(col)
            rows_touched.add(row_i)

    plan = {
        "n_target_rows": n,
        "n_template_rows": int(len(template)),
        "cells_to_update": len(updates),
        "rows_to_update": sorted(rows_touched),
        "columns_to_update": sorted(cols_touched),
        "updates_sample": updates[:50],
        "illegal_data_tier": illegal_data_tier,
        "illegal_crack_width_definition_id": illegal_def_id,
        "illegal_needs_manual_review": illegal_review,
        "governance_columns_only": list(CRACK_GOVERNANCE_COLUMN_NAMES),
        "forbidden_write_columns": sorted(FORBIDDEN_WRITE_COLUMNS),
    }
    return plan, problems


def apply_merge(target: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    tpl = template.set_index(
        pd.to_numeric(template["row_index"], errors="coerce").astype(int)
    )

    for row_i in range(len(out)):
        trow = tpl.loc[row_i]
        if isinstance(trow, pd.DataFrame):
            trow = trow.iloc[0]
        for col in CRACK_GOVERNANCE_COLUMN_NAMES:
            raw = trow[col] if col in trow.index else None
            if _validate_governance_cell(col, raw) is not None:
                continue
            if not _cell_filled(raw):
                continue
            out.at[row_i, col] = _coerce_governance_value(col, raw)
    return out


def _run_diagnose(csv_path: Path) -> int:
    if not DIAGNOSE_SCRIPT.exists():
        print(f"未找到诊断脚本: {DIAGNOSE_SCRIPT}")
        return 1
    proc = subprocess.run(
        [sys.executable, str(DIAGNOSE_SCRIPT), "--csv", str(csv_path)],
        cwd=str(_ROOT),
        check=False,
    )
    return int(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="将治理补值模板合并回 training_data（仅治理列）"
    )
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    ap.add_argument("--encoding", type=str, default="utf-8-sig")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只报告将更新的单元格与校验结果，不写盘",
    )
    ap.add_argument(
        "--skip-diagnose",
        action="store_true",
        help="写入后不重跑 diagnose_crack_training_governance.py",
    )
    args = ap.parse_args()

    if not args.target.exists():
        print(f"主表不存在: {args.target}")
        raise SystemExit(1)
    if not args.template.exists():
        print(f"模板不存在: {args.template}")
        raise SystemExit(1)

    target = pd.read_csv(args.target, encoding=args.encoding)
    template = pd.read_csv(args.template, encoding=args.encoding)

    # 模板中若有人误改特征/标签列，本脚本忽略，仅读取治理列
    plan, problems = plan_merge(target, template)

    report: dict[str, Any] = {
        "dry_run": bool(args.dry_run),
        "target_csv": str(args.target.resolve()),
        "template_csv": str(args.template.resolve()),
        "plan": plan,
        "validation_ok": len(problems) == 0,
        "validation_errors": problems,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if problems:
        print("校验未通过，未写入主表。")
        raise SystemExit(1)

    if args.dry_run:
        print("dry-run 完成，未修改主表。")
        return

    merged = apply_merge(target, template)
    merged.to_csv(args.target, index=False, encoding=args.encoding)
    print(f"已写入主表: {args.target.resolve()}")

    if args.skip_diagnose:
        print("请手动执行: py scripts/diagnose_crack_training_governance.py")
    else:
        print("正在重跑治理诊断…")
        code = _run_diagnose(args.target)
        if code != 0:
            print(f"诊断脚本退出码: {code}")


if __name__ == "__main__":
    main()
