"""
论文表（易梦）补列：生成 paper_yimeng_28d_prepared_review.csv 供人工复核。

- 不覆盖原始 CSV。
- 仅对指定缺失列填空；已有非空数值不覆盖。
- 补完后写 JSON 报告并可选调用 gate 脚本（不训练）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import FIBER_MATERIAL_MAP, STRENGTH_GRADE_ENC, STRENGTH_GRADE_TO_MPA

# 读取时依次尝试的编码
READ_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030")

# 本脚本负责补全的列（不改动表内其它已有数值列）
COLUMNS_TO_FILL = [
    "fiber_type_enc",
    "fiber_material_enc",
    "cube_strength_mpa",
    "strength_grade_enc",
    "concrete_type_enc",
    "slag_grade_enc",
    "sand_type_enc",
    "sand_ratio",
    "admixture_enc",
    "temperature",
    "humidity",
    "casting_method_enc",
]


def _read_csv_auto_encoding(path: Path) -> tuple[pd.DataFrame, str]:
    """尝试多种编码读取 CSV。"""
    last_err: str | None = None
    for enc in READ_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError as e:
            last_err = f"{enc}: {e}"
            continue
    raise ValueError(f"无法用编码 {READ_ENCODINGS} 读取: {last_err}")


def _fill_scalar_where_missing(df: pd.DataFrame, col: str, value: Any) -> None:
    """列不存在则新建；存在则只对缺失位置写入标量 value。"""
    if col not in df.columns:
        df[col] = value
        return
    m = df[col].isna()
    df.loc[m, col] = value


def _fill_series_where_missing(df: pd.DataFrame, col: str, series: pd.Series) -> None:
    """对缺失格写入与 series 对齐的值。"""
    if col not in df.columns:
        df[col] = series
        return
    m = df[col].isna()
    df.loc[m, col] = series[m]


def _map_grade(grade: object) -> tuple[float | None, float | None]:
    """strength_grade -> (cube_strength_mpa, strength_grade_enc)，严格用 features 字典。"""
    if grade is None or (isinstance(grade, float) and pd.isna(grade)):
        return None, None
    g = str(grade).strip().upper()
    if g not in STRENGTH_GRADE_TO_MPA:
        return None, None
    return float(STRENGTH_GRADE_TO_MPA[g]), float(STRENGTH_GRADE_ENC[g])


def _fiber_material_from_mix_label(label: object) -> tuple[float | None, str]:
    """mix_label 子串 -> (fiber_material_enc, 备注片段)。"""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None, ""
    s = str(label).strip()
    if "基准" in s:
        return (
            0.0,
            "基准组无纤维，fiber_material_enc=0 仅为兼容 FEATURE_COLUMNS 的占位编码",
        )
    if "钢纤维" in s:
        return float(FIBER_MATERIAL_MAP["钢纤维"]), ""
    if "玄武岩" in s:
        return float(FIBER_MATERIAL_MAP["玄武岩纤维"]), ""
    if "聚丙烯" in s:
        return float(FIBER_MATERIAL_MAP["聚丙烯纤维"]), ""
    if "玻璃" in s:
        return float(FIBER_MATERIAL_MAP["玻璃纤维"]), ""
    return None, "mix_label 未匹配已知纤维关键词"


def _sand_ratio_one(s: Any, g: Any) -> float | None:
    """单行 sand_ratio；不可算则 None。"""
    try:
        fs = float(s)
        fg = float(g)
    except (TypeError, ValueError):
        return None
    if pd.isna(s) or pd.isna(g):
        return None
    if fs + fg <= 0:
        return None
    return float(fs / (fs + fg) * 100.0)


def _row_fill_status(
    sand_ratio_val: Any,
    fiber_type_val: Any,
    mix_label: Any,
    *,
    assumed_concrete: bool,
    assumed_casting: bool,
) -> str:
    """
    单行状态（互斥）：
    - needs_manual_review：fiber_type_enc 仍空，或 sand_ratio 无法计算
    - ok_with_assumption：基准组 fiber 占位、或本行曾对 concrete/casting 做工程占位补全
    - ok_direct：fiber_type 与 sand_ratio 均有效，且本行无上述占位假设
    """
    sand_bad = sand_ratio_val is None or (
        not isinstance(sand_ratio_val, (int, float)) and pd.isna(sand_ratio_val)
    )
    if isinstance(sand_ratio_val, (int, float)) and not pd.isna(sand_ratio_val):
        sand_bad = False
    fiber_bad = fiber_type_val is None or pd.isna(fiber_type_val)

    if sand_bad or fiber_bad:
        return "needs_manual_review"

    mix_s = str(mix_label).strip() if mix_label is not None and not pd.isna(mix_label) else ""
    if "基准" in mix_s or assumed_concrete or assumed_casting:
        return "ok_with_assumption"
    return "ok_direct"


def prepare_review_csv(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    root: Path | None = None,
    run_gate: bool = True,
) -> dict[str, Any]:
    root = root or _ROOT
    inp = (root / input_path).resolve() if not input_path.is_absolute() else input_path
    outp = (root / output_path).resolve() if not output_path.is_absolute() else output_path
    rep = (root / report_path).resolve() if not report_path.is_absolute() else report_path

    df, enc = _read_csv_auto_encoding(inp)
    df = df.copy()
    # 记录原始列名，用于报告「新增列」
    base_cols = set(df.columns)
    n_rows = len(df)
    warning_items: list[str] = []

    # 记录哪些行在本脚本中对 concrete/casting 做了「缺失补占位」（用于 row_fill_status）
    conc_was_na = (
        df["concrete_type_enc"].isna()
        if "concrete_type_enc" in df.columns
        else pd.Series(True, index=df.index)
    )
    cast_was_na = (
        df["casting_method_enc"].isna()
        if "casting_method_enc" in df.columns
        else pd.Series(True, index=df.index)
    )

    # ----- 整表常量 -----
    _fill_scalar_where_missing(df, "slag_grade_enc", 1.0)  # 论文 S95
    _fill_scalar_where_missing(df, "sand_type_enc", 0.0)  # 天然砂
    _fill_scalar_where_missing(df, "admixture_enc", 1.0)  # 减水剂类
    _fill_scalar_where_missing(df, "temperature", 20.0)
    _fill_scalar_where_missing(df, "humidity", 95.0)
    _fill_scalar_where_missing(df, "concrete_type_enc", 0.0)
    _fill_scalar_where_missing(df, "casting_method_enc", 0.0)

    # ----- 强度 -----
    if "strength_grade" not in df.columns:
        warning_items.append("缺少 strength_grade，cube/strength_enc 无法映射")
        cube_ser = pd.Series([pd.NA] * n_rows, index=df.index)
        enc_ser = pd.Series([pd.NA] * n_rows, index=df.index)
    else:
        cubes, encs = [], []
        for _, row in df.iterrows():
            c, e = _map_grade(row["strength_grade"])
            cubes.append(c)
            encs.append(e)
        cube_ser = pd.Series(cubes, index=df.index)
        enc_ser = pd.Series(encs, index=df.index)
    _fill_series_where_missing(df, "cube_strength_mpa", cube_ser)
    _fill_series_where_missing(df, "strength_grade_enc", enc_ser)

    # ----- sand_ratio -----
    if "sand_content" not in df.columns or "stone_content" not in df.columns:
        warning_items.append("缺少 sand_content 或 stone_content，sand_ratio 无法计算")
        sr = pd.Series([pd.NA] * n_rows, index=df.index)
    else:
        sc = pd.to_numeric(df["sand_content"], errors="coerce")
        st = pd.to_numeric(df["stone_content"], errors="coerce")
        sr_list: list[Any] = []
        for i in df.index:
            sr_list.append(_sand_ratio_one(sc.loc[i], st.loc[i]))
        sr = pd.Series(sr_list, index=df.index)
        n_fail = int(sr.isna().sum())
        if n_fail:
            warning_items.append(
                f"sand_ratio：{n_fail}/{n_rows} 行无法计算（缺数或分母为 0），已留空"
            )
    _fill_series_where_missing(df, "sand_ratio", sr)

    # ----- fiber_material_enc -----
    if "mix_label" not in df.columns:
        warning_items.append("缺少 mix_label，fiber_material_enc 无法映射")
        fm = pd.Series([pd.NA] * n_rows, index=df.index)
        fm_note = [""] * n_rows
    else:
        fv, notes = [], []
        for _, row in df.iterrows():
            v, n = _fiber_material_from_mix_label(row["mix_label"])
            fv.append(v)
            notes.append(n)
        fm = pd.Series(fv, index=df.index)
        fm_note = notes
    _fill_series_where_missing(df, "fiber_material_enc", fm)

    # ----- fiber_type_enc：不推断，缺失 + 布尔列 -----
    if "fiber_type_enc" not in df.columns:
        df["fiber_type_enc"] = pd.NA
    df["fiber_type_enc_needs_manual_review"] = True

    # ----- 审核列 -----
    static_rules = (
        "slag_grade_enc=1：论文矿粉等级 S95；"
        "sand_type_enc=0：论文天然砂；"
        "admixture_enc=1：聚羧酸减水剂归为减水剂；"
        "temperature=20、humidity=95：标准养护 20±2℃、RH>95% 之名义值；"
        "concrete_type_enc=0：普通混凝土，依据论文配合比作工程假设，待人工确认；"
        "casting_method_enc=0：常规浇筑，待人工确认；"
        "fiber_material_enc：mix_label 子串/基准占位规则；"
        "sand_ratio：sand_content/(sand_content+stone_content)*100；"
        "fiber_type_enc：未从现有列推断。"
    )
    df["source_fill_rule_summary"] = static_rules

    review_col: list[str] = []
    status_col: list[str] = []
    for j, i in enumerate(df.index):
        segs = [
            "temperature=20、humidity=95：来源为标准养护 20±2℃、RH>95%，此处取名义值 20 与 95",
            "concrete_type_enc=0：普通混凝土，依据论文配合比作工程假设，待人工确认",
            "casting_method_enc=0：常规浇筑，待人工确认",
        ]
        if "mix_label" in df.columns and fm_note[j]:
            segs.append(fm_note[j])
        review_col.append(" | ".join(segs))

        st = _row_fill_status(
            df.loc[i, "sand_ratio"] if "sand_ratio" in df.columns else None,
            df.loc[i, "fiber_type_enc"] if "fiber_type_enc" in df.columns else None,
            df.loc[i, "mix_label"] if "mix_label" in df.columns else None,
            assumed_concrete=bool(conc_was_na.loc[i]),
            assumed_casting=bool(cast_was_na.loc[i]),
        )
        status_col.append(st)

    df["review_note"] = review_col
    df["row_fill_status"] = status_col

    # 各补全列非空计数（补全操作后）
    count_keys = COLUMNS_TO_FILL + [
        "fiber_type_enc_needs_manual_review",
        "review_note",
        "source_fill_rule_summary",
        "row_fill_status",
    ]
    filled_counts: dict[str, int] = {}
    for c in count_keys:
        if c not in df.columns:
            filled_counts[c] = 0
        elif c == "fiber_type_enc_needs_manual_review" and df[c].dtype == bool:
            filled_counts[c] = int(df[c].sum())
        else:
            filled_counts[c] = int(df[c].notna().sum())

    n_manual = int((df["row_fill_status"] == "needs_manual_review").sum())

    added_col_names = [c for c in df.columns if c not in base_cols]

    report: dict[str, Any] = {
        "input_path": str(inp),
        "output_path": str(outp),
        "read_encoding": enc,
        "n_rows": n_rows,
        "added_columns": added_col_names,
        "filled_counts_by_column": {
            k: filled_counts[k] for k in filled_counts if k in df.columns
        },
        "n_rows_needs_manual_review": n_manual,
        "warning_items": warning_items,
    }

    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False, encoding="utf-8-sig")

    rep.parent.mkdir(parents=True, exist_ok=True)
    if run_gate:
        gate_script = root / "scripts" / "gate_paper_yimeng_lab_strength.py"
        proc = subprocess.run(
            [sys.executable, str(gate_script), "--input", str(outp)],
            capture_output=True,
            text=True,
            cwd=str(root),
        )
        report["gate_exit_code"] = proc.returncode
        report["gate_ok"] = proc.returncode == 0
        report["gate_stdout_tail"] = (proc.stdout or "")[-4000:]
        report["gate_stderr_tail"] = (proc.stderr or "")[-4000:]
        if proc.returncode != 0:
            # 仅打印失败原因（闸门 stdout 已含问题清单）
            print("--- gate 未通过（不训练）---")
            print(proc.stdout or proc.stderr or f"exit={proc.returncode}")
        else:
            print("--- gate 已通过 ---")

    with open(rep, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def run_cli() -> None:
    import argparse

    root = _ROOT
    ap = argparse.ArgumentParser(description="易梦论文表补列 -> prepared_review")
    ap.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "lab_strength" / "paper_yimeng_28d_final_for_lab_strength_residual.csv",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "lab_strength" / "paper_yimeng_28d_prepared_review.csv",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=root / "outputs" / "lab_strength" / "paper_yimeng_prepare_report.json",
    )
    ap.add_argument("--no-gate", action="store_true", help="补完后不自动运行 gate")
    args = ap.parse_args()

    prepare_review_csv(
        args.input,
        args.output,
        args.report,
        root=root,
        run_gate=not args.no_gate,
    )
    print("已写入:", args.output.resolve())
    print("报告:", args.report.resolve())


if __name__ == "__main__":
    run_cli()
