"""
易梦论文表：仅补全可由现有列合法推导的字段，并生成缺口说明（不训练）。

见 scripts/prepare_paper_yimeng_lab_strength_partial.py 入口。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_BOOT = Path(__file__).resolve().parent.parent.parent
if str(_BOOT) not in sys.path:
    sys.path.insert(0, str(_BOOT))

import pandas as pd

from src.features import (
    FIBER_MATERIAL_MAP,
    STRENGTH_GRADE_ENC,
    STRENGTH_GRADE_TO_MPA,
)
from src.lab_strength_residual.lab_mix_features import LAB_STRENGTH_FEATURE_COLUMNS
from src.lab_strength_residual.training_data_gate import validate_for_lab_strength_training

MIX_LABEL_TO_MATERIAL_NAME: dict[str, str] = {
    "钢纤维": "钢纤维",
    "玄武岩纤维": "玄武岩纤维",
    "聚丙烯纤维": "聚丙烯纤维",
    "玻璃纤维": "玻璃纤维",
    "基准": "基准",
}

REFERENCE_TRAINING_CSVS: tuple[Path, ...] = (
    Path("data/lab_strength_training.csv"),
    Path("data/lab_strength_training.example.csv"),
    Path("data/training_data.csv"),
    Path("data/training_data.example.csv"),
)


def _project_root() -> Path:
    # .../SteelFiberCrackPredictor/src/lab_strength_residual/this_file.py
    return Path(__file__).resolve().parent.parent.parent


def try_read_csv(path: Path, root: Path) -> tuple[pd.DataFrame, str]:
    full = path if path.is_absolute() else root / path
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(full, encoding=enc), enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法用 utf-8-sig/utf-8/gbk/gb18030 读取: {full}")


def _scan_baseline_fiber_material_enc(root: Path) -> tuple[Path | None, int | None, int]:
    """在仓库既有训练 CSV 中寻找 fiber_content=aspect_ratio=tensile_strength=0 的行，取 fiber_material_enc 众数。"""
    for rel in REFERENCE_TRAINING_CSVS:
        p = root / rel
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, nrows=100_000)
        except Exception:
            continue
        need = ("fiber_content", "aspect_ratio", "tensile_strength", "fiber_material_enc")
        if not all(c in df.columns for c in need):
            continue
        fc = pd.to_numeric(df["fiber_content"], errors="coerce")
        ar = pd.to_numeric(df["aspect_ratio"], errors="coerce")
        ts = pd.to_numeric(df["tensile_strength"], errors="coerce")
        m = (fc == 0) & (ar == 0) & (ts == 0)
        n = int(m.sum())
        if n == 0:
            continue
        mode = df.loc[m, "fiber_material_enc"].mode(dropna=True)
        if len(mode) == 0:
            continue
        return p, int(mode.iloc[0]), n
    return None, None, 0


def _verify_sand_ratio_formula_against_training(root: Path) -> dict[str, Any]:
    """
    检验 sand_ratio 是否恒等于 100*S/(S+G)（S=sand_content, G=stone_content）。
    若在任一可读训练样本上不一致，则认定与现有训练集定义不一致，不计算。
    """
    out: dict[str, Any] = {
        "formula_tested": "100 * sand_content / (sand_content + stone_content)",
        "matches_training_example_csv": None,
        "max_abs_error": None,
        "decision": "do_not_compute",
    }
    p = root / "data" / "lab_strength_training.example.csv"
    if not p.exists():
        out["note"] = "无 lab_strength_training.example.csv，跳过公式对照。"
        return out
    df = pd.read_csv(p)
    if not all(c in df.columns for c in ("sand_ratio", "sand_content", "stone_content")):
        out["note"] = "示例训练集缺少 sand_ratio/sand_content/stone_content 之一。"
        return out
    s = pd.to_numeric(df["sand_content"], errors="coerce")
    g = pd.to_numeric(df["stone_content"], errors="coerce")
    sr = pd.to_numeric(df["sand_ratio"], errors="coerce")
    den = s + g
    ok = den > 0 & sr.notna() & s.notna() & g.notna()
    if not ok.any():
        out["note"] = "无有效行用于对照。"
        return out
    pred = 100.0 * s[ok] / den[ok]
    err = (pred - sr[ok]).abs()
    mx = float(err.max())
    out["max_abs_error"] = mx
    out["matches_training_example_csv"] = bool(mx < 1e-6)
    if mx < 1e-6:
        out["decision"] = "could_compute"
    else:
        out["decision"] = "do_not_compute"
        out["note"] = (
            "在 data/lab_strength_training.example.csv 上，"
            "sand_ratio 与 100*sand/(sand+stone) 最大绝对误差为 "
            f"{mx:.6f}，与「砂率=细骨料质量占比」类公式不一致，故不对易梦表计算 sand_ratio。"
        )
    return out


def prepare_partial(
    input_path: Path,
    output_csv: Path,
    gap_json_path: Path,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or _project_root()
    df, enc = try_read_csv(input_path, root)
    df = df.copy()

    gap: dict[str, Any] = {
        "input_csv": str((root / input_path).resolve() if not input_path.is_absolute() else input_path),
        "read_encoding": enc,
        "filled_columns": [],
        "not_filled_intentionally": [],
        "sand_ratio_decision": _verify_sand_ratio_formula_against_training(root),
    }

    ref_path, baseline_enc, baseline_n = _scan_baseline_fiber_material_enc(root)
    gap["baseline_fiber_material_reference"] = {
        "reference_csv": str(ref_path.resolve()) if ref_path else None,
        "n_matching_rows_in_reference": baseline_n,
        "fiber_material_enc_mode": baseline_enc,
    }

    # --- strength_grade -> cube_strength_mpa, strength_grade_enc ---
    grades = df["strength_grade"].astype(str).str.strip().str.upper() if "strength_grade" in df.columns else None
    cube_list: list[float | None] = []
    enc_list: list[float | None] = []
    bad_grades: list[str] = []
    if grades is None:
        gap["filled_columns"].append(
            {
                "column": "cube_strength_mpa / strength_grade_enc",
                "status": "skipped",
                "reason": "缺少 strength_grade 列",
            }
        )
    else:
        for g in grades:
            if g not in STRENGTH_GRADE_TO_MPA:
                cube_list.append(None)
                enc_list.append(None)
                if g not in bad_grades:
                    bad_grades.append(g)
            else:
                cube_list.append(float(STRENGTH_GRADE_TO_MPA[g]))
                enc_list.append(float(STRENGTH_GRADE_ENC[g]))
        df["cube_strength_mpa"] = cube_list
        df["strength_grade_enc"] = enc_list
        gap["filled_columns"].append(
            {
                "column": "cube_strength_mpa",
                "status": "filled",
                "source": "src.features.STRENGTH_GRADE_TO_MPA，由列 strength_grade 文本映射（如 C30→30）",
            }
        )
        gap["filled_columns"].append(
            {
                "column": "strength_grade_enc",
                "status": "filled",
                "source": "src.features.STRENGTH_GRADE_ENC（与 STRENGTH_GRADE_ORDER 顺序一致，非手写猜测）",
            }
        )
        if bad_grades:
            gap["strength_grade_parse_warnings"] = (
                f"以下等级不在 STRENGTH_GRADE_ORDER 中，对应行 cube/enc 为缺失: {bad_grades}"
            )

    # --- mix_label -> fiber_material_enc ---
    if "mix_label" not in df.columns:
        gap["filled_columns"].append(
            {
                "column": "fiber_material_enc",
                "status": "skipped",
                "reason": "缺少 mix_label",
            }
        )
    else:
        ml = df["mix_label"].astype(str).str.strip()
        enc_mat: list[float | None] = []
        for label in ml:
            if label not in MIX_LABEL_TO_MATERIAL_NAME:
                enc_mat.append(None)
                continue
            if label == "基准":
                if baseline_enc is not None and ref_path is not None:
                    enc_mat.append(float(baseline_enc))
                else:
                    enc_mat.append(None)
                continue
            name = MIX_LABEL_TO_MATERIAL_NAME[label]
            if name not in FIBER_MATERIAL_MAP:
                enc_mat.append(None)
                continue
            enc_mat.append(float(FIBER_MATERIAL_MAP[name]))
        df["fiber_material_enc"] = enc_mat
        gap["filled_columns"].append(
            {
                "column": "fiber_material_enc",
                "status": "partial",
                "source_non_baseline": "mix_label 与 src.features.FIBER_MATERIAL_MAP 键一致时映射（钢纤维/玄武岩纤维/聚丙烯纤维/玻璃纤维）",
                "source_baseline": (
                    f"基准组：使用参考训练集 {ref_path} 中 fiber_content=aspect_ratio=tensile_strength=0 行的 fiber_material_enc 众数={baseline_enc}"
                    if baseline_enc is not None
                    else "基准组：仓库内扫描的参考训练 CSV 中无上述三重零「同类」行，按约定不臆造，fiber_material_enc 保持缺失"
                ),
            }
        )

    # --- sand_ratio ---
    if gap["sand_ratio_decision"].get("decision") == "could_compute":
        s = pd.to_numeric(df["sand_content"], errors="coerce")
        g = pd.to_numeric(df["stone_content"], errors="coerce")
        den = s + g
        df["sand_ratio"] = (100.0 * s / den).where(den > 0)
        gap["filled_columns"].append(
            {
                "column": "sand_ratio",
                "status": "filled",
                "source": gap["sand_ratio_decision"]["formula_tested"],
            }
        )
    else:
        df["sand_ratio"] = pd.NA
        gap["not_filled_intentionally"].append(
            {
                "column": "sand_ratio",
                "reason": gap["sand_ratio_decision"].get("note")
                or "与现有训练集 sand_ratio 定义无法通过 sand_content/stone_content 在示例集上验证为同一公式，整列保留为空（pandas 缺失）。",
            }
        )

    need = list(LAB_STRENGTH_FEATURE_COLUMNS) + [
        "compressive_true",
        "flexural_true",
        "source_group",
    ]
    incomplete: list[str] = []
    for c in need:
        if c not in df.columns:
            incomplete.append(c)
        elif c == "fiber_material_enc" and bool(df[c].isna().any()):
            n_na = int(df[c].isna().sum())
            incomplete.append(f"fiber_material_enc（{n_na}/{len(df)} 行仍缺失）")
        elif c == "sand_ratio" and bool(df[c].isna().all()):
            incomplete.append("sand_ratio（全列缺失，见 sand_ratio_decision）")
        elif bool(df[c].isna().all()):
            incomplete.append(c)

    policy_cols = [
        "fiber_type_enc",
        "concrete_type_enc",
        "slag_grade_enc",
        "sand_type_enc",
        "admixture_enc",
        "temperature",
        "humidity",
        "casting_method_enc",
    ]
    gap["still_missing_or_incomplete_for_gate"] = sorted(set(incomplete + policy_cols))
    gap["columns_held_empty_by_policy"] = [
        "fiber_type_enc",
        "concrete_type_enc",
        "slag_grade_enc",
        "sand_type_enc",
        "admixture_enc",
        "temperature",
        "humidity",
        "casting_method_enc",
    ]
    gap["note_policy"] = (
        "以上 8 列无论文/原始记录则一律不填；本脚本未写入。"
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    gap_json_path.parent.mkdir(parents=True, exist_ok=True)
    gate = validate_for_lab_strength_training(df)
    gap["gate_after_prepare"] = gate

    with open(gap_json_path, "w", encoding="utf-8") as f:
        json.dump(gap, f, ensure_ascii=False, indent=2)

    return gap


def run_cli() -> None:
    import argparse

    root = _project_root()
    ap = argparse.ArgumentParser(description="易梦表部分列推导 + 缺口报告 + gate")
    ap.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "lab_strength" / "paper_yimeng_28d_final_for_lab_strength_residual.csv",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=root / "data" / "lab_strength" / "paper_yimeng_28d_prepared_partial.csv",
    )
    ap.add_argument(
        "--gap-json",
        type=Path,
        default=root / "outputs" / "lab_strength" / "paper_yimeng_partial_gap_report.json",
    )
    args = ap.parse_args()

    prepare_partial(args.input, args.out_csv, args.gap_json, root=root)
    print("已写入:", args.out_csv.resolve())
    print("缺口报告:", args.gap_json.resolve())
    print("--- gate_after_prepare ---")
    with open(args.gap_json, encoding="utf-8") as f:
        g = json.load(f)
    print(json.dumps(g["gate_after_prepare"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_cli()
