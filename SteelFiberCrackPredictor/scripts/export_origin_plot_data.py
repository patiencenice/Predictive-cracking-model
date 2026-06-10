"""
从既有 true_pred CSV 导出 Origin 散点图用 Excel（不改动模型/训练/预测逻辑）。

输入：
  - outputs/evaluation/crack_width_true_pred.csv
  - outputs/evaluation/crack_density_true_pred.csv
  - outputs/lab_strength/lab_strength_compressive_true_pred.csv
  - outputs/lab_strength/lab_strength_flexural_true_pred.csv

输出：
  - outputs/evaluation/origin_plot_data.xlsx
  - outputs/evaluation/origin_plot_readme.md

用法：
  py scripts/export_origin_plot_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
LAB_STRENGTH_DIR = PROJECT_ROOT / "outputs" / "lab_strength"

DEFAULT_OUT_XLSX = EVAL_DIR / "origin_plot_data.xlsx"
DEFAULT_OUT_README = EVAL_DIR / "origin_plot_readme.md"

SHEET_COMPRESSIVE = "抗压强度"
SHEET_FLEXURAL = "抗折强度"
SHEET_CRACK_WIDTH = "裂缝宽度"
SHEET_CRACK_DENSITY = "裂缝密度"

COL_MEASURED = "Measured"
COL_PREDICTED = "Predicted"
COL_Y_EQ_X = "y=x"


def _regression_origin_frame(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    measured = pd.to_numeric(y_true, errors="coerce")
    predicted = pd.to_numeric(y_pred, errors="coerce")
    mask = measured.notna() & predicted.notna()
    measured = measured[mask].astype(float).reset_index(drop=True)
    predicted = predicted[mask].astype(float).reset_index(drop=True)
    return pd.DataFrame(
        {
            COL_MEASURED: measured,
            COL_PREDICTED: predicted,
            COL_Y_EQ_X: measured,
        }
    )


def _load_crack_eval(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"{csv_path} 需包含列 y_true, y_pred")
    return _regression_origin_frame(df["y_true"], df["y_pred"])


def _load_lab_strength(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "实测值_MPa" not in df.columns or "预测值_MPa" not in df.columns:
        raise ValueError(f"{csv_path} 需包含列 实测值_MPa, 预测值_MPa")
    return _regression_origin_frame(df["实测值_MPa"], df["预测值_MPa"])


def export_origin_plot_data(
    *,
    eval_dir: Path = EVAL_DIR,
    lab_strength_dir: Path = LAB_STRENGTH_DIR,
    out_xlsx: Path = DEFAULT_OUT_XLSX,
    out_readme: Path = DEFAULT_OUT_README,
) -> dict[str, int]:
    sources = {
        SHEET_COMPRESSIVE: lab_strength_dir / "lab_strength_compressive_true_pred.csv",
        SHEET_FLEXURAL: lab_strength_dir / "lab_strength_flexural_true_pred.csv",
        SHEET_CRACK_WIDTH: eval_dir / "crack_width_true_pred.csv",
        SHEET_CRACK_DENSITY: eval_dir / "crack_density_true_pred.csv",
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入文件：{path}")

    sheets: dict[str, pd.DataFrame] = {}
    row_counts: dict[str, int] = {}

    sheets[SHEET_COMPRESSIVE] = _load_lab_strength(sources[SHEET_COMPRESSIVE])
    sheets[SHEET_FLEXURAL] = _load_lab_strength(sources[SHEET_FLEXURAL])
    sheets[SHEET_CRACK_WIDTH] = _load_crack_eval(sources[SHEET_CRACK_WIDTH])
    sheets[SHEET_CRACK_DENSITY] = _load_crack_eval(sources[SHEET_CRACK_DENSITY])

    for name, frame in sheets.items():
        row_counts[name] = len(frame)

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for name in (
            SHEET_COMPRESSIVE,
            SHEET_FLEXURAL,
            SHEET_CRACK_WIDTH,
            SHEET_CRACK_DENSITY,
        ):
            sheets[name].to_excel(writer, sheet_name=name, index=False)

    readme = _build_readme(
        out_xlsx=out_xlsx,
        sources=sources,
        row_counts=row_counts,
    )
    out_readme.write_text(readme, encoding="utf-8")
    return row_counts


def _build_readme(
    *,
    out_xlsx: Path,
    sources: dict[str, Path],
    row_counts: dict[str, int],
) -> str:
    rel_xlsx = out_xlsx.relative_to(PROJECT_ROOT) if out_xlsx.is_relative_to(PROJECT_ROOT) else out_xlsx

    def _rel(p: Path) -> str:
        return str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)

    lines = [
        "# Origin 散点图数据说明",
        "",
        f"数据文件：`{rel_xlsx}`",
        "",
        "本表仅整理既有离线评估 / OOF 导出结果，**未重新训练或推理**。"
        "",
        "",
        "## 列定义（每个 Sheet 相同）",
        "",
        "| 列 | 字母 | 含义 | Origin 用法 |",
        "|----|------|------|-------------|",
        f"| {COL_MEASURED} | A | 实测值 | 横轴（X） |",
        f"| {COL_PREDICTED} | B | 预测值 | 纵轴（Y） |",
        f"| {COL_Y_EQ_X} | C | y=x 参考线 | 与 A 同值；作第二组 Y 或 Line 图可画 1:1 线 |",
        "",
        "## Sheet 与论文图对应关系",
        "",
        "| Sheet | 论文章节 | 对应论文图 | 说明 |",
        "|-------|----------|------------|------|",
        f"| {SHEET_COMPRESSIVE} | 7.3.1 / 7.3.3 | **图7-5 (a)** 抗压强度预测值与实测值对比 | 双子图左 panel |",
        f"| {SHEET_FLEXURAL} | 7.3.1 / 7.3.3 | **图7-5 (b)** 抗折强度预测值与实测值对比 | 双子图右 panel |",
        f"| {SHEET_CRACK_WIDTH} | 7.3.2 / 7.3.3 | **图7-3**（建议）裂缝宽度预测值与实测值对比 | hold-out 测试集；单位 mm |",
        f"| {SHEET_CRACK_DENSITY} | 7.3.2 / 7.3.3 | **图7-4**（建议）裂缝密度预测值与实测值对比 | hold-out 测试集；单位 条/m² |",
        "",
        "> 图7-3、图7-4 编号若与定稿不一致，请在 Word 中统一调整后，Origin 仍按本表 A/B 列作散点即可。",
        "",
        "## 数据来源",
        "",
        "| Sheet | 源 CSV | 行数 |",
        "|-------|--------|------|",
    ]
    for sheet in (
        SHEET_COMPRESSIVE,
        SHEET_FLEXURAL,
        SHEET_CRACK_WIDTH,
        SHEET_CRACK_DENSITY,
    ):
        lines.append(f"| {sheet} | `{_rel(sources[sheet])}` | {row_counts[sheet]} |")

    lines.extend(
        [
            "",
            "## Origin 导入提示",
            "",
            "1. `File` → `Import` → `Single ASCII` / Excel，选中对应 Sheet。",
            "2. 散点：X = **Measured**，Y = **Predicted**。",
            "3. y=x：将 **Measured** 作 X，**y=x** 作 Y，线型设为虚线；或与散点共用 X=Measured、Y=y=x。",
            "4. 力学强度子图单位：MPa；裂缝宽度：mm；裂缝密度：条/m²。",
            "",
            "## 未纳入本表的文件",
            "",
            "- `cracking_risk_true_pred.csv`：分类任务，不适合 Measured/Predicted 回归散点，请用混淆矩阵等图。",
            "",
            "## 重新生成",
            "",
            "```powershell",
            "cd d:\\cursor\\code\\SteelFiberCrackPredictor",
            "py -3 scripts/export_origin_plot_data.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="导出 Origin 用 true_pred Excel")
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--lab-strength-dir", type=Path, default=LAB_STRENGTH_DIR)
    ap.add_argument("--out-xlsx", type=Path, default=DEFAULT_OUT_XLSX)
    ap.add_argument("--out-readme", type=Path, default=DEFAULT_OUT_README)
    args = ap.parse_args()

    counts = export_origin_plot_data(
        eval_dir=args.eval_dir,
        lab_strength_dir=args.lab_strength_dir,
        out_xlsx=args.out_xlsx,
        out_readme=args.out_readme,
    )
    print(f"已写入：{args.out_xlsx.resolve()}")
    print(f"说明文档：{args.out_readme.resolve()}")
    for sheet, n in counts.items():
        print(f"  {sheet}: {n} 行")


if __name__ == "__main__":
    main()
