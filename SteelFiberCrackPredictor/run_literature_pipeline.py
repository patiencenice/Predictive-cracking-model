"""
一键运行：文献结构化抽取 → 映射 → 单位 → 标签口径 → 质量分 → 合并用户数据 → 写出规定产物。

不将 PDF 正文作为模型输入；请先将论文表格导出为 CSV 再作为 --raw-input。

示例:
  py run_literature_pipeline.py
  py run_literature_pipeline.py --raw-input data/literature/example_raw_extracted.csv --user-csv data/training_data.csv
  py run_literature_pipeline.py --crack-width-family CW_MAX_SURFACE_MM
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.literature_pipeline.label_standardizer import (
    assign_default_definition_ids,
    validate_single_definition_family,
    write_label_definition_md,
)
from src.literature_pipeline.literature_extractor import run_extraction, save_raw_extracted
from src.literature_pipeline.merge_datasets import merge_user_and_literature, save_merged
from src.literature_pipeline.crack_width_definition_filter import (
    definition_id_counts,
    emit_mixed_warning_if_needed,
    filter_by_crack_width_family,
    known_family_help,
    mixed_definition_warning_text,
)
from src.literature_pipeline.quality_scoring import (
    add_quality_columns,
    build_data_quality_report,
)
from src.literature_pipeline.schema_mapper import (
    build_model_frame,
    drop_rows_missing_critical_labels,
)
from src.literature_pipeline.source_registry import (
    enrich_row_level_notes,
    save_source_provenance_csv,
)
from src.literature_pipeline.unit_normalizer import normalize_all
from src.paths import OUTPUTS_DIR, PROJECT_ROOT
from annotate_user_training_data import (
    annotate_user_training_data_file,
    annotate_user_training_data_df,
    _parse_group_cols,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="文献→结构化训练集管线")
    ap.add_argument(
        "--raw-input",
        type=Path,
        default=PROJECT_ROOT / "data" / "literature" / "example_raw_extracted.csv",
        help="已整理的宽表 CSV（从论文表格导出）",
    )
    ap.add_argument(
        "--user-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "training_data.csv",
        help="用户试验数据（与 train_model 格式兼容）",
    )
    ap.add_argument(
        "--extract-mode",
        choices=("auto", "table", "text"),
        default="auto",
        help="抽取模式：table=CSV，text=键值文本",
    )
    ap.add_argument(
        "--crack-width-family",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "仅保留该 crack_width_definition_id（如 CW_MAX_SURFACE_MM）。"
            f"已知 ID：{known_family_help()}"
        ),
    )
    ap.add_argument(
        "--auto-annotate-user",
        action="store_true",
        help="开启用户表自动补全（默认关闭）",
    )
    ap.add_argument(
        "--annotated-user-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "training_data.annotated.csv",
        help="自动补全后的用户表输出路径（可覆盖已存在文件）",
    )
    ap.add_argument(
        "--default-crack-width-family",
        type=str,
        default="CW_MAX_SURFACE_MM",
        help="自动补全时 crack_width_definition_id 默认值",
    )
    ap.add_argument(
        "--default-source-doi",
        type=str,
        default="USER_LOCAL_LAB",
        help="自动补全时 source_doi 默认值",
    )
    ap.add_argument(
        "--source-group-cols",
        type=str,
        default="",
        help="自动补全时按这些列拼接 source_group（逗号分隔，如 batch,series,date）",
    )
    ap.add_argument(
        "--dry-run-auto-annotate",
        action="store_true",
        help="仅预览用户表自动补全结果，不落地 annotated 文件且不继续后续管线",
    )
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # dry-run 优先：只做用户表自动补全预览，不执行后续文献抽取/合并等流程
    if args.dry_run_auto_annotate:
        # 要求显式提供 --user-csv，避免误用默认路径
        if "--user-csv" not in sys.argv:
            raise SystemExit(
                "开启 --dry-run-auto-annotate 时，必须显式提供 --user-csv 路径。"
            )
        if not args.user_csv.exists():
            raise SystemExit(
                f"开启 --dry-run-auto-annotate 但 user-csv 不存在: {args.user_csv}"
            )
        raw_user = pd.read_csv(args.user_csv)
        annotated_df, ann_summary = annotate_user_training_data_df(
            raw_user,
            default_crack_width_family=args.default_crack_width_family,
            default_source_doi=args.default_source_doi,
            source_group_cols=_parse_group_cols(args.source_group_cols),
        )
        print("=== dry-run auto annotate preview ===")
        print(f"输入 user-csv 路径: {args.user_csv}")
        print(
            "是否启用默认 crack_width_definition_id: "
            + str(ann_summary.get("used_default_crack_width_family"))
        )
        print(
            "是否启用默认 source_doi: "
            + str(ann_summary.get("used_default_source_doi"))
        )
        print("source_group 生成策略: " + str(ann_summary.get("source_group_mode")))
        print("总行数: " + str(ann_summary.get("n_rows")))
        print("新增列: " + str(ann_summary.get("added_columns")))
        print(
            "crack_width_definition_id 分布: "
            + str(ann_summary.get("crack_width_definition_id_counts"))
        )
        print("source_doi Top10 分布: " + str(ann_summary.get("source_doi_counts_top10")))
        print("source_group 分组数: " + str(ann_summary.get("source_group_nunique")))
        print("source_group Top20 分布: " + str(ann_summary.get("source_group_counts_top20")))
        print(
            "是否存在大量 DEFAULT 分组: "
            + str(ann_summary.get("has_many_default_groups"))
            + f" (label={ann_summary.get('default_group_label')}, "
            + f"count={ann_summary.get('default_group_count')}, "
            + f"ratio={ann_summary.get('default_group_ratio'):.3f})"
        )

        # 建议输出：预览 JSON + 前 50 行 CSV（非 annotated 正式文件）
        preview_json = OUTPUTS_DIR / "auto_annotate_preview.json"
        preview_csv = OUTPUTS_DIR / "auto_annotate_preview.csv"
        with open(preview_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input_user_csv": str(args.user_csv.resolve()),
                    "dry_run_auto_annotate": True,
                    "annotate_user_summary": ann_summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        annotated_df.head(50).to_csv(preview_csv, index=False, encoding="utf-8-sig")
        print(f"已写预览 JSON: {preview_json}")
        print(f"已写预览 CSV(前50行): {preview_csv}")
        return

    # 1) 抽取中间表
    raw = run_extraction(args.raw_input, mode=args.extract_mode)
    save_raw_extracted(raw, OUTPUTS_DIR / "raw_extracted_rows.csv")

    # 2) 映射 + 单位 + 标签默认 ID
    df = build_model_frame(raw)
    df = normalize_all(df)
    df = assign_default_definition_ids(df)
    df = enrich_row_level_notes(df)

    # 2b) source_group：空值用 source_doi 回填（与 merge 中文献逻辑一致；source_doi 溯源不变）
    if "source_group" not in df.columns:
        df["source_group"] = df["source_doi"].astype(str).str.strip()
    else:
        sg = df["source_group"].astype(str).str.strip()
        sd = df["source_doi"].astype(str).str.strip()
        empty = sg.isna() | (sg == "") | (sg == "nan")
        df.loc[empty, "source_group"] = sd[empty]

    # 3) 剔除标签不清/缺失行
    df, drop_reasons = drop_rows_missing_critical_labels(df)
    warns = validate_single_definition_family(df)
    drop_reasons.extend(warns)

    # 3b) 按 crack_width 测量口径过滤；未指定且多口径并存时强警告
    counts_pre = definition_id_counts(df)
    if not args.crack_width_family:
        emit_mixed_warning_if_needed(counts_pre)
    df, def_stats = filter_by_crack_width_family(df, args.crack_width_family)
    print(
        f"[crack_width_definition_id] 过滤前样本数: {def_stats['n_before']} -> "
        f"过滤后: {def_stats['n_after']}（剔除 {def_stats['dropped']} 行）"
    )
    if len(df) == 0:
        raise SystemExit(
            "过滤后无剩余样本：请检查 --crack-width-family 是否与文献表中 "
            "crack_width_definition_id 一致，或放宽过滤条件。"
        )
    mix_warn = (
        mixed_definition_warning_text(counts_pre)
        if not args.crack_width_family
        else None
    )

    # 4) 质量分
    df = add_quality_columns(df)

    # 5) 文献训练集（须含 FEATURE_COLUMNS + 目标 + 溯源）
    lit_path = OUTPUTS_DIR / "literature_training_data.csv"
    df.to_csv(lit_path, index=False, encoding="utf-8-sig")

    # 6) 溯源聚合
    save_source_provenance_csv(df, OUTPUTS_DIR / "source_provenance.csv")

    # 7) label_definition.md
    write_label_definition_md(OUTPUTS_DIR / "label_definition.md")

    # 8) 质量报告（含各 definition_id 计数、过滤剔除数、混训警告）
    dq = build_data_quality_report(
        df,
        drop_reasons,
        definition_filter_stats=def_stats,
        mixed_definition_warning=mix_warn,
    )

    # 9) 合并用户数据（可选先自动补全；不覆盖原始 user-csv）
    user_path = args.user_csv
    annotate_summary: dict | None = None
    user_path_for_merge = user_path

    print(f"[auto_annotate_user] 启用状态: {args.auto_annotate_user}")
    print(f"[auto_annotate_user] 原始用户表路径: {user_path}")
    print(
        "[auto_annotate_user] 补全后用户表路径: "
        + (
            str(args.annotated_user_output)
            if args.auto_annotate_user
            else "(未启用自动补全)"
        )
    )

    if args.auto_annotate_user:
        if not user_path.exists():
            raise SystemExit(
                "启用了 --auto-annotate-user，但 --user-csv 不存在。"
                "请提供有效用户表路径后重试。"
            )
        if args.annotated_user_output.exists():
            print(
                f"[auto_annotate_user] 提示：补全输出已存在，将覆盖: {args.annotated_user_output}"
            )
        annotate_summary = annotate_user_training_data_file(
            input_path=user_path,
            output_path=args.annotated_user_output,
            default_crack_width_family=args.default_crack_width_family,
            default_source_doi=args.default_source_doi,
            source_group_cols=_parse_group_cols(args.source_group_cols),
            allow_overwrite_output=True,
        )
        user_path_for_merge = args.annotated_user_output
        print(f"[auto_annotate_user] 补全后用户表路径: {user_path_for_merge}")
        print(
            "[auto_annotate_user] source_group 生成策略: "
            + str(annotate_summary.get("source_group_mode"))
        )
        print(
            "[auto_annotate_user] crack_width_definition_id 分布: "
            + str(annotate_summary.get("crack_width_definition_id_counts", {}))
        )
    else:
        print("[auto_annotate_user] 未启用，沿用原始 user-csv。")

    if user_path_for_merge.exists():
        merged = merge_user_and_literature(user_path_for_merge, lit_path)
    else:
        merged = pd.read_csv(lit_path)
        print(
            f"未找到用户数据 {user_path_for_merge}，training_data_merged.csv 仅含文献行。"
        )
    save_merged(merged, OUTPUTS_DIR / "training_data_merged.csv")
    print(f"已写出合并表: {OUTPUTS_DIR / 'training_data_merged.csv'}")

    # 写入 data_quality_report（附加自动补全记录）
    dq["auto_annotate_user_enabled"] = bool(args.auto_annotate_user)
    dq["annotated_user_csv_path"] = (
        str(user_path_for_merge.resolve())
        if args.auto_annotate_user and user_path_for_merge.exists()
        else None
    )
    dq["annotate_user_summary"] = annotate_summary
    with open(OUTPUTS_DIR / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(dq, f, indent=2, ensure_ascii=False, default=str)

    print(f"完成。文献表: {lit_path}")
    print(f"中间表: {OUTPUTS_DIR / 'raw_extracted_rows.csv'}")


if __name__ == "__main__":
    main()
