"""文献抽取：仅从「已整理的结构化表」或「半结构化文本」生成候选行，不直接把 PDF 正文当训练特征。

本模块输出 raw_extracted_rows.csv，供后续 schema_mapper 映射到统一字段。
PDF 二进制解析依赖 heavy 库且不稳定，故推荐：人工从论文表格导出 CSV 后作为本模块输入。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def extract_from_structured_table(path: Path) -> pd.DataFrame:
    """
    从已导出的 CSV/TSV 读取「宽表」候选行（列名可为任意语言/缩写）。
    这是推荐路径：作者从 PDF 表格复制到 Excel 后另存为 CSV。
    """
    suf = path.suffix.lower()
    if suf in (".csv",):
        return pd.read_csv(path)
    if suf in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"不支持的扩展名: {suf}")


def extract_from_tagged_text(path: Path) -> pd.DataFrame:
    """
    从简易标记文本抽取键值对行（演示用）：每行形如 key=value 或 key: value。
    连续空行分隔多条「逻辑样本」；块内必须含 source_doi=...
    不适合复杂论文，仅作小规模录入辅助。
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n+", text.strip())
    rows: list[dict[str, Any]] = []
    for block in blocks:
        if not block.strip():
            continue
        row: dict[str, Any] = {}
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^\s*([^:=\s]+)\s*[:=]\s*(.+)\s*$", line)
            if m:
                k, v = m.group(1).strip(), m.group(2).strip()
                row[k] = v
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def run_extraction(input_path: Path, mode: str = "auto") -> pd.DataFrame:
    """
    mode: auto | table | text
    - table: 强制按 CSV/TSV 读
    - text: 强制按键值文本读
    - auto: 按后缀 .txt 且非 csv 时用 text，否则 table
    """
    if mode == "table":
        return extract_from_structured_table(input_path)
    if mode == "text":
        return extract_from_tagged_text(input_path)
    if input_path.suffix.lower() in (".txt",) and mode == "auto":
        return extract_from_tagged_text(input_path)
    return extract_from_structured_table(input_path)


def save_raw_extracted(df: pd.DataFrame, out_path: Path) -> Path:
    """写出中间表 raw_extracted_rows.csv。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path
