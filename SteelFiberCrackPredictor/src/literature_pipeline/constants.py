"""文献管线：统一列名、溯源字段与目标列定义（与推理端 FEATURE_COLUMNS 对齐）。"""

from __future__ import annotations

# 每条样本必须保留的溯源与试验元数据（不入模型特征，但入训练集 CSV 与报表）
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "source_doi",  # 文献 DOI 或内部编号；用户自有数据用固定占位
    # GroupKFold 优先按此列分组（可与 DOI 不同粒度）；无则回退 source_doi
    "source_group",
    "source_table",  # 表格编号，如 Table 3
    "source_figure",  # 图号，如 Fig.2；无则空
    "source_paper_title",  # 论文题目（可选）
    "test_method",  # 裂缝测量/加载试验方法简述
    "specimen_size",  # 试件尺寸，如 100mm 立方体+梁
)

# 标签口径标识：用于拒绝混用不同定义的 crack_width / crack_density
LABEL_META_COLUMNS: tuple[str, ...] = (
    "crack_width_definition_id",
    "crack_density_definition_id",
)


TARGET_COLUMNS: tuple[str, ...] = (
    "crack_width",
    "crack_density",
    "cracking_risk",
)

# 质量分与样本权重列（由 quality_scoring 写入）
QUALITY_COLUMNS: tuple[str, ...] = (
    "source_quality_score",
    "sample_weight",
)

# 用户合并占位 DOI：与文献区分；细分批次用 source_group，如 USER_LOCAL_LAB/BATCH_001
USER_SOURCE_DOI_PLACEHOLDER = "USER_LOCAL_LAB"

# 用户未填 source_group 时的默认子组（与自定义 BATCH_* 形式一致）
USER_SOURCE_GROUP_DEFAULT = "USER_LOCAL_LAB/DEFAULT"
