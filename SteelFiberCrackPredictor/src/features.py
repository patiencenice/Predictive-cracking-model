"""与训练/推理一致的特征列顺序与类别编码。"""

from __future__ import annotations

import pandas as pd

# 几何外形（钢纤维常用；其它材质也可选相近外形作近似）
FIBER_TYPE_MAP = {"端钩型": 0, "波纹型": 1, "平直型": 2}

# 材质：刚度、粘结与阻裂机理不同，单独编码供模型学习
FIBER_MATERIAL_MAP = {
    "钢纤维": 0,
    "玄武岩纤维": 1,
    "聚丙烯纤维": 2,
    "玻璃纤维": 3,
}

ADMIXTURE_MAP = {"无": 0, "减水剂": 1, "膨胀剂": 2}
CASTING_METHOD_MAP = {"常规": 0, "泵送": 1}

# 立方体抗压强度标准值对应等级（GB 常用系列）
STRENGTH_GRADE_ORDER = [
    "C15",
    "C20",
    "C25",
    "C30",
    "C35",
    "C40",
    "C45",
    "C50",
    "C55",
    "C60",
    "C65",
    "C70",
    "C75",
    "C80",
]
STRENGTH_GRADE_TO_MPA = {g: int(g[1:]) for g in STRENGTH_GRADE_ORDER}
STRENGTH_GRADE_ENC = {g: i for i, g in enumerate(STRENGTH_GRADE_ORDER)}

# 混凝土类型（配合比与施工特征不同，抗裂表现有差异）
CONCRETE_TYPE_MAP = {
    "普通混凝土": 0,
    "高强混凝土": 1,
    "自密实混凝土(SCC)": 2,
    "轻骨料混凝土": 3,
}

# 粒化高炉矿渣粉活性等级（GB/T 18046 常用 S75 / S95 / S105）
SLAG_GRADE_MAP = {
    "S75": 0,
    "S95": 1,
    "S105": 2,
}

# 细骨料：天然砂 / 机制砂
SAND_TYPE_MAP = {
    "天然砂": 0,
    "机制砂": 1,
}

FEATURE_COLUMNS = [
    "fiber_content",
    "aspect_ratio",
    "tensile_strength",
    "fiber_type_enc",
    "fiber_material_enc",
    # 强度与混凝土类型（配合比区填写）
    "cube_strength_mpa",
    "strength_grade_enc",
    "concrete_type_enc",
    # 胶材—骨料—外加剂（与侧栏「混凝土配合比」顺序一致）
    "binder_content",
    "cement_content",
    "fly_ash",
    "slag_grade_enc",
    "slag_powder",
    "mixing_water",
    "w_b_ratio",
    "sand_type_enc",
    "sand_content",
    "sand_ratio",
    "stone_content",
    "admixture_enc",
    "admixture_dosage",
    "curing_days",
    "temperature",
    "humidity",
    "casting_method_enc",
    "fiber_content_x_aspect_ratio",
]


def user_inputs_to_feature_frame(user_inputs: dict) -> pd.DataFrame:
    """将表单/API 原始字段转为模型输入 DataFrame（单行，列顺序固定）。"""
    row = {
        "fiber_content": float(user_inputs["fiber_content"]),
        "aspect_ratio": float(user_inputs["aspect_ratio"]),
        "tensile_strength": float(user_inputs["tensile_strength"]),
        "fiber_type_enc": int(FIBER_TYPE_MAP[user_inputs["fiber_type"]]),
        "fiber_material_enc": int(FIBER_MATERIAL_MAP[user_inputs["fiber_material"]]),
        "cube_strength_mpa": float(
            STRENGTH_GRADE_TO_MPA[user_inputs["strength_grade"]]
        ),
        "strength_grade_enc": int(STRENGTH_GRADE_ENC[user_inputs["strength_grade"]]),
        "concrete_type_enc": int(CONCRETE_TYPE_MAP[user_inputs["concrete_type"]]),
        "binder_content": float(user_inputs["binder_content"]),
        "cement_content": float(user_inputs["cement_content"]),
        "fly_ash": float(user_inputs["fly_ash"]),
        "slag_grade_enc": int(SLAG_GRADE_MAP[user_inputs["slag_grade"]]),
        "slag_powder": float(user_inputs["slag_powder"]),
        "mixing_water": float(user_inputs["mixing_water"]),
        "w_b_ratio": float(user_inputs["w_b_ratio"]),
        "sand_type_enc": int(SAND_TYPE_MAP[user_inputs["sand_type"]]),
        "sand_content": float(user_inputs["sand_content"]),
        "sand_ratio": float(user_inputs["sand_ratio"]),
        "stone_content": float(user_inputs["stone_content"]),
        "admixture_enc": int(ADMIXTURE_MAP[user_inputs["admixture"]]),
        "admixture_dosage": float(user_inputs["admixture_dosage"]),
        "curing_days": int(user_inputs["curing_days"]),
        "temperature": float(user_inputs["temperature"]),
        "humidity": float(user_inputs["humidity"]),
        "casting_method_enc": int(CASTING_METHOD_MAP[user_inputs["casting_method"]]),
        "fiber_content_x_aspect_ratio": float(user_inputs["fiber_content"])
        * float(user_inputs["aspect_ratio"]),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
