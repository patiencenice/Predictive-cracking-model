from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.features import (
    CONCRETE_TYPE_MAP,
    SAND_TYPE_MAP,
    SLAG_GRADE_MAP,
    STRENGTH_GRADE_ORDER,
    user_inputs_to_feature_frame,
)
from src.input_defaults import (
    PREDICTION_INPUT_DEFAULTS,
    normalize_prediction_inputs,
    validate_choice_fields,
)
from src.lab_strength_residual.lab_mix_features import (
    WATER_REDUCER_TYPE_UI_OPTIONS,
    lab_mix_extra_dict_from_user_inputs,
)
from src.environment_inputs import render_environment_sidebar_inputs
from src.fiber_inputs import render_fiber_sidebar_inputs
from src.thermal_stress_inputs import render_thermal_stress_sidebar_inputs


def render_lab_water_reducer_sidebar_inputs(data: dict) -> None:
    """
    侧栏：减水剂类型 + 减水率（%）+「减水率未知」。
    写入 data：water_reducer_type, water_reduction_rate_unknown, water_reduction_rate_pct（展示用；未知时后端以 flag 为准写 -1）。
    """
    adm = data.get("admixture", "无")
    if adm != "减水剂":
        st.caption(
            "外加剂非「减水剂」：减水剂类型按「无」闭合，减水率按 0；"
            "与 lab_strength 六列语义一致（enc=4，adjusted_w_b_ratio 等同水胶比）。"
        )
        data["water_reducer_type"] = "无"
        data["water_reduction_rate_unknown"] = False
        data["water_reduction_rate_pct"] = 0.0
        return

    st.markdown("###### 减水剂类型与减水率（lab_strength 派生）")
    opts = list(WATER_REDUCER_TYPE_UI_OPTIONS)
    cur = str(data.get("water_reducer_type", "无"))
    idx = opts.index(cur) if cur in opts else 0
    data["water_reducer_type"] = st.selectbox(
        "减水剂类型",
        opts,
        index=idx,
        key="sfc_water_reducer_type",
    )

    if data["water_reducer_type"] == "无":
        st.caption("类型为「无」：减水率固定为 0，不参与折算争议；减水率输入已省略。")
        data["water_reduction_rate_unknown"] = False
        data["water_reduction_rate_pct"] = 0.0
        return

    data["water_reduction_rate_unknown"] = st.checkbox(
        "减水率未知",
        value=bool(data.get("water_reduction_rate_unknown")),
        key="sfc_water_reduction_rate_unknown",
        help="勾选后：water_reduction_rate_pct=-1、water_reduction_rate_missing_flag=1，"
        "adjusted_w_b_ratio=-1（不将未知当作 0）。",
    )

    if data["water_reduction_rate_unknown"]:
        st.caption(
            "已选「减水率未知」：后端按缺失语义处理；**本页抗压/抗折公式基线仍不读取**该派生水胶比。"
        )
        data["water_reduction_rate_pct"] = 0.0
        return

    dv = float(PREDICTION_INPUT_DEFAULTS.get("water_reduction_rate_pct", 20.0))
    try:
        raw = float(data.get("water_reduction_rate_pct", dv))
        dv = max(0.0, min(80.0, raw))
    except (TypeError, ValueError):
        dv = 20.0
    data["water_reduction_rate_pct"] = st.number_input(
        "减水率（%）",
        min_value=0.0,
        max_value=80.0,
        value=float(dv),
        step=0.5,
        key="sfc_water_reduction_rate_pct",
    )


def _widget_default_for_field(key: str, meta: tuple) -> Any:
    """与 PREDICTION_INPUT_DEFAULTS 对齐的初始值，并限制在控件 min/max 内。"""
    if key not in PREDICTION_INPUT_DEFAULTS:
        if meta[-1] == "categorical":
            return meta[1][0]
        min_v, max_v, dtype = meta[1], meta[2], meta[3]
        if dtype is int:
            return int((int(min_v) + int(max_v)) // 2)
        return float((float(min_v) + float(max_v)) / 2.0)
    raw = PREDICTION_INPUT_DEFAULTS[key]
    if meta[-1] == "categorical":
        opts: list = meta[1]
        return raw if raw in opts else opts[0]
    min_v, max_v, dtype = meta[1], meta[2], meta[3]
    if dtype is int:
        v = int(raw)
        return max(int(min_v), min(int(max_v), v))
    v = float(raw)
    return max(float(min_v), min(float(max_v), v))


INPUT_CATEGORIES = {
    "纤维参数": {
        "fiber_engineering_bundle": (
            "工程抗裂纤维体系",
            None,
            None,
            None,
            "compound_fiber",
        ),
    },
    "混凝土配合比": {
        "strength_grade": (
            "强度等级（立方体抗压强度标准值 fcu,k）",
            list(STRENGTH_GRADE_ORDER),
            "categorical",
        ),
        "concrete_type": (
            "混凝土类型",
            list(CONCRETE_TYPE_MAP.keys()),
            "categorical",
        ),
        "binder_content": ("胶材用量（kg/m³）", 280, 650, float),
        "cement_content": ("水泥用量（kg/m³）", 300, 550, float),
        "fly_ash": ("粉煤灰用量（kg/m³）", 0, 180, float),
        "slag_grade": (
            "矿粉等级（活性指数）",
            list(SLAG_GRADE_MAP.keys()),
            "categorical",
        ),
        "slag_powder": ("矿粉用量（kg/m³）", 0, 180, float),
        "mixing_water": ("用水量（kg/m³）", 120, 240, float),
        "w_b_ratio": ("水胶比", 0.3, 0.5, float),
        "sand_type": (
            "砂（细骨料类型）",
            list(SAND_TYPE_MAP.keys()),
            "categorical",
        ),
        "sand_content": ("砂用量（细骨料，kg/m³）", 400, 950, float),
        "sand_ratio": ("砂率（%）", 30, 55, float),
        "stone_content": ("石用量（粗骨料，kg/m³）", 900, 1350, float),
        "admixture": ("外加剂类型", ["无", "减水剂", "膨胀剂"], "categorical"),
        "admixture_dosage": ("外加剂用量（kg/m³）", 0, 15, float),
        "water_reducer_type": (
            "减水剂类型 / 减水率（lab_strength 派生列）",
            None,
            None,
            None,
            "compound_lab_wr",
        ),
    },
    "环境与工艺": {
        "environment_engineering_bundle": (
            "工程开裂环境场 + 施工工艺场",
            None,
            None,
            None,
            "compound_environment",
        ),
    },
    "温度应力解释（Phase 1，可选）": {
        "thermal_phase1_bundle": (
            "温度应力解释（Phase 1）可选输入",
            None,
            None,
            None,
            "compound_thermal_p1",
        ),
    },
}


class InputSchema:
    def __init__(self, categories):
        self.categories = categories

    def render(self):
        """在侧边栏渲染所有输入组件，返回 dict 形式的用户输入。"""
        data = {}
        _expanded_default = frozenset({"纤维参数"})
        for group_name, fields in self.categories.items():
            with st.expander(
                group_name,
                expanded=group_name in _expanded_default,
            ):
                for key, meta in fields.items():
                    label = meta[0]
                    wkey = f"sfc_{key}"
                    if len(meta) >= 5 and meta[-1] == "compound_fiber":
                        render_fiber_sidebar_inputs(data)
                        continue
                    if len(meta) >= 5 and meta[-1] == "compound_environment":
                        render_environment_sidebar_inputs(data)
                        continue
                    if len(meta) >= 5 and meta[-1] == "compound_lab_wr":
                        render_lab_water_reducer_sidebar_inputs(data)
                        continue
                    if len(meta) >= 5 and meta[-1] == "compound_thermal_p1":
                        render_thermal_stress_sidebar_inputs(data)
                        continue
                    if meta[-1] == "categorical":
                        options = meta[1]
                        dv = _widget_default_for_field(key, meta)
                        idx = options.index(dv) if dv in options else 0
                        data[key] = st.selectbox(
                            label, options, index=idx, key=wkey
                        )
                    else:
                        min_v, max_v, dtype = meta[1], meta[2], meta[3]
                        dv = _widget_default_for_field(key, meta)
                        if dtype is int:
                            data[key] = st.number_input(
                                label,
                                min_value=int(min_v),
                                max_value=int(max_v),
                                value=int(dv),
                                step=1,
                                key=wkey,
                            )
                        else:
                            data[key] = st.number_input(
                                label,
                                min_value=float(min_v),
                                max_value=float(max_v),
                                value=float(dv),
                                key=wkey,
                            )
        return data


def build_input_schema():
    return InputSchema(INPUT_CATEGORIES)


def _cube_mpa_from_grade(grade: str) -> int:
    return int(grade[1:])


def collect_validation_warnings(user_inputs: dict) -> list[str]:
    warnings: list[str] = []
    mat = user_inputs.get("fiber_material", "钢纤维")
    fc = user_inputs["fiber_content"]
    w_b = user_inputs["w_b_ratio"]
    grade = user_inputs.get("strength_grade", "C30")
    ctype = user_inputs.get("concrete_type", "普通混凝土")
    fcu = _cube_mpa_from_grade(grade)

    if fc > 2.5 and w_b > 0.45:
        warnings.append(
            "纤维掺量与水胶比同时偏高，可能导致和易性严重下降，请谨慎使用。"
        )

    if mat == "聚丙烯纤维" and fc > 1.8:
        warnings.append(
            "聚丙烯纤维体积掺量过高时易结团、分散变差，工程中常用掺量往往低于钢纤维，请结合配合比试验核实。"
        )
    if mat == "玻璃纤维":
        warnings.append(
            "玻璃纤维在碱性孔溶液中可能发生长期强度退化；除模型参数外，宜采用耐碱玻璃纤维或经表面处理的制品，并满足相关规范。"
        )
    if mat == "玄武岩纤维" and user_inputs["humidity"] < 45:
        warnings.append(
            "玄武岩纤维与浆体界面粘结对早期失水较敏感，低湿度环境下建议加强覆盖保湿养护。"
        )

    wind = float(user_inputs.get("wind_speed_ms") or 0.0)
    if wind >= 6.0 and user_inputs.get("humidity", 100) < 55:
        warnings.append(
            "高风速叠加偏低湿度，塑性期表面蒸发加快，宜加强防风保湿与覆盖养护。"
        )
    if user_inputs.get("solar_exposure_level") == "high" and float(
        user_inputs.get("day_night_temp_diff_c") or 0.0
    ) >= 10.0:
        warnings.append(
            "强暴晒与较大昼夜温差并存，表层温度梯度与热裂缝风险升高，宜结合温控措施复核。"
        )

    if ctype == "高强混凝土" and fcu < 55:
        warnings.append(
            "已选「高强混凝土」但强度等级低于 C55，与常见工程划分不一致，请核对设计等级与类型。"
        )
    if ctype == "普通混凝土" and fcu >= 60:
        warnings.append(
            "强度等级已达 C60 及以上，通常归入高强混凝土范畴，建议将「混凝土类型」改为高强或核对配合比资料。"
        )
    if ctype == "自密实混凝土(SCC)" and w_b > 0.42:
        warnings.append(
            "自密实混凝土常要求较低水胶比与良好浆体黏度，当前水胶比偏高，可能影响填充性与抗裂，请结合试验确认。"
        )
    if ctype == "轻骨料混凝土" and user_inputs["sand_ratio"] > 48:
        warnings.append(
            "轻骨料混凝土砂率往往需结合轻骨料吸水与粒形单独设计，当前砂率偏高时请留意浆体包裹与工作度。"
        )
    if fcu >= 60 and w_b > 0.38:
        warnings.append(
            "高强度等级下若水胶比仍偏高，实际强度与耐久可能达不到预期，抗裂模型结果仅供参考，应以配合比试验为准。"
        )

    binder = user_inputs.get("binder_content", 0.0)
    fa = user_inputs.get("fly_ash", 0.0)
    slg = user_inputs.get("slag_powder", 0.0)
    cem = user_inputs.get("cement_content", 0.0)
    if fa + slg > binder + 1e-6:
        warnings.append(
            "粉煤灰与矿粉用量之和大于胶材用量，请核对胶材总量是否为水泥+掺合料之和。"
        )
    if cem > binder + 1e-6:
        warnings.append(
            "水泥用量大于胶材用量不合理，请核对「胶材用量」是否包含水泥及矿物掺合料。"
        )

    mw = float(user_inputs.get("mixing_water", 0.0))
    wb = float(user_inputs.get("w_b_ratio", 0.0))
    if binder > 1e-6 and wb > 1e-6:
        approx = wb * binder
        if abs(mw - approx) > max(25.0, 0.12 * approx):
            warnings.append(
                "用水量与水胶比×胶材用量偏差较大，请核对拌合水计量或水胶比定义是否与胶材总量一致。"
            )

    if user_inputs.get("admixture") == "无" and float(
        user_inputs.get("admixture_dosage", 0.0)
    ) > 0.05:
        warnings.append(
            "外加剂类型为「无」但外加剂用量大于 0，请核对是否应改为减水剂/膨胀剂或将用量置 0。"
        )

    return warnings


def validate_and_transform(user_inputs: dict, emit_streamlit_warnings: bool = True):
    """
    逻辑检查与特征编码，与训练特征列对齐。
    会先合并默认值（兼容 API 缺字段），再校验选项合法性。
    返回: (is_valid, X_dataframe, message, extra_info, warnings)
    """
    merged = normalize_prediction_inputs(user_inputs)
    ok, msg = validate_choice_fields(merged)
    if not ok:
        return False, pd.DataFrame(), msg, {}, []

    if merged.get("admixture") == "减水剂":
        wt = merged.get("water_reducer_type", "无")
        if wt != "无" and not bool(merged.get("water_reduction_rate_unknown")):
            try:
                r = float(merged.get("water_reduction_rate_pct", 0.0))
            except (TypeError, ValueError):
                return False, pd.DataFrame(), "减水率（%）无法解析为数值。", {}, []
            if r < 0.0 or r > 80.0:
                return (
                    False,
                    pd.DataFrame(),
                    "减水率（%）须在 0～80 之间，或勾选「减水率未知」。",
                    {},
                    [],
                )

    warnings = collect_validation_warnings(merged)
    if emit_streamlit_warnings:
        for w in warnings:
            st.warning(w)

    try:
        X = user_inputs_to_feature_frame(merged)
    except (KeyError, TypeError, ValueError) as e:
        err = f"参数无法转为模型输入: {e}"
        return False, pd.DataFrame(), err, {}, warnings

    extra_info: dict = {
        "fiber_material": merged["fiber_material"],
        "strength_grade": merged["strength_grade"],
        "concrete_type": merged["concrete_type"],
        "lab_mix_extra": lab_mix_extra_dict_from_user_inputs(merged),
    }
    return True, X, "OK", extra_info, warnings
