import sys
from pathlib import Path

# 无论从哪个工作目录启动 streamlit，都能导入 src 并定位资源目录
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import traceback

import streamlit as st

try:
    from streamlit.errors import StreamlitAPIException
except ImportError:
    StreamlitAPIException = Exception  # type: ignore[misc, assignment]

try:
    st.set_page_config(
        page_title="纤维混凝土抗裂性能预测系统",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except StreamlitAPIException:
    pass

from src.data_processor import build_input_schema, validate_and_transform
from src.features import FEATURE_COLUMNS
from src.paths import CONFIG_YAML, MODELS_DIR
from src.predictor import SteelFiberCrackPredictor
from src.ui_theme import (
    empty_state_hint_html,
    hero_banner_html,
    inject_streamlit_theme,
    sidebar_footer_html,
)
from src.visualizer import Visualizer


@st.cache_resource
def load_predictor(_feature_dim: int = len(FEATURE_COLUMNS)):
    """全局加载模型；特征列数量变化时自动失效缓存，避免维数不一致。"""
    return SteelFiberCrackPredictor(
        model_dir=str(MODELS_DIR),
        config_path=str(CONFIG_YAML) if CONFIG_YAML.exists() else None,
    )


def _show_section_error(section: str, exc: BaseException) -> None:
    """单块 UI 失败时提示，不拖垮整页（模型预测已完成）。"""
    st.error(
        f"「{section}」展示失败，其它 Tab 仍可查看。请向下看具体错误；"
        "若刚改过代码，关闭终端后重新执行 `py run_web.py` 即可。"
    )
    st.code(f"{type(exc).__name__}: {exc}", language="text")
    with st.expander("详细堆栈（便于排查）", expanded=False):
        st.code(traceback.format_exc(), language="text")


def _run_tab_section(section: str, fn) -> None:
    try:
        fn()
    except Exception as e:
        _show_section_error(section, e)


def _risk_level_from_result(pred_result: dict | None) -> str | None:
    if not pred_result:
        return None
    preds = pred_result.get("predictions", {}) or {}
    rd = preds.get("risk_dimension") or {}
    return str(rd.get("alert_level", preds.get("risk_level", "待评估")))


def render_main_tabs(pred_result, predictor, visualizer, user_inputs, X_df):
    (
        tab_overview,
        tab_mech_support,
        tab_mech_expl,
        tab_trust,
        tab_rd,
    ) = st.tabs(
        [
            "① 预测结果",
            "② 力学性能",
            "③ 开裂机理",
            "④ 模型可信度",
            "⑤ 高级分析",
        ]
    )

    with tab_overview:
        _run_tab_section(
            "预测结果",
            lambda: visualizer.show_tab_comprehensive_prediction(
                pred_result, user_inputs
            ),
        )

    with tab_mech_support:
        _run_tab_section(
            "力学性能",
            lambda: visualizer.show_tab_mechanical_support(user_inputs),
        )

    with tab_mech_expl:
        _run_tab_section(
            "开裂机理",
            lambda: visualizer.show_tab_crack_mechanism(
                predictor, X_df, pred_result, user_inputs
            ),
        )

    with tab_trust:
        _run_tab_section(
            "模型可信度",
            lambda: visualizer.show_tab_data_trust(user_inputs, pred_result),
        )

    with tab_rd:
        _run_tab_section(
            "高级分析",
            lambda: visualizer.show_tab_rd_diagnostics(
                pred_result, user_inputs, predictor, X_df
            ),
        )


def main():
    inject_streamlit_theme()
    _hero_subtitle = (
        "工程化开裂风险监测与配合比评估 · 主模型输出裂缝与风险概率 · "
        "力学与温度为辅助解释层，请结合试验与规范使用。"
    )

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error("模型加载失败。可尝试删除本目录下 `models` 文件夹内全部 `.pkl` 后刷新页面。")
        st.code(f"{type(e).__name__}: {e}", language="text")
        st.markdown(
            "安装依赖后，在 **SteelFiberCrackPredictor** 目录执行：  \n"
            "`py -m pip install -r requirements.txt`  \n"
            "`py -m streamlit run app.py`"
        )
        return

    visualizer = Visualizer()

    with st.sidebar:
        st.header("参数输入")
        st.caption("左侧分组与工程表单一致；调整后下方可实时预测。")
        schema = build_input_schema()
        user_inputs = schema.render()
        st.divider()
        auto_predict = st.checkbox("实时预测", value=True)
        predict_btn = st.button("开始预测", type="primary")
        st.markdown(sidebar_footer_html(), unsafe_allow_html=True)

    should_predict = auto_predict or predict_btn

    if not should_predict:
        st.markdown(
            hero_banner_html(
                "纤维混凝土抗裂性能预测系统",
                _hero_subtitle,
                risk_level="待预测",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            empty_state_hint_html(
                "请勾选左侧「实时预测」，或点击「开始预测」后，从「① 预测结果」查看主输出。"
            ),
            unsafe_allow_html=True,
        )
        return

    try:
        valid, X, msg, extra_info, _ = validate_and_transform(user_inputs)
        if not valid:
            st.error(msg)
            return

        with st.spinner("模型预测中，请稍候..."):
            pred_result = predictor.predict_all(X, extra_info=extra_info)

        st.markdown(
            hero_banner_html(
                "纤维混凝土抗裂性能预测系统",
                _hero_subtitle,
                risk_level=_risk_level_from_result(pred_result),
            ),
            unsafe_allow_html=True,
        )
        render_main_tabs(pred_result, predictor, visualizer, user_inputs, X)
    except Exception as e:
        st.error("预测或结果展示过程出错（见下方具体错误）。")
        st.markdown(
            "**说明：** 该提示里的「Clear cache」在 **浏览器里打开的 Streamlit 页面** 右上角 "
            "（⋮ 或 **Deploy** 旁菜单），**不在 Cursor 编辑器里**。若找不到该菜单，可改用：\n"
            "1. 浏览器 **Ctrl+F5** 强制刷新；\n"
            "2. 关闭运行 `py run_web.py` 的黑色终端窗口，在项目目录重新执行 `py run_web.py`；\n"
            "3. 把下方灰色框里的 **错误类型与文字** 发给我，便于精确定位。"
        )
        st.code(f"{type(e).__name__}: {e}", language="text")
        with st.expander("详细堆栈", expanded=False):
            st.code(traceback.format_exc(), language="text")


main()
