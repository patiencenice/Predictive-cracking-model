import os
from typing import Any, Callable, Dict, List

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS
from src.model_bootstrap import ensure_default_models


OptimizationCondition = Callable[[Dict[str, Any]], bool]


def _cube_strength_mpa_from_grade(grade: object) -> int | None:
    if not isinstance(grade, str) or len(grade) < 2 or not grade.upper().startswith("C"):
        return None
    try:
        return int(grade[1:])
    except ValueError:
        return None


def _rule_high_strength_crack(x: Dict[str, Any]) -> bool:
    m = _cube_strength_mpa_from_grade(x.get("strength_grade"))
    return (
        m is not None
        and m >= 55
        and (x.get("cracking_risk_score", 0) > 0.52)
    )


def _risk_probability_from_multiclass(proba: np.ndarray, classes: np.ndarray) -> float:
    """将三分类概率压缩为 [0,1] 连续开裂风险概率 P（低/中/高类分别赋权 0 / 0.5 / 1）。"""
    w_map = {0: 0.0, 1: 0.5, 2: 1.0}
    weights = np.array([w_map.get(int(c), 0.5) for c in classes], dtype=np.float64)
    return float(np.dot(proba, weights))


def _alert_level_from_p(p: float) -> str:
    """分级预警：低风险 P<0.3；中风险 0.3≤P≤0.7；高风险 P>0.7。"""
    if p < 0.3:
        return "低风险"
    if p <= 0.7:
        return "中风险"
    return "高风险"


def _stress_strength_ratio_heuristic(x: dict, risk_p: float) -> float:
    """收缩应力/抗拉强度比的启发式无量纲指标（演示用，非试验标定）。"""
    w_b = float(x.get("w_b_ratio", 0.4))
    binder = float(x.get("binder_content", 400.0))
    ft = max(float(x.get("tensile_strength", 1200.0)), 1.0)
    # 水胶比与胶材用量推高约束应力水平；纤维抗拉作归一化参照
    raw = 0.22 + 0.48 * risk_p + 0.12 * max(0.0, w_b - 0.32) * 6.0
    raw += 0.00015 * max(0.0, binder - 420.0) / 10.0
    raw += 0.08 * (1.0 - min(ft / 2000.0, 1.0))
    return float(np.clip(raw, 0.05, 1.35))


def _fallback_crack_density_per_m2(w_mm: float, risk_p: float) -> float:
    """无裂缝密度回归模型或预测无效时，由裂缝宽度与风险概率给出可展示的经验换算（条/m²）。"""
    w = float(w_mm) if np.isfinite(w_mm) else 0.15
    p = float(risk_p) if np.isfinite(risk_p) else 0.5
    p = float(np.clip(p, 0.0, 1.0))
    base = 0.65 + 3.8 * p + 1.6 * max(0.0, w - 0.12)
    return float(np.clip(base, 0.15, 12.0))


def _crack_width_gb50010_reference(w_mm: float) -> Dict[str, Any]:
    """
    信息性参照 GB 50010-2010（2015 年版）表 3.4.5：室内正常环境钢筋混凝土构件裂缝验算限值常用 0.30 mm。
    非本系统自动完成规范验算，实际以设计环境类别与条文为准。
    """
    w = float(w_mm) if np.isfinite(w_mm) else 0.0
    w_lim = 0.30
    if w <= w_lim + 1e-9:
        cmp_cn = (
            f"预测最大裂缝宽度 {w:.3f} mm，未超过室内正常环境下钢筋混凝土构件常见验算限值 "
            f"{w_lim:.2f} mm（信息性参照，见 GB 50010 表 3.4.5）。"
        )
    else:
        cmp_cn = (
            f"预测最大裂缝宽度 {w:.3f} mm，超过上述信息性参照限值 {w_lim:.2f} mm；"
            "工程应用须按实际环境类别、构件类型与荷载组合由设计验算。"
        )
    return {
        "w_lim_ref_mm": w_lim,
        "environment_scope_cn": "室内正常环境、钢筋混凝土构件（规范表 3.4.5 典型取值之一）",
        "comparison_cn": cmp_cn,
    }


def _standards_bundle() -> Dict[str, Any]:
    return {
        "referenced_specs": [
            "GB/T 50081-2019《普通混凝土力学性能试验方法标准》（立方体抗压强度 fcu，标准试件边长 150 mm）",
            "GB 50010-2010（2015 年版）《混凝土结构设计规范》（构件裂缝宽度验算限值见表 3.4.5）",
            "GB/T 50082-2009《普通混凝土长期性能和耐久性能试验方法标准》（收缩与抗裂性能试验）",
            "GB 50164-2011《混凝土质量控制标准》（浇筑、养护与质量控制）",
            "JGJ/T 221-2010《纤维混凝土应用技术规程》（钢纤维混凝土工程应用）",
        ],
        "disclaimer": (
            "本系统输出为基于参数与机器学习模型的抗裂性能辅助评估，不能替代结构设计、施工验收、试验检测及监理依据的完整合规判定。"
        ),
        "risk_tier_note": (
            "界面「低/中/高风险」分级按平台设定概率阈值 P 划分，不属于强制性国家标准条文本身。"
        ),
    }


def _derive_time_metrics(x: dict, risk_p: float) -> Dict[str, float]:
    """
    时间维度启发式（演示用）：与真实热–力耦合有限元或现场监测有差异，需结合试验校准。
    单位：开裂时间、安全窗口为小时；临界龄期为天。
    """
    w_b = float(x.get("w_b_ratio", 0.4))
    hum = float(x.get("humidity", 60.0))
    cube = float(x.get("cube_strength_mpa", 30.0))

    # 开裂时间（h）：风险高、湿度低、水胶比高则更早出现可见裂缝
    t_crack_h = 40.0 + 380.0 * risk_p - 0.35 * hum + 90.0 * max(0.0, w_b - 0.35)
    t_crack_h = float(np.clip(t_crack_h, 8.0, 720.0))

    # 临界龄期（d）：温度/收缩约束应力与早期抗拉强度发展的竞争
    t_crit_d = 1.2 + 5.5 * risk_p + 0.1 * max(0.0, 40.0 - cube)
    t_crit_d = float(np.clip(t_crit_d, 0.5, 28.0))

    # 安全窗口（h）：保守估计「不致因过早撤除养护而开裂」的可保留保湿时长
    t_safe_h = 20.0 + (1.0 - risk_p) * 100.0 + 0.45 * hum
    t_safe_h = float(np.clip(t_safe_h, 12.0, 336.0))

    return {
        "cracking_time_hours": t_crack_h,
        "critical_age_days": t_crit_d,
        "safety_window_hours": t_safe_h,
    }


def _time_dimension_note_gb() -> str:
    return (
        "时间为结合配合比与环境参数的估算量，宜与现场测温、应变监测及 GB/T 50082 等试验方法对照；"
        "保湿撤除时间应同时满足 GB 50164 对养护制度与结构受荷条件的要求。"
    )


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _prediction_based_recommendations(x: dict, result: dict) -> List[Dict[str, Any]]:
    """根据本次预测数值（时间/状态/风险维度）生成补充优化建议，与参数规则建议合并展示。"""
    preds = result.get("predictions", {}) or {}
    sd = preds.get("state_dimension") or {}
    td = preds.get("time_dimension") or {}
    rd = preds.get("risk_dimension") or {}

    p = _safe_float(sd.get("risk_probability", preds.get("risk_confidence")), 0.0)
    p = float(np.clip(p, 0.0, 1.0))
    w_mm = _safe_float(sd.get("crack_width_mm", preds.get("crack_width")), 0.0)
    dens = _safe_float(sd.get("crack_density_per_m2", preds.get("crack_density")), 0.0)
    ssr = _safe_float(sd.get("stress_strength_ratio"), 0.0)
    t_crack = _safe_float(td.get("cracking_time_hours"), 999.0)
    t_safe = _safe_float(td.get("safety_window_hours"), 999.0)
    t_crit = _safe_float(td.get("critical_age_days"), 99.0)
    alert = str(rd.get("alert_level", ""))

    w_b = _safe_float(x.get("w_b_ratio"), 0.4)
    hum = _safe_float(x.get("humidity"), 60.0)
    binder = _safe_float(x.get("binder_content"), 0.0)
    mid = result.get("intermediate") or {}
    fiber = str(mid.get("fiber_material", x.get("fiber_material", "")))

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def push(
        key: str,
        title: str,
        suggestion: str,
        expected: str = "",
        cost: str = "",
    ) -> None:
        if key in seen:
            return
        seen.add(key)
        out.append(
            {
                "title": title,
                "suggestion": suggestion,
                "expected_improvement": expected,
                "cost_impact": cost,
                "source": "prediction",
            }
        )

    # —— 与预警等级、P 直接挂钩 ——
    if p > 0.7:
        push(
            "p_high",
            "【预测】开裂风险概率偏高",
            (
                f"本次综合风险概率 P≈{p:.2f}（{alert}）。建议优先核查水胶比、胶材用量与砂率是否偏不利；"
                "加强浇筑后 72 h 内保湿防风与内外温差控制；在强度与工作性允许下可适当提高纤维体积掺量或选用端钩/波纹等桥接型外形。"
            ),
            "有望压低裂缝宽度与早期开裂概率（需配合试配与现场验证）。",
            "纤维与外加剂微调可能增加单方材料费；延长养护增加人工与措施费。",
        )
    elif p >= 0.3:
        push(
            "p_mid",
            "【预测】开裂风险概率处于中等水平",
            (
                f"本次 P≈{p:.2f}（{alert}）。建议重点检查早期失水、环境风速与昼夜温差；"
                "核对减水剂、膨胀剂（若使用）与纤维的相容性及分散均匀性；约束较强部位可考虑跳仓、后浇带或滑动层等构造措施。"
            ),
            "早期表面裂缝与贯通微裂缝风险有望下降。",
            "措施成本通常可控，以养护与施工组织为主。",
        )
    else:
        push(
            "p_low",
            "【预测】开裂风险概率较低",
            (
                f"本次 P≈{p:.2f}（{alert}）。模型假设下组合相对有利，仍应按 GB 50164 落实标准养护、"
                "接缝与约束部位构造，避免施工离析、冷缝与局部急干等使实测劣于预测。"
            ),
            "维持设计与施工一致性，降低「模型与现场脱节」风险。",
            "以常规质量控制成本为主。",
        )

    # —— 裂缝宽度（与信息性 0.30 mm 参照衔接）——
    if w_mm > 0.30:
        push(
            "w_over_ref",
            "【预测】最大裂缝宽度超过常规信息性参照",
            (
                f"预测最大裂缝宽度约 {w_mm:.3f} mm，高于室内正常环境下钢筋混凝土构件常见信息性参照 0.30 mm（非自动规范验算）。"
                "可尝试在强度允许范围内降低水胶比、优化骨料级配与砂率，或提高纤维掺量与长径比；大体积与强约束构件宜同步考虑温控与浇筑顺序。"
            ),
            "宏观裂缝宽度与耐久观感通常更易满足管理目标。",
            "胶材与纤维用量变化对单方造价影响需经济比选。",
        )
    elif w_mm > 0.20:
        push(
            "w_near_ref",
            "【预测】裂缝宽度接近信息性上限",
            (
                f"预测约 {w_mm:.3f} mm，已接近 0.30 mm 信息性参照。可作敏感性方向：略减水胶比、延长保湿养护或微调纤维掺量，观察对宽度余量的改善。"
            ),
            "为后续正式配合比留出安全余量。",
            "改动幅度一般较小。",
        )

    # —— 裂缝密度 ——
    if dens >= 4.5:
        extra = ""
        if "聚丙烯" in fiber:
            extra = " 聚丙烯纤维有利于细化早期裂缝，若条数仍偏多可评估与钢纤维或玄武岩纤维混掺。"
        push(
            "dens_high",
            "【预测】单位面积裂缝条数偏多",
            (
                f"预测裂缝密度约 {dens:.2f} 条/m²，表面龟裂或细密裂缝风险相对突出。"
                "除配合比外，宜强化浆体抗裂（纤维、养护剂或薄膜保湿）并控制抹面时机与风速。"
                + extra
            ),
            "细密裂缝与渗透路径有望减少。",
            "养护与纤维方案调整带来的成本因项目而异。",
        )

    # —— 应力-强度比 ——
    if ssr >= 0.82:
        push(
            "ssr_high",
            "【预测】应力-强度比偏高（启发式）",
            (
                f"启发式指标 σ/fₜ≈{ssr:.2f}，约束拉应力相对抗拉能力裕度偏紧。"
                "可延缓拆模与上部加载、改善保湿与内外温差控制；强约束节点处复核构造释放措施。"
            ),
            "早期拉裂与温度裂缝风险有望降低。",
            "工期与模板周转可能受一定影响。",
        )

    # —— 时间维度 ——
    if t_crack < 72.0:
        push(
            "t_crack_early",
            "【预测】首条可见缝出现时间偏早",
            (
                f"模型估算浇筑后约 {t_crack:.1f} h 即可能出现可见缝，塑性收缩与早期失水敏感。"
                "建议初凝前后即开始有效保湿（薄膜、喷雾或养护剂），防风防晒并避免过晚抹面。"
            ),
            "塑性收缩裂缝与表面龟裂常可明显减轻。",
            "养护材料与人工略有增加。",
        )

    if t_safe < 72.0 and p >= 0.35:
        push(
            "t_safe_short",
            "【预测】保湿安全窗口偏短",
            (
                f"模型估算安全保湿窗口约 {t_safe:.1f} h；在 P 不低时宜保守执行，不宜早于该量级撤除保湿。"
                "应结合现场同条件试块强度与环境条件综合决定拆模与养护终止时间。"
            ),
            "降低因过早撤除养护诱发开裂的概率。",
            "延长养护占用模板与场地时间可能略增。",
        )

    if t_crit < 2.8 and p > 0.42:
        push(
            "t_crit_early",
            "【预测】临界龄期较早",
            (
                f"模型估算临界龄期约 {t_crit:.2f} d，早期温度–收缩耦合较敏感。"
                "宜加强测温、控制入模温度与内外温差，避免冷击与急剧降温。"
            ),
            "温度裂缝与早期贯通缝风险有望受控。",
            "测温与保温措施有一定现场管理成本。",
        )

    # —— 配合比与环境 ——
    if w_b >= 0.46:
        push(
            "wb_high",
            "【预测】水胶比偏高",
            (
                f"当前水胶比约 {w_b:.2f}，在强度与工作性允许范围内适度降低有利于减小孔隙率与收缩，改善抗裂表现。"
            ),
            "裂缝宽度与渗透性常同步改善。",
            "可能略增胶材或高效减水剂用量。",
        )

    if hum < 55.0 and p > 0.40:
        push(
            "hum_low",
            "【预测】环境湿度偏低",
            (
                f"相对湿度约 {hum:.0f}%，表面蒸发驱动力大，与当前 P≈{p:.2f} 叠加时早期开裂风险上升。"
                "建议强制保湿、降低风速与日照直射，必要时二次抹面或喷雾。"
            ),
            "干缩与塑性裂缝显著减轻的案例较多。",
            "以养护与覆盖措施成本为主。",
        )

    if binder >= 520.0 and p > 0.45:
        push(
            "binder_high",
            "【预测】胶材用量较高",
            (
                f"胶材用量约 {binder:.0f} kg/m³，水化温升与自收缩倾向增大。"
                "可评估矿物掺合料替代比例、入模温度与分层浇筑，并与纤维阻裂措施协同。"
            ),
            "温度收缩与早期裂缝风险有望缓解。",
            "掺合料与温控可能略影响单方成本与工期。",
        )

    return out


OPTIMIZATION_RULES: List[Dict[str, Any]] = [
    {
        "name": "钢纤维掺量与桥接",
        "condition": lambda x: x.get("fiber_material") == "钢纤维"
        and (x.get("fiber_content", 0) < 0.85)
        and (x.get("cracking_risk_score", 0) > 0.58),
        "suggestion": "钢纤维弹性模量高、端部机械锚固强，可适当提高体积掺量至约 1.0–1.2%，并优先选用端钩/波纹型以改善裂缝桥接。",
        "expected_improvement": "宏观裂缝宽度与间距通常可明显改善（需配合试验验证）。",
        "cost_impact": "材料成本一般上升约 8–15%。",
    },
    {
        "name": "聚丙烯纤维阻裂与混掺",
        "condition": lambda x: x.get("fiber_material") == "聚丙烯纤维"
        and (x.get("cracking_risk_score", 0) > 0.52),
        "suggestion": "聚丙烯纤维主要通过抑制塑性收缩与细化裂缝发挥作用；开裂风险仍偏高时，可考虑与钢纤维或玄武岩纤维混掺，并控制水胶比与砂率。",
        "expected_improvement": "早期塑性裂缝与表面龟裂往往减轻；结构级裂缝需靠钢纤维或配筋体系。",
        "cost_impact": "PP 单价相对低，混掺时成本增幅取决于钢/玄武岩比例。",
    },
    {
        "name": "玄武岩纤维粘结与养护",
        "condition": lambda x: x.get("fiber_material") == "玄武岩纤维"
        and (
            (x.get("cracking_risk_score", 0) > 0.52)
            or (x.get("humidity", 100) < 50)
        ),
        "suggestion": "玄武岩纤维表面粘结与浆体密实度、养护湿度密切相关；建议加强早期保湿与足够养护龄期，必要时优化减水剂与浆体黏度以改善分散。",
        "expected_improvement": "界面脱粘与细密裂缝扩展风险有望降低。",
        "cost_impact": "养护措施成本通常有限；外加剂微调可能略增胶材成本。",
    },
    {
        "name": "玻璃纤维耐碱与替代",
        "condition": lambda x: x.get("fiber_material") == "玻璃纤维"
        and (x.get("cracking_risk_score", 0) > 0.5),
        "suggestion": "普通玻璃纤维在碱性孔溶液中长期强度退化风险较高；除配合比优化外，宜优先采用耐碱玻璃纤维或表面处理产品，并控制碱含量与裂缝宽度。",
        "expected_improvement": "长期耐久与韧性保持更可靠。",
        "cost_impact": "耐碱玻纤或涂层产品价格通常高于普通玻纤。",
    },
    {
        "name": "高强混凝土抗裂与养护",
        "condition": _rule_high_strength_crack,
        "suggestion": "强度等级较高时，胶凝材料水化快、自收缩与温度收缩更突出；建议优化养护制度（保湿、控温）、控制拆模与加载龄期，并复核纤维分散与减水剂相容性。",
        "expected_improvement": "早期收缩裂缝与温度裂缝风险有望降低。",
        "cost_impact": "养护与温控措施可能增加现场管理成本。",
    },
    {
        "name": "自密实混凝土工作度与抗裂",
        "condition": lambda x: x.get("concrete_type") == "自密实混凝土(SCC)"
        and (x.get("cracking_risk_score", 0) > 0.5),
        "suggestion": "SCC 对浆体稳定性与泌水敏感，开裂风险偏高时宜检查粉体-水胶比、粘度改性剂与纤维分散，避免离析与塑性沉降裂缝。",
        "expected_improvement": "表面沉降裂缝与贯通微裂缝可减少。",
        "cost_impact": "外加剂与胶材微调可能略增单方材料费。",
    },
]


class SteelFiberCrackPredictor:
    def __init__(self, model_dir: str, config_path: str | None = None):
        self.model_dir = model_dir
        self.config_path = config_path
        ensure_default_models(model_dir)
        self._load_models()

    def _load_models(self) -> None:
        try:
            self.crack_width_model = joblib.load(
                os.path.join(self.model_dir, "crack_regressor.pkl")
            )

            crack_density_path = os.path.join(
                self.model_dir, "crack_density_regressor.pkl"
            )
            if os.path.exists(crack_density_path):
                self.crack_density_model = joblib.load(crack_density_path)
            else:
                self.crack_density_model = None

            self.crack_risk_model = joblib.load(
                os.path.join(self.model_dir, "crack_classifier.pkl")
            )

            self.scaler = joblib.load(
                os.path.join(self.model_dir, "feature_scaler.pkl")
            )
        except Exception as e:
            raise RuntimeError(
                "加载 models 目录中的 pkl 失败。请删除该目录下全部 .pkl 后重新打开页面，"
                "系统将自动重建演示模型。"
            ) from e

    def _preprocess_for_model(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"输入缺少特征列: {missing}；当前需要 {len(FEATURE_COLUMNS)} 维。")
        Xo = X[FEATURE_COLUMNS].astype(np.float64, copy=False)
        return self.scaler.transform(Xo)

    def predict_all(self, X: pd.DataFrame, extra_info: dict | None = None) -> dict:
        X_scaled = self._preprocess_for_model(X)

        cw_raw = float(self.crack_width_model.predict(X_scaled)[0])
        crack_width = float(np.clip(cw_raw, 0.02, 2.0)) if np.isfinite(cw_raw) else 0.10

        proba = self.crack_risk_model.predict_proba(X_scaled)[0]
        classes = np.asarray(self.crack_risk_model.classes_)
        risk_p = _risk_probability_from_multiclass(proba, classes)
        if not np.isfinite(risk_p):
            risk_p = 0.5
        risk_p = float(np.clip(risk_p, 0.0, 1.0))
        alert_level = _alert_level_from_p(risk_p)

        density_source = "regressor"
        if self.crack_density_model is not None:
            cd_raw = float(self.crack_density_model.predict(X_scaled)[0])
            if np.isfinite(cd_raw) and cd_raw >= 0:
                density_val = float(np.clip(cd_raw, 0.05, 25.0))
            else:
                density_val = _fallback_crack_density_per_m2(crack_width, risk_p)
                density_source = "fallback"
        else:
            density_val = _fallback_crack_density_per_m2(crack_width, risk_p)
            density_source = "fallback"

        density_source_cn = (
            "由裂缝密度回归模型给出"
            if density_source == "regressor"
            else "无有效密度回归输出时，按裂缝宽度与风险概率经验换算（仅保证结果可展示）"
        )

        x0 = X.iloc[0].to_dict()
        time_dim = _derive_time_metrics(x0, risk_p)
        ssr = float(_stress_strength_ratio_heuristic(x0, risk_p))
        if not np.isfinite(ssr):
            ssr = 0.35

        width_ref = _crack_width_gb50010_reference(crack_width)
        std = _standards_bundle()
        std["crack_width_reference_gb50010"] = width_ref

        try:
            idx_am = int(np.argmax(proba))
            rca = classes[idx_am]
            risk_argmax = int(rca) if isinstance(rca, (int, np.integer)) else int(float(rca))
        except (TypeError, ValueError, IndexError):
            risk_argmax = 0

        result: Dict[str, Any] = {
            "standards": std,
            "predictions": {
                "time_dimension": {
                    "cracking_time_hours": time_dim["cracking_time_hours"],
                    "critical_age_days": time_dim["critical_age_days"],
                    "safety_window_hours": time_dim["safety_window_hours"],
                    "note": _time_dimension_note_gb(),
                },
                "state_dimension": {
                    "risk_probability": risk_p,
                    "crack_width_mm": crack_width,
                    "crack_density_per_m2": density_val,
                    "crack_density_source_cn": density_source_cn,
                    "stress_strength_ratio": ssr,
                    "stress_strength_note_cn": (
                        "对应约束拉应力与抗拉强度之比的量级估计，非 GB 规定试验直接测定值；"
                        "抗拉强度宜以棱柱体或劈裂试验按 GB/T 50081 测定后复核。"
                    ),
                    "crack_width_gb50010": width_ref,
                    "note": (
                        "裂缝宽度、裂缝条数密度为模型或经验换算输出；"
                        "与 GB 50010 的信息性比对见「裂缝宽度与规范参照」说明。"
                    ),
                },
                "risk_dimension": {
                    "alert_level": alert_level,
                    "bands": "低风险：P<0.3；中风险：0.3≤P≤0.7；高风险：P>0.7",
                    "risk_probability": risk_p,
                    "note": std["risk_tier_note"],
                },
                "crack_width": crack_width,
                "crack_density": density_val,
                "risk_level": alert_level,
                "risk_confidence": risk_p,
            },
            "intermediate": {
                "risk_raw_proba": proba.tolist(),
                "risk_class_argmax": risk_argmax,
                "crack_density_source": density_source,
            },
        }

        if extra_info:
            result["intermediate"].update(extra_info)

        result["recommendations"] = self._generate_recommendations(
            X.iloc[0].to_dict(), result
        )

        return result

    def _generate_recommendations(self, x: dict, result: dict) -> List[Dict[str, Any]]:
        enriched = dict(x)
        preds = result.get("predictions", {})
        enriched["cracking_risk_score"] = float(
            preds.get("risk_confidence", 0.0)
        )
        mid = result.get("intermediate") or {}
        for key in ("fiber_material", "strength_grade", "concrete_type"):
            v = mid.get(key)
            if isinstance(v, str):
                enriched[key] = v

        rule_recs: List[Dict[str, Any]] = []
        for rule in OPTIMIZATION_RULES:
            cond: OptimizationCondition = rule["condition"]
            try:
                if cond(enriched):
                    rule_recs.append(
                        {
                            "title": rule["name"],
                            "suggestion": rule["suggestion"],
                            "expected_improvement": rule.get(
                                "expected_improvement", ""
                            ),
                            "cost_impact": rule.get("cost_impact", ""),
                            "source": "rule",
                        }
                    )
            except Exception:
                continue

        try:
            pred_recs = _prediction_based_recommendations(x, result)
        except Exception:
            pred_recs = [
                {
                    "title": "【预测】建议条目生成异常",
                    "suggestion": (
                        "无法根据本次预测自动生成建议，请尝试：右上角菜单 **Clear cache** 后刷新；"
                        "或确认已保存最新 `src/predictor.py` 并从项目根目录启动 Streamlit。"
                    ),
                    "expected_improvement": "",
                    "cost_impact": "",
                    "source": "prediction",
                }
            ]
        merged = pred_recs + rule_recs
        if not merged:
            p0 = _safe_float(
                (result.get("predictions") or {}).get("risk_confidence"), 0.0
            )
            merged = [
                {
                    "title": "【预测】通用提示",
                    "suggestion": (
                        f"当前开裂风险概率 P≈{p0:.2f}。请结合配合比、养护与环境条件进行综合控制；"
                        "若本页长期无具体条目，请清除 Streamlit 缓存后重试。"
                    ),
                    "expected_improvement": "",
                    "cost_impact": "",
                    "source": "prediction",
                }
            ]
        return merged

