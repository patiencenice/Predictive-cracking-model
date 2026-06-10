"""温度应力解释链（Phase 1）：派生与无量纲指数，非主模型特征。"""

from experiments.thermal_stress.derive import (
    derive_thermal_stress_features,
    thermal_stress_explain_sentence_zh,
)
from experiments.thermal_stress.engineering_chain import (
    derive_thermal_engineering_display,
    thermal_engineering_conclusion_zh,
)
from experiments.thermal_stress.optional_fields import (
    derive_thermal_optional_context,
    resolve_f_t_for_eta,
)

__all__ = [
    "derive_thermal_stress_features",
    "thermal_stress_explain_sentence_zh",
    "derive_thermal_engineering_display",
    "thermal_engineering_conclusion_zh",
    "derive_thermal_optional_context",
    "resolve_f_t_for_eta",
]
