import sys
from pathlib import Path
from typing import Any, Dict

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from fastapi import FastAPI
from pydantic import BaseModel

from src.data_processor import validate_and_transform
from src.paths import CONFIG_YAML, MODELS_DIR
from src.predictor import SteelFiberCrackPredictor


app = FastAPI(title="Steel Fiber Concrete Crack Predictor API")

predictor = SteelFiberCrackPredictor(
    model_dir=str(MODELS_DIR),
    config_path=str(CONFIG_YAML) if CONFIG_YAML.exists() else None,
)


class PredictRequest(BaseModel):
    fiber_parameters: Dict[str, Any]
    concrete_mix: Dict[str, Any]
    env_process: Dict[str, Any] | None = None


@app.post("/predict")
def predict(req: PredictRequest):
    raw: Dict[str, Any] = {
        **req.fiber_parameters,
        **req.concrete_mix,
        **(req.env_process or {}),
    }

    valid, X, msg, extra, warnings = validate_and_transform(
        raw, emit_streamlit_warnings=False
    )
    if not valid:
        return {"success": False, "message": msg}

    result = predictor.predict_all(X, extra)
    return {"success": True, "data": result, "warnings": warnings}

