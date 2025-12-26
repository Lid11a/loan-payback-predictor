# src/api/app.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Loan Payback Prediction API", version="0.1.0")

# Global variables for the loaded model bundle
BUNDLE: Dict[str, Any] | None = None
FEATURES: List[str] = []
ACTIVE_RUN_ID: str | None = None


class PredictRequest(BaseModel):
    """
    Input schema for prediction.

    The request contains one client record as a dictionary of features.
    """
    features: Dict[str, Any] = Field(..., description="Client features as key-value pairs")


class PredictResponse(BaseModel):
    """
    Output schema for prediction.

    proba: predicted probability (float)
    prediction: 0/1 decision based on stored threshold
    threshold: threshold used for decision
    """
    proba: float
    prediction: int
    threshold: float


class PredictBatchRequest(BaseModel):
    """
    Input schema for batch prediction.

    The request contains multiple client records as a list of feature dictionaries.
    """
    items: List[Dict[str, Any]] = Field(..., description="List of client feature dicts")


class PredictBatchResponse(BaseModel):
    """
    Output schema for batch prediction.

    results is a list of per-record predictions, matching the single PredictResponse schema.
    """
    results: List[PredictResponse]


def load_bundle(model_path: str | Path = "models/best_model.joblib") -> Dict[str, Any]:
    """
    The function loads a model bundle from disk.

    The bundle is expected to contain:
    - model
    - preprocessor
    - threshold
    - meta (optional)
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")

    bundle = joblib.load(path)
    return bundle


def get_expected_features(bundle: Dict[str, Any]) -> List[str]:
    """
    The function extracts the expected feature list from the bundle meta.

    Returns a list of feature names (order does not matter).
    """
    meta = bundle.get("meta", {})
    numeric = meta.get("numeric_features", [])
    categorical = meta.get("categorical_features", [])
    return list(categorical) + list(numeric)


def load_active_run_id(path: str | Path = "models/latest_run.txt") -> str | None:
    """
    The function loads the active MLflow run_id from disk.

    Returns run_id as a string if the file exists and is non-empty.
    Otherwise, returns None.
    """
    p = Path(path)
    if not p.exists():
        return None

    run_id = p.read_text(encoding="utf-8").strip()
    return run_id or None


def get_proba(bundle: Dict[str, Any], x_proc: Any) -> float:
    """
    The function extracts a probability from the model output.

    If the model supports predict_proba (sklearn-style), the function uses it.
    Otherwise, the function falls back to predict (LightGBM Booster style).

    Returns a probability in [0, 1].
    """
    model = bundle["model"]

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(x_proc)[0, 1])
    else:
        proba = float(model.predict(x_proc)[0])

    if not (0.0 <= proba <= 1.0):
        raise ValueError(f"Model returned a non-probability value: {proba}")

    return proba


@app.on_event("startup")
def startup_event() -> None:
    """
    The function loads the model bundle at application startup.
    """
    global BUNDLE, FEATURES, ACTIVE_RUN_ID

    try:
        BUNDLE = load_bundle("models/best_model.joblib")
        FEATURES = get_expected_features(BUNDLE)
        ACTIVE_RUN_ID = load_active_run_id("models/latest_run.txt")

        logger.info(
            "API startup: model loaded. threshold=%.6f features=%s",
            float(BUNDLE["threshold"]),
            len(FEATURES),
        )
    except Exception as e:
        # The API can still start, but /ready will show it's not ready
        BUNDLE = None
        FEATURES = []
        ACTIVE_RUN_ID = None
        logger.exception("API startup: failed to load model. error=%s", e)


@app.get("/health")
def health() -> Dict[str, str]:
    """
    The endpoint checks that the API process is alive.
    """
    return {"status": "ok"}


@app.get("/ready")
def ready() -> Dict[str, Any]:
    """
    The endpoint checks that the model bundle is loaded and ready for inference.
    """
    return {
        "model_loaded": BUNDLE is not None,
        "n_features_expected": len(FEATURES),
        "active_run_id": ACTIVE_RUN_ID,
    }


@app.get("/features")
def features() -> Dict[str, Any]:
    """
    The endpoint returns the expected feature list for inference.
    """
    if BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /ready.")

    return {
        "n_features_expected": len(FEATURES),
        "features": FEATURES,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    The endpoint generates a prediction for one record.
    """
    if BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /ready.")

    # Validate that the request contains exactly the required features
    req_keys = set(req.features.keys())
    expected_keys = set(FEATURES)

    missing = sorted(list(expected_keys - req_keys))
    extra = sorted(list(req_keys - expected_keys))

    if missing or extra:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid feature set",
                "missing_features": missing,
                "extra_features": extra,
                "n_expected": len(FEATURES),
            },
        )

    x = pd.DataFrame([req.features], columns=FEATURES)

    try:
        x_proc = BUNDLE["preprocessor"].transform(x)
        proba = get_proba(BUNDLE, x_proc)
    except Exception as e:
        logger.exception("Prediction failed. error=%s", e)
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.") from e

    threshold = float(BUNDLE["threshold"])
    pred = int(proba >= threshold)

    return PredictResponse(proba=proba, prediction=pred, threshold=threshold)


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    """
    The endpoint generates predictions for multiple records.
    """
    if BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /ready.")

    if not req.items:
        raise HTTPException(status_code=400, detail="Empty items list.")

    expected_keys = set(FEATURES)

    # Validate feature sets for all items first
    for i, feats in enumerate(req.items):
        req_keys = set(feats.keys())
        missing = sorted(list(expected_keys - req_keys))
        extra = sorted(list(req_keys - expected_keys))

        if missing or extra:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid feature set",
                    "item_index": i,
                    "missing_features": missing,
                    "extra_features": extra,
                    "n_expected": len(FEATURES),
                },
            )

    x = pd.DataFrame(req.items, columns=FEATURES)

    try:
        x_proc = BUNDLE["preprocessor"].transform(x)

        model = BUNDLE["model"]
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(x_proc)[:, 1]
        else:
            probas = model.predict(x_proc)

        probas = [float(p) for p in probas]
    except Exception as e:
        logger.exception("Batch prediction failed. error=%s", e)
        raise HTTPException(status_code=500, detail="Batch prediction failed. Check server logs.") from e

    threshold = float(BUNDLE["threshold"])

    results = [
        PredictResponse(
            proba=p,
            prediction=int(p >= threshold),
            threshold=threshold,
        )
        for p in probas
    ]

    return PredictBatchResponse(results=results)
