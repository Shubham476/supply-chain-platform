"""Loads the trained model artefact once at process startup and exposes a
single `predict()` function. The model is never retrained on request."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from app.config import MODEL_PATH
from ml.features import ALL_FEATURES

logger = logging.getLogger("model")

_artefact: Optional[dict] = None


def load_model(path: Path = MODEL_PATH) -> dict:
    global _artefact
    if _artefact is None:
        if not path.exists():
            raise FileNotFoundError(
                f"Model artefact not found at {path}. Run `python ml/train_model.py` first."
            )
        _artefact = joblib.load(path)
        logger.info("Loaded model artefact: %s", _artefact.get("model_name"))
    return _artefact


def predict_one(feature_row: pd.DataFrame) -> tuple[float, str, str, float]:
    artefact = load_model()
    pipeline = artefact["pipeline"]
    proba = float(pipeline.predict_proba(feature_row[ALL_FEATURES])[:, 1][0])
    label = "HIGH" if proba >= 0.5 else "LOW"
    return proba, label, artefact["model_name"], artefact["on_time_threshold_hours"]
