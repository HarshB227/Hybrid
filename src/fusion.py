"""
Fusion Engine for Hybrid Risk Assessment
Combines ML predictions with market and behavioral analysis.
"""

import logging
import joblib
import pandas as pd

from src.config import XGB_MODEL_PATH, RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD
from src.feature_engineering import build_features

logger = logging.getLogger(__name__)

# Lazy model reference – loaded on first call to avoid import-time crash
_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load(XGB_MODEL_PATH)
            logger.info(f"Fusion: model loaded from {XGB_MODEL_PATH}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Model not found at {XGB_MODEL_PATH}. "
                "Run `python -m src.train_model` first."
            )
    return _model


def compute_alert(features: dict) -> dict:
    """
    Compute hybrid risk assessment combining ML and market factors.

    Args:
        features (dict): Must contain keys:
            year, day, length, weight, count, looped, neighbors, income,
            avg_value, gas_used, failed_tx

    Returns:
        dict: fraud_probability, market_risk, hybrid_risk, risk_level
    """
    # Build the model-ready feature DataFrame using the shared utility
    model_features = {
        "year":      features.get("year", 2024),
        "day":       features.get("day", 180),
        "length":    features["count"],
        "weight":    features["avg_value"],
        "count":     features["count"],
        "looped":    features["failed_tx"],
        "neighbors": features["unique_contacts"],
        "income":    features["avg_value"] * features["count"],
    }

    df = build_features(model_features)

    ml_prob = float(_get_model().predict_proba(df)[0][1])

    # Market risk factors
    market_risk = 0.0
    if features.get("avg_value", 0) > 10:
        market_risk += 0.2
    if features.get("gas_used", 0) > 50_000:
        market_risk += 0.15
    fail_rate = features.get("failed_tx", 0) / (features.get("count", 0) + 1)
    if fail_rate > 0.3:
        market_risk += 0.25

    hybrid_risk = (ml_prob * 0.7) + (market_risk * 0.3)

    if hybrid_risk > RISK_HIGH_THRESHOLD:
        risk_level = "HIGH"
    elif hybrid_risk > RISK_MEDIUM_THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "fraud_probability": ml_prob,
        "market_risk":       market_risk,
        "hybrid_risk":       hybrid_risk,
        "risk_level":        risk_level,
    }