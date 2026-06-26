"""
BK Sentinel — ML Prediction Service
Handles all interactions with the trained Random Forest model.
"""

import numpy as np
import pandas as pd
from database.loader import get_model, encode_features, encode_dataframe
from config import STATES, STATE_ORDER

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Default"}


def predict_single(row_dict: dict) -> dict:
    """
    Run prediction for a single account row.

    Returns:
        predicted_state: the class with highest probability
        probabilities:   dict of {state: probability} for all 4 classes
        pred_class_idx:  integer class index (for SHAP)
    """
    model, encoders, features = get_model()
    X_row = encode_features(row_dict, encoders, features)
    proba = model.predict_proba(X_row.reshape(1, -1))[0]
    pred  = int(model.predict(X_row.reshape(1, -1))[0])

    return {
        "X_row":          X_row,
        "predicted_state": RISK_LABELS[pred],
        "pred_class_idx":  pred,
        "probabilities":  {RISK_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)},
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions for an entire DataFrame of accounts.
    Adds columns: pred, p_default, p_high, p_medium, p_low, score
    """
    model, encoders, features = get_model()
    X = encode_dataframe(df, encoders, features)

    proba = model.predict_proba(X)
    preds = model.predict(X)

    df = df.copy()
    df["pred"]      = [RISK_LABELS[p] for p in preds]
    df["p_low"]     = proba[:, 0]
    df["p_medium"]  = proba[:, 1]
    df["p_high"]    = proba[:, 2]
    df["p_default"] = proba[:, 3]
    df["score"]     = df["p_default"] + df["p_high"]

    # Deterioration flag
    df["cur_rank"]  = df["risk_state"].map(STATE_ORDER)
    df["pred_rank"] = df["pred"].map(STATE_ORDER)
    df["is_worse"]  = df["pred_rank"] > df["cur_rank"]

    return df


def get_prediction_summary(df_predicted: pd.DataFrame) -> dict:
    """Return count of accounts predicted in each state next month."""
    counts = df_predicted["pred"].value_counts()
    return {s: int(counts.get(s, 0)) for s in STATES}
