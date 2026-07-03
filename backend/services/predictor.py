"""
BK Sentinel — ML Prediction Service
Handles all interactions with the trained ML model.
"""

import numpy as np
import pandas as pd
from database.loader import get_model, get_portfolio_df, encode_features, encode_dataframe
from config import STATES, STATE_ORDER

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Default"}

TRAJ_FEATURES = {"prev_risk_state_code", "months_in_state", "dpd_band_position", "dpd_delta"}


def _add_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag/trajectory features to a snapshot using the full 16-month portfolio.
    Required when the saved model was trained with these features.
    """
    df         = df.copy()
    full       = get_portfolio_df()
    all_months = sorted(full["observation_month"].unique())
    obs_month  = df["observation_month"].iloc[0] if len(df) > 0 else None

    # ── Previous-month state and DPD ─────────────────────────────────────────
    if obs_month and obs_month in all_months:
        m_idx = all_months.index(obs_month)
        if m_idx > 0:
            prev_month = all_months[m_idx - 1]
            prev = (
                full[full["observation_month"] == prev_month]
                [["loan_id", "risk_state_code", "days_in_arrears"]]
                .rename(columns={"risk_state_code": "prev_risk_state_code",
                                 "days_in_arrears": "prev_dpd"})
            )
            df = df.merge(prev, on="loan_id", how="left")
        else:
            df["prev_risk_state_code"] = df["risk_state_code"]
            df["prev_dpd"]             = df["days_in_arrears"]
    else:
        df["prev_risk_state_code"] = df["risk_state_code"]
        df["prev_dpd"]             = df["days_in_arrears"]

    df["prev_risk_state_code"] = df["prev_risk_state_code"].fillna(df["risk_state_code"]).astype(int)
    df["prev_dpd"]             = df["prev_dpd"].fillna(0)
    df["dpd_delta"]            = df["days_in_arrears"] - df["prev_dpd"]

    # ── Months in current state (streak from full history) ───────────────────
    if obs_month:
        loan_ids      = df["loan_id"].tolist()
        hist          = full[full["loan_id"].isin(loan_ids)].copy().sort_values(["loan_id", "observation_month"])
        prev_s        = hist.groupby("loan_id")["risk_state"].shift(1)
        changed       = (prev_s != hist["risk_state"]) | prev_s.isna()
        hist["_sid"]  = changed.groupby(hist["loan_id"]).cumsum()
        hist["months_in_state"] = hist.groupby(["loan_id", "_sid"]).cumcount() + 1
        streak = hist[hist["observation_month"] == obs_month][["loan_id", "months_in_state"]]
        df     = df.merge(streak, on="loan_id", how="left")

    df["months_in_state"] = df["months_in_state"].fillna(1).astype(int)

    # ── DPD position within BNR band ─────────────────────────────────────────
    dpd  = df["days_in_arrears"]
    cond = [df["risk_state"] == "Low",    df["risk_state"] == "Medium",
            df["risk_state"] == "High",   df["risk_state"] == "Default"]
    vals = [dpd / 3.0, (dpd - 5).clip(lower=0) / 24.0,
            (dpd - 31).clip(lower=0) / 58.0, (dpd / 280.0).clip(upper=1.0)]
    df["dpd_band_position"] = np.select(cond, vals, default=0.0).clip(0, 1)

    return df


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

    # Compute trajectory features if the model was trained with them
    if TRAJ_FEATURES & set(features):
        df = _add_trajectory_features(df)

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
