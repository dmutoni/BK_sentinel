"""
BK Sentinel — Data & Model Loader
All file loading happens here with in-memory caching.
The cache is populated once on first request and reused for speed.
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
from functools import lru_cache
from config import (
    VERIFIED_CSV, TRANSITION_MATRIX, TIME_TO_ABSORPTION,
    FUNDAMENTAL_MATRIX, ABSORPTION_PROBS,
    MODEL_PKL, ENCODERS_PKL, FEATURE_COLS_JSON,
    STATES
)

warnings.filterwarnings("ignore")

# ── in-memory cache ───────────────────────────────────────────
_cache: dict = {}


def get_portfolio_df() -> pd.DataFrame:
    """Load and cache the full 16-month panel dataset."""
    if "df" not in _cache:
        print("[Loader] Loading bk_sentinel_verified.csv...")
        _cache["df"] = pd.read_csv(VERIFIED_CSV, low_memory=False)
        print(f"[Loader] Loaded {len(_cache['df']):,} rows.")
    return _cache["df"]


def get_transition_matrix() -> pd.DataFrame:
    """Load and cache the empirical transition matrix."""
    if "P_df" not in _cache:
        print("[Loader] Loading transition matrix...")
        df = pd.read_csv(TRANSITION_MATRIX, index_col=0)
        _cache["P_df"] = df.reindex(index=STATES, columns=STATES)
        _cache["P"]    = _cache["P_df"].values
    return _cache["P_df"]


def get_P_numpy() -> np.ndarray:
    """Return the transition matrix as a numpy array."""
    get_transition_matrix()
    return _cache["P"]


def get_model():
    """Load and cache the trained Random Forest model and encoders."""
    if "model" not in _cache:
        print("[Loader] Loading ML model...")
        _cache["model"]    = pickle.load(open(MODEL_PKL, "rb"))
        _cache["encoders"] = pickle.load(open(ENCODERS_PKL, "rb"))
        _cache["features"] = json.load(open(FEATURE_COLS_JSON))
        print("[Loader] Model loaded successfully.")
    return _cache["model"], _cache["encoders"], _cache["features"]


def get_absorption_data():
    """Load and cache Layer 3 absorption analysis outputs."""
    if "t_df" not in _cache:
        print("[Loader] Loading absorption data...")
        try:
            _cache["t_df"] = pd.read_csv(TIME_TO_ABSORPTION)
            _cache["N_df"] = pd.read_csv(FUNDAMENTAL_MATRIX, index_col=0)
            print("[Loader] Absorption data loaded.")
        except FileNotFoundError:
            print("[Loader] WARNING: Absorption files not found. Run Notebook 04 first.")
            _cache["t_df"] = None
            _cache["N_df"] = None
    return _cache["t_df"], _cache["N_df"]


def clear_cache():
    """Clear all cached data — useful after retraining models."""
    _cache.clear()
    print("[Loader] Cache cleared.")


def encode_features(row_dict: dict, encoders: dict, features: list) -> np.ndarray:
    """
    Encode a single account row for ML prediction.
    Handles categorical encoding and missing feature columns.
    """
    df = pd.DataFrame([row_dict])
    df["segment_enc"]   = encoders["segment"].transform(
        df["segment"].astype(str).fillna("UNKNOWN"))
    df["loan_type_enc"] = encoders["loan_type"].transform(
        df["loan_type"].astype(str).fillna("UNKNOWN"))
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df[features] = df[features].fillna(0)
    return df[features].values[0]


def encode_dataframe(df: pd.DataFrame, encoders: dict, features: list) -> np.ndarray:
    """
    Encode an entire DataFrame for batch ML prediction.
    Returns a numpy array ready for model.predict().
    """
    df = df.copy()
    df["segment_enc"]   = encoders["segment"].transform(
        df["segment"].astype(str).fillna("UNKNOWN"))
    df["loan_type_enc"] = encoders["loan_type"].transform(
        df["loan_type"].astype(str).fillna("UNKNOWN"))
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df[features] = df[features].fillna(0)
    return df[features].values
