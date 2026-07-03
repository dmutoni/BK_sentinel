"""
BK Sentinel — Data & Model Loader
All file loading happens here with in-memory caching.
The cache is populated once on first request and reused for speed.
"""

import json
import pickle
import sys
import warnings
import traceback

import numpy as np
import pandas as pd
from config import (
    ABSORPTION_PROBS,
    ENCODERS_PKL,
    FEATURE_COLS_JSON,
    FUNDAMENTAL_MATRIX,
    MODEL_PKL,
    STATES,
    TIME_TO_ABSORPTION,
    TRANSITION_MATRIX,
    VERIFIED_CSV,
)

# NumPy pickle compatibility shim (for artifacts saved in different envs)
try:
    import numpy.core as _np_core
    import numpy.core.multiarray as _np_multiarray
    import numpy.core.numerictypes as _np_numerictypes

    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
    sys.modules.setdefault("numpy._core.numerictypes", _np_numerictypes)
except Exception:
    pass

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
        _cache["P"] = _cache["P_df"].values
    return _cache["P_df"]


def get_P_numpy() -> np.ndarray:
    """Return the transition matrix as a numpy array."""
    get_transition_matrix()
    return _cache["P"]


def get_model():
    """Load and cache the trained model, encoders, and feature list."""
    if "model" not in _cache:
        print("[Loader] Loading ML model artifacts...")
        try:
            with open(MODEL_PKL, "rb") as f:
                _cache["model"] = pickle.load(f)

            with open(ENCODERS_PKL, "rb") as f:
                _cache["encoders"] = pickle.load(f)

            with open(FEATURE_COLS_JSON, "r") as f:
                _cache["features"] = json.load(f)

            print("[Loader] Model loaded successfully.")
        except Exception as e:
            print(f"[Loader] ERROR loading artifacts: {e}")
            print(traceback.format_exc())
            raise

    return _cache["model"], _cache["encoders"], _cache["features"]


# NOTE: we used to build a shap.TreeExplainer here and cache it, but the
# `shap` package's native TreeExplainer segfaults against this model on
# some machines (a hard crash in its numba/llvmlite dependency that no
# Python try/except can catch). services/shap_service.py now computes the
# same Tree SHAP values using XGBoost's own built-in
# Booster.predict(..., pred_contribs=True) instead, which needs no
# extra explainer object and has no numba dependency.


def get_absorption_data():
    """Load and cache Layer 3 absorption analysis outputs."""
    if "t_df" not in _cache:
        print("[Loader] Loading absorption data...")
        try:
            _cache["t_df"] = pd.read_csv(TIME_TO_ABSORPTION)
            _cache["N_df"] = pd.read_csv(FUNDAMENTAL_MATRIX, index_col=0)
            # Optional file in config; safe load if present
            try:
                _cache["B_df"] = pd.read_csv(ABSORPTION_PROBS, index_col=0)
            except Exception:
                _cache["B_df"] = None

            print("[Loader] Absorption data loaded.")
        except FileNotFoundError:
            print("[Loader] WARNING: Absorption files not found. Run Notebook 04 first.")
            _cache["t_df"] = None
            _cache["N_df"] = None
            _cache["B_df"] = None

    return _cache["t_df"], _cache["N_df"]


def clear_cache():
    """Clear all cached data — useful after retraining models."""
    _cache.clear()
    print("[Loader] Cache cleared.")


def safe_label_transform(series: pd.Series, encoder, fallback=None) -> np.ndarray:
    """
    Safely transform categorical labels with LabelEncoder.
    Unknown values are mapped to fallback (default: first known class).
    """
    known = set(encoder.classes_)

    if fallback is None:
        fallback = encoder.classes_[0]

    cleaned = (
        series.astype(str)
        .fillna(str(fallback))
        .apply(lambda x: x if x in known else fallback)
    )
    return encoder.transform(cleaned)


def encode_features(row_dict: dict, encoders: dict, features: list) -> np.ndarray:
    """
    Encode a single account row for ML prediction.
    Returns shape (1, n_features).
    """
    df = pd.DataFrame([row_dict]).copy()

    df["segment_enc"] = safe_label_transform(df["segment"], encoders["segment"])
    df["loan_type_enc"] = safe_label_transform(df["loan_type"], encoders["loan_type"])

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df[features] = df[features].fillna(0)
    return df[features].to_numpy()


def encode_dataframe(df: pd.DataFrame, encoders: dict, features: list) -> np.ndarray:
    """
    Encode an entire DataFrame for batch ML prediction.
    Returns shape (n_rows, n_features).
    """
    df = df.copy()

    df["segment_enc"] = safe_label_transform(df["segment"], encoders["segment"])
    df["loan_type_enc"] = safe_label_transform(df["loan_type"], encoders["loan_type"])

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df[features] = df[features].fillna(0)
    return df[features].to_numpy()