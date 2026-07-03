"""
BK Sentinel Backend — Configuration
All paths, constants, and settings live here.
"""

import os
from pathlib import Path

# ── base paths ────────────────────────────────────────────────
# The data directory is one level up from this file (the BK_sentinel folder).
# Override with the DATA_DIR env var if your deployment platform lays files
# out differently (e.g. Railway) — local dev needs no env var at all.
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR.parent / "model-training")))

# ── data file paths ───────────────────────────────────────────
VERIFIED_CSV        = DATA_DIR / "bk_sentinel_verified.csv"
TRANSITIONS_CSV     = DATA_DIR / "bk_sentinel_transitions.csv"
TRANSITION_MATRIX   = DATA_DIR / "bk_transition_matrix.csv"
TRANSITION_COUNTS   = DATA_DIR / "bk_transition_counts.csv"
TIME_TO_ABSORPTION  = DATA_DIR / "bk_time_to_absorption.csv"
FUNDAMENTAL_MATRIX  = DATA_DIR / "bk_fundamental_matrix.csv"
ABSORPTION_PROBS    = DATA_DIR / "bk_absorption_probabilities.csv"

# ── model file paths ──────────────────────────────────────────
MODEL_PKL           = DATA_DIR / "bk_best_model.pkl"
ENCODERS_PKL        = DATA_DIR / "bk_label_encoders.pkl"
FEATURE_COLS_JSON   = DATA_DIR / "bk_feature_cols.json"

# ── risk states ───────────────────────────────────────────────
STATES    = ["Low", "Medium", "High", "Default"]
TRANSIENT = ["Low", "Medium", "High"]

STATE_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Default": 3}

# ── BNR DPD thresholds ────────────────────────────────────────
DPD_LOW     = 0
DPD_MEDIUM  = 30
DPD_HIGH    = 90

# ── auth ──────────────────────────────────────────────────────
USERS = {
    "analyst":  {"password": "bk2026",  "name": "Credit Analyst",   "role": "Credit Analyst"},
    "manager":  {"password": "bk2026",  "name": "Portfolio Manager", "role": "Portfolio Manager"},
    "denyse":   {"password": "alu2026", "name": "Denyse Mutoni",     "role": "Researcher"},
}

# Accounts created via /api/auth/signup are persisted here so they
# survive a server restart. Merged into USERS on startup.
USERS_FILE = BASE_DIR / "bk_users_store.json"

# ── feature labels for SHAP explanations ─────────────────────
FEATURE_LABELS = {
    "days_in_arrears":         "Days in arrears",
    "suspended_interest":      "Suspended interest",
    "arrears_ratio":           "Arrears ratio",
    "instalments_in_arrears":  "Instalments in arrears",
    "penal_interest":          "Penal interest",
    "number_instalments_paid": "Instalments paid",
    "principal_due":           "Principal due",
    "interest_due":            "Interest due",
    "repayment_ratio":         "Repayment ratio",
    "loan_type_enc":           "Loan type",
    "interest_rate":           "Interest rate",
    "all_crb":                 "CRB exposure",
    "principal_balance":       "Principal balance",
    "disbursed_amount_lcy":    "Disbursed amount",
    "accrued_interest":        "Accrued interest",
    "loan_term_months":        "Loan term",
    "loan_age_months":         "Loan age",
    "segment_enc":             "Customer segment",
}

# ── pagination defaults ───────────────────────────────────────
DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE     = 100

# ── SHAP ──────────────────────────────────────────────────────
SHAP_TOP_N_FEATURES = 5

# ── CORS ──────────────────────────────────────────────────────
# Comma-separated list via env var, e.g. on Railway:
#   ALLOWED_ORIGINS=https://your-app.vercel.app,http://localhost:3000
_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")
    if o.strip()
]
