"""
BK Sentinel Backend — Configuration
All paths, constants, and settings live here.
"""

from pathlib import Path

# ── base paths ────────────────────────────────────────────────
# The data directory is one level up from this file (the BK_sentinel folder)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "model-training"  # adjust this to point to your BK_sentinel folder

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
