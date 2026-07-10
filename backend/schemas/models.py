"""
BK Sentinel — Pydantic Schemas
Request and response models for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ── Auth ──────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str = Field(..., example="analyst")
    password: str = Field(..., example="bk2026")


class SignupRequest(BaseModel):
    username: str = Field(..., example="jdoe")
    password: str = Field(..., example="bk2026")
    name:     str = Field(..., example="Jane Doe")


class LoginResponse(BaseModel):
    token: str
    name:  str
    role:  str


class UserResponse(BaseModel):
    username: str
    name:     str
    role:     str


# ── Portfolio Snapshot ────────────────────────────────────────
class SnapshotResponse(BaseModel):
    total:  int
    month:  str
    counts: Dict[str, int]
    pct:    Dict[str, float]


class TrendResponse(BaseModel):
    months: List[str]
    series: Dict[str, List[int]]


# ── Transition Matrix ─────────────────────────────────────────
class KeyRates(BaseModel):
    low_retention:   float
    medium_recovery: float
    high_to_default: float
    default_persist: float


class TransitionMatrixResponse(BaseModel):
    states:    List[str]
    matrix:    List[List[float]]
    key_rates: KeyRates


class ForecastResponse(BaseModel):
    horizon:          int
    states:           List[str]
    matrix:           List[List[float]]
    high_to_default:  float


class PortfolioForecastResponse(BaseModel):
    # Dict mapping horizon (int as str) to state percentages
    data: Dict[str, Dict[str, float]]


# ── Absorption ────────────────────────────────────────────────
class AbsorptionRow(BaseModel):
    state:       str
    probability: float
    months:      float
    years:       float


class FundamentalMatrix(BaseModel):
    states: List[str]
    values: List[List[float]]


class AbsorptionResponse(BaseModel):
    summary:             List[AbsorptionRow]
    fundamental_matrix:  FundamentalMatrix


# ── Watchlist ─────────────────────────────────────────────────
class WatchlistRecord(BaseModel):
    loan_id:               str
    customer_id:           str
    segment:               str
    risk_state:            str
    pred:                  str
    transition:            str
    days_in_arrears:       float
    instalments_in_arrears: float
    p_default:             float
    horizon_risk:          str


class WatchlistSummary(BaseModel):
    Low:     int
    Medium:  int
    High:    int
    Default: int


class WatchlistResponse(BaseModel):
    total:       int
    page:        int
    page_size:   int
    total_pages: int
    summary:     Dict[str, int]
    records:     List[Dict[str, Any]]


# ── Account Lookup ────────────────────────────────────────────
class AccountInfo(BaseModel):
    loan_id:                str
    customer_id:            str
    segment:                str
    loan_type:              str
    last_month:             str
    risk_state:             str
    days_in_arrears:        float
    instalments_in_arrears: float
    repayment_ratio:        float
    penal_interest:         float
    suspended_interest:     float
    principal_balance:      float
    disbursed_amount:       float
    interest_rate:          float
    loan_term_months:       float


class ShapReason(BaseModel):
    feature: str
    value:   float
    shap:    float


class PredictionResult(BaseModel):
    predicted_state: str
    probabilities:   Dict[str, float]
    shap_reasons:    List[ShapReason]


class HistoryRecord(BaseModel):
    observation_month:      str
    risk_state:             str
    days_in_arrears:        float
    instalments_in_arrears: float
    repayment_ratio:        float
    penal_interest:         float
    suspended_interest:     float


class AccountResponse(BaseModel):
    account:    AccountInfo
    prediction: PredictionResult
    history:    List[Dict[str, Any]]
