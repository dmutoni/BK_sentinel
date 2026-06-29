"""
BK Sentinel — Watchlist Router
Risk watchlist endpoint with multiple filter types and pagination.
"""

from fastapi import APIRouter, Depends
from database.loader import get_portfolio_df, get_P_numpy
from services.predictor import predict_batch, get_prediction_summary
from services.markov import compute_markov_horizon_risk
from middleware.auth import get_current_user
from config import STATES, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from numpy.linalg import matrix_power
import numpy as np

router = APIRouter(prefix="/api/watchlist", tags=["Risk Watchlist"])


FILTER_DESCRIPTIONS = {
    "about_to_default":   "High accounts with P(Default) > 25%",
    "high_risk_any":      "All currently High-risk accounts",
    "high_to_high":       "High staying High with elevated default risk",
    "medium_to_high":     "Medium predicted to jump to High",
    "medium_elevated":    "Medium with P(Default) > 10%",
    "medium_to_default":  "Medium predicted to jump to Default",
    "low_to_medium":      "Low accounts starting to slip",
    "low_to_high":        "Low predicted to jump to High or Default",
    "fast_deterioration": "Any account jumping to High or Default",
    "all_deteriorating":  "All accounts getting worse",
    "all_performing":     "Full performing portfolio (non-defaulted)",
}


@router.get("")
def watchlist(
    month:       str,
    segment:     str = "All Segments",
    filter_type: str = "all_deteriorating",
    horizon:     int = 1,
    page:        int = 1,
    page_size:   int = DEFAULT_PAGE_SIZE,
    user:        dict = Depends(get_current_user),
):
    # Guard against empty month
    if not month or month.strip() == "":
        return {
            "total": 0, "page": 1, "page_size": page_size,
            "total_pages": 1, "filter": filter_type,
            "description": "", "summary": {s: 0 for s in STATES},
            "records": [],
        }

    page_size = min(page_size, MAX_PAGE_SIZE)

    # Load and filter by snapshot
    df = get_portfolio_df()
    d  = df[df["observation_month"] == month].copy().reset_index(drop=True)

    if segment != "All Segments":
        d = d[d["segment"] == segment].reset_index(drop=True)

    # Guard against empty snapshot
    if len(d) == 0:
        return {
            "total": 0, "page": 1, "page_size": page_size,
            "total_pages": 1, "filter": filter_type,
            "description": FILTER_DESCRIPTIONS.get(filter_type, ""),
            "summary": {s: 0 for s in STATES},
            "records": [],
        }

    # Run ML predictions on the snapshot
    d = predict_batch(d)

    # Apply the selected filter
    filtered = _apply_filter(d, filter_type).copy()

    # Sort by risk score descending (most urgent first)
    filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)

    total       = len(filtered)
    total_pages = max(1, (total + page_size - 1) // page_size)
    start       = (page - 1) * page_size
    end         = start + page_size
    page_df     = filtered.iloc[start:end]

    summary = {s: int((filtered["pred"] == s).sum()) for s in STATES}

    records = []
    for _, row in page_df.iterrows():
        records.append({
            "loan_id":        str(row.get("loan_id", "")),
            "customer_id":    str(row.get("customer_id", "")),
            "segment":        str(row.get("segment", "")),
            "risk_state":     str(row.get("risk_state", "")),
            "pred":           str(row.get("pred", "")),
            "days_in_arrears":        int(row.get("days_in_arrears", 0)),
            "instalments_in_arrears": int(row.get("instalments_in_arrears", 0)),
            "principal_balance": float(row.get("principal_balance", 0)),
            "p_default":      round(float(row.get("p_default", 0)), 4),
            "p_high":         round(float(row.get("p_high", 0)), 4),
            "score":          round(float(row.get("score", 0)), 4),
            "is_worse":       bool(row.get("is_worse", False)),
        })

    return {
        "total":       total,
        "page":        page,
        "page_size":   page_size,
        "total_pages": total_pages,
        "filter":      filter_type,
        "description": FILTER_DESCRIPTIONS.get(filter_type, ""),
        "summary":     summary,
        "records":     records,
    }


@router.get("/filters")
def list_filters(user: dict = Depends(get_current_user)):
    """Return all available filter types with descriptions."""
    return [{"id": k, "label": v} for k, v in FILTER_DESCRIPTIONS.items()]


def _apply_filter(d, filter_type: str):
    """Apply the selected watchlist filter to the dataframe."""
    if filter_type == "about_to_default":
        return d[(d["risk_state"] == "High") & (d["p_default"] >= 0.25)]

    elif filter_type == "high_risk_any":
        return d[d["risk_state"] == "High"]

    elif filter_type == "high_to_high":
        return d[(d["risk_state"] == "High") & (d["pred"] == "High") & (d["p_default"] >= 0.10)]

    elif filter_type == "medium_to_high":
        return d[(d["risk_state"] == "Medium") & (d["pred"] == "High")]

    elif filter_type == "medium_elevated":
        return d[(d["risk_state"] == "Medium") & (d["p_default"] >= 0.10)]

    elif filter_type == "medium_to_default":
        return d[(d["risk_state"] == "Medium") & (d["pred"] == "Default")]

    elif filter_type == "low_to_medium":
        return d[(d["risk_state"] == "Low") & (d["pred"] == "Medium")]

    elif filter_type == "low_to_high":
        return d[(d["risk_state"] == "Low") & (d["pred"].isin(["High", "Default"]))]

    elif filter_type == "fast_deterioration":
        return d[(d["risk_state"] != "Default") & (d["pred"].isin(["High", "Default"]))]

    elif filter_type == "all_deteriorating":
        return d[(d["risk_state"] != "Default") & (d["is_worse"])]

    else:  # all_performing
        return d[d["risk_state"] != "Default"]
