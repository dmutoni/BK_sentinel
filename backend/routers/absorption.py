"""
BK Sentinel — Absorption Analysis Router
Layer 3: Absorbing Markov chain analysis endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from database.loader import get_absorption_data, get_P_numpy
from services.markov import compute_absorption_analysis
from middleware.auth import get_current_user
from config import TRANSIENT

router = APIRouter(prefix="/api/absorption", tags=["Absorption Analysis"])


@router.get("/summary")
def absorption_summary(user: dict = Depends(get_current_user)):
    """
    Return Layer 3 absorption analysis results.

    Includes:
    - Long-run Default absorption probability per transient state
    - Expected months to Default per transient state
    - Fundamental matrix N = (I - Q)^-1
    """
    t_df, N_df = get_absorption_data()

    if t_df is None:
        raise HTTPException(
            status_code=404,
            detail="Absorption data not found. Please run Notebook 04 first."
        )

    summary = []
    for state in TRANSIENT:
        row = t_df[t_df["risk_state"] == state].iloc[0]
        summary.append({
            "state":       state,
            "probability": round(float(row["absorption_probability"]), 4),
            "months":      round(float(row["expected_months_to_default"]), 1),
            "years":       round(float(row["expected_years_to_default"]), 1),
        })

    N_values = N_df.values.tolist() if N_df is not None else []

    return {
        "summary": summary,
        "fundamental_matrix": {
            "states": TRANSIENT,
            "values": N_values,
        },
        "interpretation": {
            "absorbing_state":  "Default",
            "self_transition":  round(float(get_P_numpy()[3, 3]), 4),
            "criterion":        "Default qualifies as absorbing (self-transition >= 0.95)",
        }
    }


@router.get("/recompute")
def recompute_absorption(user: dict = Depends(get_current_user)):
    """
    Recompute absorption analysis live from the current transition matrix.
    Useful for verifying Notebook 04 results or after retraining.
    """
    P = get_P_numpy()
    result = compute_absorption_analysis(P)

    return {
        "transient_states": result["transient_states"],
        "expected_months_to_default": {
            s: round(float(result["t"][i]), 1)
            for i, s in enumerate(result["transient_states"])
        },
        "fundamental_matrix": {
            "states": result["transient_states"],
            "values": result["N"].tolist(),
        },
        "absorption_probabilities": {
            s: round(float(result["B"][i, 0]), 4)
            for i, s in enumerate(result["transient_states"])
        },
    }
