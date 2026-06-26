"""
BK Sentinel — Transition Matrix Router
Markov chain transition matrix and multi-step forecast endpoints.
"""

from fastapi import APIRouter, Depends
from database.loader import get_transition_matrix, get_P_numpy, get_portfolio_df
from services.markov import compute_forecast, compute_portfolio_evolution
from middleware.auth import get_current_user
from config import STATES

router = APIRouter(prefix="/api/transition", tags=["Transition Matrix"])


@router.get("/matrix")
def transition_matrix(user: dict = Depends(get_current_user)):
    """
    Return the empirical 1-month transition matrix P.
    Estimated from 15 consecutive monthly transition pairs.
    """
    P_df = get_transition_matrix()
    P    = get_P_numpy()

    return {
        "states": STATES,
        "matrix": P_df.values.tolist(),
        "key_rates": {
            "low_retention":   round(float(P_df.loc["Low",     "Low"]),     4),
            "medium_recovery": round(float(P_df.loc["Medium",  "Low"]),     4),
            "high_to_default": round(float(P_df.loc["High",    "Default"]), 4),
            "default_persist": round(float(P_df.loc["Default", "Default"]), 4),
        },
    }


@router.get("/forecast")
def forecast(
    horizon: int = 6,
    user:    dict = Depends(get_current_user),
):
    """
    Return P^n — the n-month transition probability matrix.
    Shows where accounts will likely be in `horizon` months.
    """
    P  = get_P_numpy()
    Pn = compute_forecast(P, horizon)

    return {
        "horizon":         horizon,
        "states":          STATES,
        "matrix":          [[round(float(v), 4) for v in row] for row in Pn],
        "high_to_default": round(float(Pn[STATES.index("High"), STATES.index("Default")]), 4),
    }


@router.get("/portfolio-forecast")
def portfolio_forecast(
    month:   str,
    segment: str = "All Segments",
    horizon: int = 12,
    user:    dict = Depends(get_current_user),
):
    """
    Forecast how the full portfolio risk distribution will shift
    over the next `horizon` months, starting from the given snapshot.
    """
    df = get_portfolio_df()
    d  = df[df["observation_month"] == month]

    if segment != "All Segments":
        d = d[d["segment"] == segment]

    P      = get_P_numpy()
    counts = d["risk_state"].value_counts()

    evolution = compute_portfolio_evolution(P, counts, max_horizon=horizon)

    return evolution
