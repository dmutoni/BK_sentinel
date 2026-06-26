"""
BK Sentinel — Overview Router
Portfolio snapshot and risk trend endpoints.
"""

from fastapi import APIRouter, Depends
from database.loader import get_portfolio_df
from middleware.auth import get_current_user
from config import STATES

router = APIRouter(prefix="/api/overview", tags=["Overview"])


@router.get("/snapshot")
def portfolio_snapshot(
    month:   str,
    segment: str = "All Segments",
    user:    dict = Depends(get_current_user),
):
    """
    Return the risk state distribution for a given month and segment.
    Used for the KPI cards on the Overview page.
    """
    # Guard against empty month
    if not month or month.strip() == "":
        return {
            "total":  0,
            "month":  "",
            "counts": {s: 0 for s in STATES},
            "pct":    {s: 0.0 for s in STATES},
        }

    df = get_portfolio_df()
    d  = df[df["observation_month"] == month]

    if segment != "All Segments":
        d = d[d["segment"] == segment]

    counts = d["risk_state"].value_counts().to_dict()
    total  = len(d)

    return {
        "total":  total,
        "month":  month,
        "counts": {s: int(counts.get(s, 0)) for s in STATES},
        "pct":    {s: round(counts.get(s, 0) / total * 100, 1) if total else 0 for s in STATES},
    }


@router.get("/trend")
def risk_trend(user: dict = Depends(get_current_user)):
    """
    Return monthly risk state counts across all 16 months.
    Used for the risk trend table on the Overview page.
    """
    df  = get_portfolio_df()
    grp = df.groupby(["observation_month", "risk_state"]).size().unstack(fill_value=0)
    grp = grp.reindex(columns=STATES, fill_value=0)

    return {
        "months": grp.index.tolist(),
        "series": {s: grp[s].tolist() for s in STATES if s in grp.columns},
    }


@router.get("/segments")
def get_segments(user: dict = Depends(get_current_user)):
    """Return available customer segments."""
    df   = get_portfolio_df()
    segs = [s for s in df["segment"].unique() if s != "UNKNOWN"]
    return ["All Segments"] + sorted(segs)


@router.get("/months")
def get_months(user: dict = Depends(get_current_user)):
    """Return all available observation months in order."""
    df = get_portfolio_df()
    return sorted(df["observation_month"].unique().tolist())
