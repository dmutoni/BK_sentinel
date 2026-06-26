"""
BK Sentinel — Account Lookup Router
Single account risk profile, prediction, and payment history.
"""

from fastapi import APIRouter, Depends, HTTPException
from database.loader import get_portfolio_df
from services.predictor import predict_single
from services.shap_service import explain_prediction
from middleware.auth import get_current_user

router = APIRouter(prefix="/api/account", tags=["Account Lookup"])


@router.get("/{query}")
def account_lookup(
    query: str,
    user:  dict = Depends(get_current_user),
):
    """
    Look up a loan account by Loan ID or Customer ID.

    Returns:
    - Full account profile (financial metrics)
    - AI prediction for next month with SHAP explanation
    - 16-month payment history
    """
    df = get_portfolio_df()

    # Search by loan_id or customer_id
    mask = (
        df["loan_id"].astype(str).str.contains(query, case=False, na=False) |
        df["customer_id"].astype(str).str.contains(query, case=False, na=False)
    )
    result = df[mask].sort_values("observation_month")

    if len(result) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No account found for '{query}'. Check the Loan ID or Customer ID."
        )

    # Use the most recent month for prediction
    latest = result.iloc[-1]

    # Run ML prediction
    prediction = predict_single(latest.to_dict())

    # Generate SHAP explanation for the predicted class
    shap_reasons = explain_prediction(
        prediction["X_row"],
        prediction["pred_class_idx"]
    )

    # Payment history (all months, most recent first)
    history_cols = [
        "observation_month", "risk_state", "days_in_arrears",
        "instalments_in_arrears", "repayment_ratio",
        "penal_interest", "suspended_interest",
    ]
    history = result[history_cols].fillna(0).to_dict(orient="records")

    return {
        "account": {
            "loan_id":               str(latest.get("loan_id", "")),
            "customer_id":           str(latest.get("customer_id", "")),
            "segment":               str(latest.get("segment", "")),
            "loan_type":             str(latest.get("loan_type", "")),
            "last_month":            str(latest.get("observation_month", "")),
            "risk_state":            str(latest.get("risk_state", "")),
            "days_in_arrears":       float(latest.get("days_in_arrears", 0) or 0),
            "instalments_in_arrears":float(latest.get("instalments_in_arrears", 0) or 0),
            "repayment_ratio":       float(latest.get("repayment_ratio", 0) or 0),
            "penal_interest":        float(latest.get("penal_interest", 0) or 0),
            "suspended_interest":    float(latest.get("suspended_interest", 0) or 0),
            "principal_balance":     float(latest.get("principal_balance", 0) or 0),
            "disbursed_amount":      float(latest.get("disbursed_amount_lcy", 0) or 0),
            "interest_rate":         float(latest.get("interest_rate", 0) or 0),
            "loan_term_months":      float(latest.get("loan_term_months", 0) or 0),
        },
        "prediction": {
            "predicted_state": prediction["predicted_state"],
            "probabilities":   prediction["probabilities"],
            "shap_reasons":    shap_reasons,
        },
        "history": history,
    }
