"""
BK Sentinel — FastAPI Backend
Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle, json, warnings
from numpy.linalg import matrix_power

warnings.filterwarnings("ignore")

app = FastAPI(title="BK Sentinel API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── simple token auth ─────────────────────────
USERS = {
    "analyst":  {"password": "bk2026",  "name": "Credit Analyst",   "role": "Credit Analyst"},
    "manager":  {"password": "bk2026",  "name": "Portfolio Manager", "role": "Portfolio Manager"},
    "denyse":   {"password": "alu2026", "name": "Denyse Mutoni",     "role": "Researcher"},
}
TOKENS = {}  # token -> username

STATES    = ["Low", "Medium", "High", "Default"]
TRANSIENT = ["Low", "Medium", "High"]

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

# ── data loading ──────────────────────────────
_cache = {}

def get_df():
    if "df" not in _cache:
        _cache["df"] = pd.read_csv("bk_sentinel_verified.csv", low_memory=False)
    return _cache["df"]

def get_P():
    if "P" not in _cache:
        df = pd.read_csv("bk_transition_matrix.csv", index_col=0)
        _cache["P"] = df.reindex(index=STATES, columns=STATES)
    return _cache["P"]

def get_model():
    if "model" not in _cache:
        _cache["model"]    = pickle.load(open("bk_best_model.pkl","rb"))
        _cache["encoders"] = pickle.load(open("bk_label_encoders.pkl","rb"))
        _cache["features"] = json.load(open("bk_feature_cols.json"))
    return _cache["model"], _cache["encoders"], _cache["features"]

def get_absorption():
    if "t_df" not in _cache:
        try:
            _cache["t_df"] = pd.read_csv("bk_time_to_absorption.csv")
            _cache["N_df"] = pd.read_csv("bk_fundamental_matrix.csv", index_col=0)
        except:
            _cache["t_df"] = None
            _cache["N_df"] = None
    return _cache["t_df"], _cache["N_df"]

def encode_row(row_dict, encoders, features):
    df = pd.DataFrame([row_dict])
    df["segment_enc"]   = encoders["segment"].transform(df["segment"].astype(str).fillna("UNKNOWN"))
    df["loan_type_enc"] = encoders["loan_type"].transform(df["loan_type"].astype(str).fillna("UNKNOWN"))
    for c in features:
        if c not in df.columns: df[c] = 0
    df[features] = df[features].fillna(0)
    return df[features].values[0]

def get_shap_values(model, X_row, features, pred_class):
    try:
        import shap
        exp  = shap.TreeExplainer(model)
        sv   = exp.shap_values(X_row.reshape(1,-1))
        vals = sv[pred_class][0] if isinstance(sv,list) else \
               (np.array(sv)[0,:,pred_class] if np.array(sv).ndim==3 else np.array(sv)[0])
        idx  = np.argsort(np.abs(vals))[::-1][:5]
        return [{"feature": FEATURE_LABELS.get(features[i], features[i]),
                 "value":   round(float(X_row[i]), 2),
                 "shap":    round(float(vals[i]), 4)} for i in idx]
    except:
        return []

# ── auth helper ───────────────────────────────
security = HTTPBearer(auto_error=False)

def current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(401, "Not authenticated")
    token = credentials.credentials
    if token not in TOKENS:
        raise HTTPException(401, "Invalid token")
    return TOKENS[token]

# ── auth endpoints ────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = USERS.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(401, "Invalid credentials")
    import secrets
    token = secrets.token_hex(32)
    TOKENS[token] = {"username": req.username, **user}
    return {"token": token, "name": user["name"], "role": user["role"]}

@app.post("/api/auth/logout")
def logout(user = Depends(current_user)):
    # remove token
    token_to_remove = None
    for t, u in TOKENS.items():
        if u["username"] == user["username"]:
            token_to_remove = t
            break
    if token_to_remove:
        del TOKENS[token_to_remove]
    return {"message": "Logged out"}

@app.get("/api/auth/me")
def me(user = Depends(current_user)):
    return {"username": user["username"], "name": user["name"], "role": user["role"]}

# ── meta endpoints ────────────────────────────
@app.get("/api/meta/months")
def get_months(user = Depends(current_user)):
    df = get_df()
    return sorted(df["observation_month"].unique().tolist())

@app.get("/api/meta/segments")
def get_segments(user = Depends(current_user)):
    df = get_df()
    segs = [s for s in df["segment"].unique() if s != "UNKNOWN"]
    return ["All Segments"] + sorted(segs)

# ── overview endpoints ────────────────────────
@app.get("/api/overview/snapshot")
def portfolio_snapshot(month: str, segment: str = "All Segments", user = Depends(current_user)):
    df = get_df()
    d  = df[df["observation_month"] == month]
    if segment != "All Segments":
        d = d[d["segment"] == segment]
    counts = d["risk_state"].value_counts().to_dict()
    total  = len(d)
    return {
        "total":  total,
        "month":  month,
        "counts": {s: int(counts.get(s, 0)) for s in STATES},
        "pct":    {s: round(counts.get(s,0)/total*100,1) if total else 0 for s in STATES}
    }

@app.get("/api/overview/trend")
def risk_trend(user = Depends(current_user)):
    df  = get_df()
    grp = df.groupby(["observation_month","risk_state"]).size().unstack(fill_value=0)
    grp = grp.reindex(columns=STATES, fill_value=0)
    return {
        "months": grp.index.tolist(),
        "series": {s: grp[s].tolist() for s in STATES if s in grp.columns}
    }

# ── transition endpoints ──────────────────────
@app.get("/api/transition/matrix")
def transition_matrix(user = Depends(current_user)):
    P_df = get_P()
    return {
        "states": STATES,
        "matrix": P_df.values.tolist(),
        "key_rates": {
            "low_retention":   round(float(P_df.loc["Low","Low"]),4),
            "medium_recovery": round(float(P_df.loc["Medium","Low"]),4),
            "high_to_default": round(float(P_df.loc["High","Default"]),4),
            "default_persist": round(float(P_df.loc["Default","Default"]),4),
        }
    }

@app.get("/api/transition/forecast")
def transition_forecast(horizon: int = 6, user = Depends(current_user)):
    P   = get_P().values
    Pn  = matrix_power(P, horizon)
    return {
        "horizon":  horizon,
        "states":   STATES,
        "matrix":   [[round(float(v),4) for v in row] for row in Pn],
        "high_to_default": round(float(Pn[STATES.index("High"), STATES.index("Default")]),4)
    }

@app.get("/api/transition/portfolio-forecast")
def portfolio_evolution(month: str, segment: str = "All Segments",
                        horizon: int = 12, user = Depends(current_user)):
    df  = get_df()
    d   = df[df["observation_month"] == month]
    if segment != "All Segments":
        d = d[d["segment"] == segment]
    P      = get_P().values
    counts = d["risk_state"].value_counts()
    total  = counts.sum()
    if total == 0:
        return {}
    pi0 = np.array([counts.get(s,0)/total for s in STATES])
    result = {}
    for n in range(horizon+1):
        pi = pi0 if n==0 else pi0 @ matrix_power(P, n)
        result[n] = {s: round(float(pi[i]*100),2) for i,s in enumerate(STATES)}
    return result

# ── absorption endpoints ──────────────────────
@app.get("/api/absorption/summary")
def absorption_summary(user = Depends(current_user)):
    t_df, N_df = get_absorption()
    if t_df is None:
        raise HTTPException(404, "Absorption data not found. Run Notebook 04 first.")
    rows = []
    for state in TRANSIENT:
        row = t_df[t_df["risk_state"] == state].iloc[0]
        rows.append({
            "state":          state,
            "probability":    round(float(row["absorption_probability"]),4),
            "months":         round(float(row["expected_months_to_default"]),1),
            "years":          round(float(row["expected_years_to_default"]),1),
        })
    N_list = N_df.values.tolist() if N_df is not None else []
    return {
        "summary": rows,
        "fundamental_matrix": {"states": TRANSIENT, "values": N_list}
    }

# ── watchlist endpoints ───────────────────────
@app.get("/api/watchlist")
def watchlist(
    month: str,
    segment: str = "All Segments",
    filter_type: str = "all_deteriorating",
    horizon: int = 1,
    page: int = 1,
    page_size: int = 25,
    user = Depends(current_user)
):
    df = get_df()
    d  = df[df["observation_month"] == month].copy().reset_index(drop=True)
    if segment != "All Segments":
        d = d[d["segment"] == segment].reset_index(drop=True)

    model, encoders, features = get_model()
    d["segment_enc"]   = encoders["segment"].transform(d["segment"].astype(str).fillna("UNKNOWN"))
    d["loan_type_enc"] = encoders["loan_type"].transform(d["loan_type"].astype(str).fillna("UNKNOWN"))
    for c in features:
        if c not in d.columns: d[c] = 0
    d[features] = d[features].fillna(0)
    X      = d[features].values
    proba  = model.predict_proba(X)
    preds  = model.predict(X)
    rl     = {0:"Low",1:"Medium",2:"High",3:"Default"}
    d["pred"]      = [rl[p] for p in preds]
    d["p_default"] = proba[:,3]
    d["p_high"]    = proba[:,2]
    d["score"]     = d["p_default"] + d["p_high"]

    state_order = {"Low":0,"Medium":1,"High":2,"Default":3}
    d["cur_r"]  = d["risk_state"].map(state_order)
    d["pred_r"] = d["pred"].map(state_order)
    d["worse"]  = d["pred_r"] > d["cur_r"]

    P = get_P().values
    Pn = matrix_power(P, horizon)
    d["h_risk"] = d["risk_state"].apply(
        lambda s: float(Pn[STATES.index(s),2]+Pn[STATES.index(s),3]) if s in STATES else 0)
    d["combined"] = d["score"]*0.6 + d["h_risk"]*0.4

    if filter_type == "about_to_default":
        # High accounts predicted to Default next month — most urgent
        wl = d[(d["risk_state"]=="High") & (d["pred"]=="Default")]
    elif filter_type == "high_to_high":
        # High staying High — still in danger zone
        wl = d[(d["risk_state"]=="High") & (d["pred"]=="High")]
    elif filter_type == "medium_to_high":
        # Medium jumping to High — fast deterioration
        wl = d[(d["risk_state"]=="Medium") & (d["pred"]=="High")]
    elif filter_type == "medium_to_default":
        # Medium jumping directly to Default
        wl = d[(d["risk_state"]=="Medium") & (d["pred"]=="Default")]
    elif filter_type == "low_to_medium":
        # Low starting to slip — earliest warning
        wl = d[(d["risk_state"]=="Low") & (d["pred"]=="Medium")]
    elif filter_type == "low_to_high":
        # Low jumping straight to High — big jump
        wl = d[(d["risk_state"]=="Low") & (d["pred"].isin(["High","Default"]))]
    elif filter_type == "fast_deterioration":
        # Any non-defaulted account jumping to High or Default
        wl = d[(d["risk_state"]!="Default") & (d["pred"].isin(["High","Default"]))]
    elif filter_type == "all_deteriorating":
        # Any account getting worse
        wl = d[(d["risk_state"]!="Default") & (d["worse"])]
    else:
        # Full performing portfolio
        wl = d[d["risk_state"]!="Default"]

    wl = wl.sort_values("combined", ascending=False)
    total = len(wl)
    start = (page-1)*page_size
    end   = start+page_size
    page_data = wl.iloc[start:end]

    cols = [c for c in ["loan_id","customer_id","segment","risk_state","pred",
                         "days_in_arrears","instalments_in_arrears","p_default","h_risk","score"]
            if c in page_data.columns]
    records = page_data[cols].copy()
    records["transition"]   = records["risk_state"] + " → " + records["pred"]
    records["horizon_risk"] = (records["h_risk"]*100).round(1).astype(str) + "%"

    # summary counts
    summary = {s: int((d["pred"]==s).sum()) for s in STATES}

    return {
        "total":       total,
        "page":        page,
        "page_size":   page_size,
        "total_pages": max(1,(total+page_size-1)//page_size),
        "summary":     summary,
        "records":     records.drop(columns=["h_risk","score"], errors="ignore").fillna("").to_dict(orient="records")
    }

# ── account lookup endpoint ───────────────────
@app.get("/api/account/{loan_id}")
def account_lookup(loan_id: str, user = Depends(current_user)):
    df = get_df()
    mask = (df["loan_id"].astype(str).str.contains(loan_id, case=False, na=False) |
            df["customer_id"].astype(str).str.contains(loan_id, case=False, na=False))
    result = df[mask].sort_values("observation_month")
    if len(result) == 0:
        raise HTTPException(404, f"No account found for: {loan_id}")

    latest  = result.iloc[-1]
    history = result[["observation_month","risk_state","days_in_arrears",
                       "instalments_in_arrears","repayment_ratio","penal_interest",
                       "suspended_interest"]].fillna(0)

    # prediction
    model, encoders, features = get_model()
    X_row = encode_row(latest.to_dict(), encoders, features)
    proba = model.predict_proba(X_row.reshape(1,-1))[0]
    pred  = int(model.predict(X_row.reshape(1,-1))[0])
    rl    = {0:"Low",1:"Medium",2:"High",3:"Default"}

    shap_reasons = get_shap_values(model, X_row, features, pred)

    return {
        "account": {
            "loan_id":          str(latest.get("loan_id","")),
            "customer_id":      str(latest.get("customer_id","")),
            "segment":          str(latest.get("segment","")),
            "loan_type":        str(latest.get("loan_type","")),
            "last_month":       str(latest.get("observation_month","")),
            "risk_state":       str(latest.get("risk_state","")),
            "days_in_arrears":  float(latest.get("days_in_arrears",0) or 0),
            "instalments_in_arrears": float(latest.get("instalments_in_arrears",0) or 0),
            "repayment_ratio":  float(latest.get("repayment_ratio",0) or 0),
            "penal_interest":   float(latest.get("penal_interest",0) or 0),
            "suspended_interest": float(latest.get("suspended_interest",0) or 0),
            "principal_balance": float(latest.get("principal_balance",0) or 0),
            "disbursed_amount": float(latest.get("disbursed_amount_lcy",0) or 0),
            "interest_rate":    float(latest.get("interest_rate",0) or 0),
            "loan_term_months": float(latest.get("loan_term_months",0) or 0),
        },
        "prediction": {
            "predicted_state": rl[pred],
            "probabilities":   {rl[i]: round(float(p),4) for i,p in enumerate(proba)},
            "shap_reasons":    shap_reasons,
        },
        "history": history.to_dict(orient="records")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
