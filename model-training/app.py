import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pickle, json, warnings, time
from numpy.linalg import matrix_power, inv

warnings.filterwarnings('ignore')

st.set_page_config(page_title="BK Sentinel", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

# ── BK colour palette ─────────────────────────
BK_NAVY   = "#003B7A"
BK_BLUE   = "#0066CC"
BK_LIGHT  = "#E8F1FB"
BK_WHITE  = "#FFFFFF"
BK_GRAY   = "#F4F7FA"
BK_BORDER = "#D1E3F6"

STATE_COLORS = {
    "Low":     "#10B981",
    "Medium":  "#F59E0B",
    "High":    "#EF4444",
    "Default": "#7C3AED",
}
STATE_BG = {
    "Low":     "#ECFDF5",
    "Medium":  "#FFFBEB",
    "High":    "#FEF2F2",
    "Default": "#F5F3FF",
}

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
    "loan_term_months":        "Loan term (months)",
    "loan_age_months":         "Loan age (months)",
    "segment_enc":             "Customer segment",
}

# ── global CSS ────────────────────────────────
st.markdown(f"""
<style>
  .main {{ background:{BK_GRAY}; }}
  .block-container {{ padding:2rem 2rem 1rem; }}
  section[data-testid="stSidebar"] {{ background:{BK_NAVY}; }}
  section[data-testid="stSidebar"] * {{ color:white !important; }}
  #MainMenu,footer,header {{ visibility:hidden; }}

  h1,h2,h3 {{ color:{BK_NAVY}; }}

  /* metric card */
  .mc {{
    background:{BK_WHITE};
    border-radius:12px;
    padding:18px 20px;
    border-left:4px solid {BK_BLUE};
    box-shadow:0 1px 4px rgba(0,60,120,.07);
    height:100%;
  }}
  .mc-label {{
    font-size:11px; font-weight:700;
    color:#64748b; text-transform:uppercase;
    letter-spacing:.06em; margin-bottom:6px;
  }}
  .mc-value {{ font-size:26px; font-weight:800; color:{BK_NAVY}; }}
  .mc-sub   {{ font-size:12px; color:#94a3b8; margin-top:4px; }}

  /* state badge */
  .sb {{ display:inline-block; padding:4px 12px; border-radius:20px;
         font-size:13px; font-weight:700; }}

  /* info box */
  .infobox {{
    background:{BK_LIGHT}; border-left:4px solid {BK_BLUE};
    border-radius:8px; padding:14px 18px;
    font-size:13px; color:{BK_NAVY}; margin:12px 0;
  }}

  /* page header */
  .pg-title {{
    font-size:26px; font-weight:800;
    color:{BK_NAVY}; margin-bottom:2px;
  }}
  .pg-sub {{
    font-size:14px; color:#64748b; margin-bottom:24px;
  }}

  /* section card */
  .section-card {{
    background:{BK_WHITE}; border-radius:12px;
    padding:20px 24px; margin-bottom:16px;
    box-shadow:0 1px 4px rgba(0,60,120,.06);
    border:1px solid {BK_BORDER};
  }}

  /* account header */
  .acct-hdr {{
    background:linear-gradient(135deg,{BK_NAVY} 0%,{BK_BLUE} 100%);
    border-radius:12px; padding:24px 28px;
    color:white; margin-bottom:20px;
  }}

  /* history table alt rows */
  .stDataFrame {{ border-radius:10px; overflow:hidden; }}
  div[data-testid="stDataFrame"] {{ border:1px solid {BK_BORDER}; border-radius:10px; }}

  /* pill */
  .pill-red  {{ display:inline-block;padding:4px 12px;border-radius:20px;
                font-size:12px;font-weight:600;background:#FEF2F2;color:#991B1B;margin:2px 3px; }}
  .pill-grn  {{ display:inline-block;padding:4px 12px;border-radius:20px;
                font-size:12px;font-weight:600;background:#ECFDF5;color:#065F46;margin:2px 3px; }}
  .pill-amb  {{ display:inline-block;padding:4px 12px;border-radius:20px;
                font-size:12px;font-weight:600;background:#FFFBEB;color:#92400E;margin:2px 3px; }}

  button[kind="primary"] {{ background:{BK_BLUE} !important; border-color:{BK_BLUE} !important; }}
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────
def state_badge(state):
    c  = STATE_COLORS.get(state, "#64748b")
    bg = STATE_BG.get(state, "#F1F5F9")
    return f'<span class="sb" style="background:{bg};color:{c};">{state}</span>'

def mc(label, value, sub="", color=BK_BLUE):
    return f"""<div class="mc" style="border-color:{color}">
      <div class="mc-label">{label}</div>
      <div class="mc-value" style="color:{color}">{value}</div>
      <div class="mc-sub">{sub}</div>
    </div>"""

# ── data loaders ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("bk_sentinel_verified.csv", low_memory=False)

@st.cache_data(show_spinner=False)
def load_P():
    df = pd.read_csv("bk_transition_matrix.csv", index_col=0)
    return df.reindex(index=STATES, columns=STATES)

@st.cache_data(show_spinner=False)
def load_absorption():
    try:
        return (pd.read_csv("bk_time_to_absorption.csv"),
                pd.read_csv("bk_fundamental_matrix.csv", index_col=0))
    except:
        return None, None

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model    = pickle.load(open("bk_best_model.pkl","rb"))
        encoders = pickle.load(open("bk_label_encoders.pkl","rb"))
        features = json.load(open("bk_feature_cols.json"))
        return model, encoders, features
    except:
        return None, None, None

def encode_df(df, encoders, features):
    df = df.copy()
    df["segment_enc"]   = encoders["segment"].transform(df["segment"].astype(str).fillna("UNKNOWN"))
    df["loan_type_enc"] = encoders["loan_type"].transform(df["loan_type"].astype(str).fillna("UNKNOWN"))
    for c in features:
        if c not in df.columns:
            df[c] = 0
    df[features] = df[features].fillna(0)
    return df

def get_shap(model, X_row, features, pred_class):
    try:
        import shap
        exp = shap.TreeExplainer(model)
        sv  = exp.shap_values(X_row.reshape(1,-1))
        if isinstance(sv, list):
            vals = sv[pred_class][0]
        else:
            arr  = np.array(sv)
            vals = arr[0,:,pred_class] if arr.ndim==3 else arr[0]
        idx  = np.argsort(np.abs(vals))[::-1][:5]
        out  = []
        for i in idx:
            fname = FEATURE_LABELS.get(features[i], features[i])
            fval  = X_row[i]
            sign  = vals[i]
            out.append({"feature":fname,"value":fval,"shap":sign})
        return out
    except:
        return []

# ── sidebar ───────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:16px 0 8px">
      <div style="font-size:28px">🏦</div>
      <div style="font-size:18px;font-weight:800;letter-spacing:.02em">BK Sentinel</div>
      <div style="font-size:11px;opacity:.7;margin-top:2px">Credit Risk Monitoring System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("Navigation",
        ["🔍  Account Lookup",
         "📋  Risk Watchlist",
         "📊  Portfolio Overview"],
        label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="font-size:12px;font-weight:700;opacity:.7;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">Filters</div>', unsafe_allow_html=True)

    with st.spinner(""):
        df_all = load_data()

    months    = sorted(df_all["observation_month"].unique())
    sel_month = st.selectbox("Month", months, index=len(months)-1)
    segs      = ["All Segments"] + sorted([s for s in df_all["segment"].unique() if s!="UNKNOWN"])
    sel_seg   = st.selectbox("Segment", segs)

    df_now = df_all[df_all["observation_month"]==sel_month].copy().reset_index(drop=True)
    if sel_seg != "All Segments":
        df_now = df_now[df_now["segment"]==sel_seg].reset_index(drop=True)

    st.markdown("---")
    st.markdown('<div style="font-size:12px;font-weight:700;opacity:.7;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">Portfolio Snapshot</div>', unsafe_allow_html=True)

    total = len(df_now)
    for s in STATES:
        n   = (df_now["risk_state"]==s).sum()
        pct = n/total*100 if total else 0
        c   = STATE_COLORS[s]
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:5px 0;border-bottom:1px solid rgba(255,255,255,.08)'>"
            f"<span style='font-size:13px'>{s}</span>"
            f"<span><b style='color:{c}'>{n:,}</b>"
            f"<span style='opacity:.5;font-size:11px'> {pct:.0f}%</span></span></div>",
            unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;opacity:.4;margin-top:8px;text-align:right'>{total:,} total</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:11px;opacity:.4;text-align:center">ALU Capstone 2026 · Denyse M.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════
# PAGE 1 ─ ACCOUNT LOOKUP
# ════════════════════════════════════════════
if "Lookup" in page:
    st.markdown('<div class="pg-title">🔍 Account Lookup</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Search any loan account to see its full risk profile, payment behaviour, and AI prediction.</div>', unsafe_allow_html=True)

    model, encoders, features = load_model()

    # search bar
    col_q, col_btn = st.columns([5,1])
    with col_q:
        query = st.text_input("", placeholder="Enter Loan ID or Customer ID — e.g. LN123456",
                              label_visibility="collapsed")
    with col_btn:
        go = st.button("Search", type="primary", use_container_width=True)

    if query:
        with st.spinner("Searching..."):
            time.sleep(0.2)
            mask   = (df_all["loan_id"].astype(str).str.contains(query, case=False, na=False) |
                      df_all["customer_id"].astype(str).str.contains(query, case=False, na=False))
            result = df_all[mask].sort_values("observation_month")

        if len(result) == 0:
            st.warning(f"No account found for **{query}**. Check the ID and try again.")
        else:
            latest = result.iloc[-1]
            state  = latest.get("risk_state","N/A")
            col = STATE_COLORS.get(state, BK_BLUE)

            # account header
            st.markdown(f"""
            <div class="acct-hdr">
              <div style="font-size:11px;opacity:.6;margin-bottom:2px;text-transform:uppercase;letter-spacing:.05em">Loan Account</div>
              <div style="font-size:24px;font-weight:800;margin-bottom:14px">{latest.get("loan_id","")}</div>
              <div style="display:flex;gap:36px;flex-wrap:wrap">
                <div><div style="font-size:11px;opacity:.6">Customer ID</div>
                     <div style="font-size:14px;font-weight:600">{str(latest.get("customer_id",""))[:24]}</div></div>
                <div><div style="font-size:11px;opacity:.6">Segment</div>
                     <div style="font-size:14px;font-weight:600">{latest.get("segment","")}</div></div>
                <div><div style="font-size:11px;opacity:.6">Loan Type</div>
                     <div style="font-size:14px;font-weight:600">{latest.get("loan_type","")}</div></div>
                <div><div style="font-size:11px;opacity:.6">Last Observed</div>
                     <div style="font-size:14px;font-weight:600">{latest.get("observation_month","")}</div></div>
                <div><div style="font-size:11px;opacity:.6">Current State</div>
                     <div style="margin-top:2px">{state_badge(state)}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # payment behaviour
            st.markdown("#### Payment Behaviour")
            dpd  = latest.get("days_in_arrears",0) or 0
            inst = latest.get("instalments_in_arrears",0) or 0
            rr   = latest.get("repayment_ratio",0) or 0
            pi   = latest.get("penal_interest",0) or 0
            si   = latest.get("suspended_interest",0) or 0

            c1,c2,c3,c4,c5 = st.columns(5)
            dpd_c  = STATE_COLORS["Default"] if dpd>90 else STATE_COLORS["High"] if dpd>30 else STATE_COLORS["Medium"] if dpd>0 else STATE_COLORS["Low"]
            rr_c   = STATE_COLORS["Low"] if rr>0.8 else STATE_COLORS["Medium"] if rr>0.5 else STATE_COLORS["High"]
            c1.markdown(mc("Days in Arrears", int(dpd), "BNR: >90 = Default", dpd_c), unsafe_allow_html=True)
            c2.markdown(mc("Instalments in Arrears", int(inst), "Consecutive missed payments", STATE_COLORS["High"] if inst>2 else STATE_COLORS["Medium"] if inst>0 else STATE_COLORS["Low"]), unsafe_allow_html=True)
            c3.markdown(mc("Repayment Ratio", f"{rr:.0%}", "Amount paid vs scheduled", rr_c), unsafe_allow_html=True)
            c4.markdown(mc("Penal Interest", f"RWF {pi:,.0f}", "Non-zero signals payment distress", STATE_COLORS["High"] if pi>0 else STATE_COLORS["Low"]), unsafe_allow_html=True)
            c5.markdown(mc("Suspended Interest", f"RWF {si:,.0f}", "Only applied at DPD > 90", STATE_COLORS["Default"] if si>0 else STATE_COLORS["Low"]), unsafe_allow_html=True)

            # AI prediction
            st.markdown("#### AI Risk Prediction — Next Month")
            if model is None:
                st.error("Model not loaded.")
            else:
                with st.spinner("Computing prediction..."):
                    df_enc = encode_df(pd.DataFrame([latest]), encoders, features)
                    X_row  = df_enc[features].values[0]
                    proba  = model.predict_proba(X_row.reshape(1,-1))[0]
                    pred   = int(model.predict(X_row.reshape(1,-1))[0])
                    rl     = {0:"Low",1:"Medium",2:"High",3:"Default"}
                    pred_state = rl[pred]

                pc1,pc2,pc3,pc4 = st.columns(4)
                for col_w, lbl, prob in zip([pc1,pc2,pc3,pc4], STATES, proba):
                    c_val = STATE_COLORS[lbl]
                    border = f"3px solid {c_val}" if lbl==pred_state else f"1px solid {BK_BORDER}"
                    col_w.markdown(
                        f"""<div class="section-card" style="border:{border};text-align:center;padding:16px">
                          <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em">{lbl}</div>
                          <div style="font-size:32px;font-weight:800;color:{c_val};margin:6px 0">{prob:.0%}</div>
                          {"<div style='font-size:11px;color:#64748b'>← predicted</div>" if lbl==pred_state else ""}
                        </div>""", unsafe_allow_html=True)

                # SHAP explanation
                st.markdown("#### Why is this account predicted as " + pred_state + "?")
                with st.spinner("Computing explanation..."):
                    reasons = get_shap(model, X_row, features, pred)

                if reasons:
                    st.markdown('<div class="infobox">These are the features that most influenced the prediction. Features in red are pushing the account toward higher risk. Features in green are stabilising it.</div>', unsafe_allow_html=True)
                    for r in reasons:
                        pred_is_risky = pred_state in ["High","Default"]
                        pushes_toward = r["shap"] > 0
                        if pred_is_risky and pushes_toward:
                            pill = "pill-red"
                            msg  = f"↑ increasing risk  ({r['feature']} = {r['value']:,.1f})"
                        elif pred_is_risky and not pushes_toward:
                            pill = "pill-grn"
                            msg  = f"↓ stabilising  ({r['feature']} = {r['value']:,.1f})"
                        elif not pred_is_risky and pushes_toward:
                            pill = "pill-grn"
                            msg  = f"↑ supporting healthy state  ({r['feature']} = {r['value']:,.1f})"
                        else:
                            pill = "pill-amb"
                            msg  = f"↓ minor factor  ({r['feature']} = {r['value']:,.1f})"
                        st.markdown(f'<span class="{pill}">{msg}</span>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                else:
                    st.info("Install the shap library to see explanations: pip install shap")

            # payment history
            st.markdown("#### Payment History — Last 16 Months")
            hist = result[["observation_month","risk_state","days_in_arrears",
                           "instalments_in_arrears","repayment_ratio","penal_interest"]].copy()
            hist.columns = ["Month","State","Days in Arrears","Instalments in Arrears","Repayment Ratio","Penal Interest"]
            hist["Repayment Ratio"] = hist["Repayment Ratio"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
            hist["Penal Interest"]  = hist["Penal Interest"].apply(lambda x: f"RWF {x:,.0f}" if pd.notna(x) else "0")
            st.dataframe(hist.style.format({"Days in Arrears":"{:.0f}","Instalments in Arrears":"{:.0f}"}),
                         use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
# PAGE 2 ─ RISK WATCHLIST
# ════════════════════════════════════════════
elif "Watchlist" in page:
    st.markdown('<div class="pg-title">📋 Risk Watchlist</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Accounts currently performing but predicted to deteriorate. Already-defaulted accounts are excluded — the bank can no longer act on those.</div>', unsafe_allow_html=True)

    model, encoders, features = load_model()
    P_df = load_P()
    P    = P_df.values

    if model is None:
        st.error("Model files not found. Ensure bk_best_model.pkl is in the same folder.")
    else:
        with st.spinner("Running AI predictions on all accounts..."):
            df_enc = encode_df(df_now, encoders, features)
            X      = df_enc[features].values
            proba  = model.predict_proba(X)
            preds  = model.predict(X)
            rl     = {0:"Low",1:"Medium",2:"High",3:"Default"}
            df_enc["pred"]      = [rl[p] for p in preds]
            df_enc["p_default"] = proba[:,3]
            df_enc["p_high"]    = proba[:,2]
            df_enc["score"]     = df_enc["p_default"] + df_enc["p_high"]

        # summary row
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(mc("Predicted Default", f"{(df_enc['pred']=='Default').sum():,}", "next month", STATE_COLORS["Default"]), unsafe_allow_html=True)
        c2.markdown(mc("Predicted High",    f"{(df_enc['pred']=='High').sum():,}",    "next month", STATE_COLORS["High"]),    unsafe_allow_html=True)
        c3.markdown(mc("Predicted Medium",  f"{(df_enc['pred']=='Medium').sum():,}",  "next month", STATE_COLORS["Medium"]),  unsafe_allow_html=True)
        c4.markdown(mc("Predicted Healthy", f"{(df_enc['pred']=='Low').sum():,}",     "next month", STATE_COLORS["Low"]),     unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # filters
        state_order = {"Low":0,"Medium":1,"High":2,"Default":3}
        df_enc["cur_rank"]  = df_enc["risk_state"].map(state_order)
        df_enc["pred_rank"] = df_enc["pred"].map(state_order)
        df_enc["worsening"] = df_enc["pred_rank"] > df_enc["cur_rank"]

        fc1, fc2 = st.columns([3,2])
        with fc1:
            filt = st.selectbox("Filter", [
                "About to Default — High accounts predicted to Default next month",
                "Fast deterioration — any account jumping to High or Default",
                "Early warning — Low accounts starting to slip",
                "All deteriorating — any account getting worse",
                "Full portfolio — all non-defaulted accounts",
            ])
        with fc2:
            horizon = st.selectbox("Markov risk horizon",
                [1,2,3,6,12], format_func=lambda x: f"{x} month{'s' if x>1 else ''} from now")

        # markov risk at horizon
        Pn = matrix_power(P, horizon)
        def markov_risk(s):
            i = STATES.index(s) if s in STATES else 0
            return float(Pn[i,2] + Pn[i,3])
        df_enc["markov_risk"]    = df_enc["risk_state"].apply(markov_risk)
        df_enc["combined_score"] = df_enc["score"]*0.6 + df_enc["markov_risk"]*0.4

        if "About to Default" in filt:
            wl = df_enc[(df_enc["risk_state"]=="High") & (df_enc["pred"]=="Default")].copy()
        elif "Fast deterioration" in filt:
            wl = df_enc[(df_enc["risk_state"]!="Default") & (df_enc["pred"].isin(["High","Default"]))].copy()
        elif "Early warning" in filt:
            wl = df_enc[(df_enc["risk_state"]=="Low") & (df_enc["worsening"])].copy()
        elif "All deteriorating" in filt:
            wl = df_enc[(df_enc["risk_state"]!="Default") & (df_enc["worsening"])].copy()
        else:
            wl = df_enc[df_enc["risk_state"]!="Default"].copy()

        wl = wl.sort_values("combined_score", ascending=False).reset_index(drop=True)
        wl["Transition"]          = wl["risk_state"] + " → " + wl["pred"]
        wl[f"Risk ({horizon}m)"]  = wl["markov_risk"].apply(lambda x: f"{x*100:.1f}%")

        if len(wl)==0:
            st.markdown(f'<div class="infobox">No accounts match this filter in <b>{sel_month}</b>. This means the model finds no accounts making this specific transition next month. Try <b>All deteriorating</b> or change the month.</div>', unsafe_allow_html=True)
        else:
            shap_on = st.toggle("Show top 3 risk drivers per account (slower)", value=False)
            if shap_on:
                with st.spinner(f"Computing SHAP for top 100 accounts..."):
                    try:
                        import shap
                        exp  = shap.TreeExplainer(model)
                        X_wl = wl[features].fillna(0).values[:100]
                        sv   = exp.shap_values(X_wl)
                        sv_d = sv[3] if isinstance(sv,list) else np.array(sv)[:,:,3]
                        top3 = []
                        for row_sv in sv_d:
                            idx = np.argsort(np.abs(row_sv))[::-1][:3]
                            top3.append(" · ".join([FEATURE_LABELS.get(features[i],features[i]) for i in idx]))
                        wl.loc[wl.index[:100],"Top 3 Drivers"] = top3
                    except:
                        st.info("SHAP unavailable.")

            show = [c for c in ["loan_id","customer_id","segment","Transition",
                                  f"Risk ({horizon}m)","days_in_arrears",
                                  "instalments_in_arrears","p_default","Top 3 Drivers"]
                    if c in wl.columns]
            rename_map = {
                "loan_id":"Loan ID","customer_id":"Customer ID","segment":"Segment",
                "days_in_arrears":"Days in Arrears",
                "instalments_in_arrears":"Instalments in Arrears",
                "p_default":"P(Default)"
            }
            wl_disp = wl[show].rename(columns=rename_map)

            # pagination
            PAGE = 25
            total_r = len(wl_disp)
            total_p = max(1,(total_r+PAGE-1)//PAGE)
            if "pg" not in st.session_state: st.session_state.pg = 0
            st.session_state.pg = max(0, min(st.session_state.pg, total_p-1))
            s = st.session_state.pg * PAGE
            e = min(s+PAGE, total_r)

            st.markdown(f"**{total_r:,} accounts** · sorted by risk score")
            st.dataframe(
                wl_disp.iloc[s:e].style.format({"P(Default)":"{:.3f}","Days in Arrears":"{:.0f}","Instalments in Arrears":"{:.0f}"}),
                use_container_width=True, height=460, hide_index=True)

            p1,p2,p3,p4 = st.columns([1,1,3,1])
            with p1:
                if st.button("◀ Prev", disabled=st.session_state.pg==0, key="prev"):
                    st.session_state.pg -= 1; st.rerun()
            with p2:
                if st.button("Next ▶", disabled=st.session_state.pg>=total_p-1, key="nxt"):
                    st.session_state.pg += 1; st.rerun()
            with p3:
                st.markdown(f"<div style='padding:8px 0;color:#64748b;font-size:13px'>Page {st.session_state.pg+1} of {total_p} · {s+1}–{e} of {total_r:,} accounts</div>", unsafe_allow_html=True)
            with p4:
                st.download_button("⬇ Export CSV", data=wl_disp.to_csv(index=False),
                    file_name=f"watchlist_{sel_month}.csv", mime="text/csv", key="dl")

# ════════════════════════════════════════════
# PAGE 3 ─ PORTFOLIO OVERVIEW
# ════════════════════════════════════════════
elif "Portfolio" in page:
    st.markdown('<div class="pg-title">📊 Portfolio Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Transition matrix, multi-step forecasts, and long-run absorption analysis for bank management.</div>', unsafe_allow_html=True)

    P_df      = load_P()
    P         = P_df.values
    t_df, N_df = load_absorption()

    # ── key rates ──
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(mc("Low → Low (monthly)", f"{P_df.loc['Low','Low']*100:.1f}%",    "Portfolio health",          STATE_COLORS["Low"]),     unsafe_allow_html=True)
    c2.markdown(mc("Medium → Low recovery", f"{P_df.loc['Medium','Low']*100:.1f}%","Intervention effective here",STATE_COLORS["Medium"]),  unsafe_allow_html=True)
    c3.markdown(mc("High → Default monthly",f"{P_df.loc['High','Default']*100:.1f}%","⚠ Critical signal",       STATE_COLORS["High"]),    unsafe_allow_html=True)
    c4.markdown(mc("Default persistence",   f"{P_df.loc['Default','Default']*100:.1f}%","Near-absorbing state",  STATE_COLORS["Default"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── matrices ──
    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Monthly transition matrix**")
        fig,ax = plt.subplots(figsize=(5,3.8))
        sns.heatmap(P_df, annot=True, fmt=".3f", cmap="Blues",
                    vmin=0, vmax=1, linewidths=.8, ax=ax,
                    annot_kws={"size":11,"weight":"bold"})
        ax.set_xlabel("To State (next month)", fontsize=9)
        ax.set_ylabel("From State (this month)", fontsize=9)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Multi-step forecast**")
        horizon = st.slider("Months ahead", 1, 12, 6)
        with st.spinner("Computing..."):
            Pn    = matrix_power(P, horizon)
            Pn_df = pd.DataFrame(Pn, index=STATES, columns=STATES)
        fig,ax = plt.subplots(figsize=(5,3.8))
        sns.heatmap(Pn_df, annot=True, fmt=".3f", cmap="Blues",
                    vmin=0, vmax=1, linewidths=.8, ax=ax,
                    annot_kws={"size":11,"weight":"bold"})
        ax.set_xlabel(f"To State (month t+{horizon})", fontsize=9)
        ax.set_ylabel("From State (now)", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.set_title(f"{horizon}-month forecast", fontsize=10, fontweight="bold", color=BK_NAVY)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        phd = Pn_df.loc["High","Default"]
        st.markdown(f'<div class="infobox">A High-risk account today has a <b>{phd*100:.1f}%</b> probability of reaching Default within {horizon} months.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── portfolio evolution ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Portfolio risk distribution — next 12 months**")
    counts = df_now["risk_state"].value_counts()
    tot    = counts.sum()
    pi0    = np.array([counts.get(s,0)/tot for s in STATES])
    evo    = {s:[] for s in STATES}
    for n in range(13):
        pi = pi0 if n==0 else pi0 @ matrix_power(P,n)
        for i,s in enumerate(STATES):
            evo[s].append(pi[i]*100)
    fig,ax = plt.subplots(figsize=(11,3.8))
    for s in STATES:
        ax.plot(range(13),evo[s],marker="o",markersize=4,linewidth=2.5,
                label=s,color=STATE_COLORS[s])
    ax.set_xlabel("Months from now",fontsize=10)
    ax.set_ylabel("Portfolio share (%)",fontsize=10)
    ax.legend(fontsize=10,loc="upper right")
    ax.grid(True,alpha=.15)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(0,12)
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    rows=[]
    for n in [0,3,6,12]:
        pi = pi0 if n==0 else pi0@matrix_power(P,n)
        row={"Horizon":"Now" if n==0 else f"In {n} months"}
        for i,s in enumerate(STATES): row[s]=f"{pi[i]*100:.1f}%"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("Horizon"),use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── absorption ──
    if t_df is not None:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Layer 3 — Expected time to Default (Kemeny & Snell, 1960)**")
        a1,a2,a3 = st.columns(3)
        for col_w, state in zip([a1,a2,a3], TRANSIENT):
            row   = t_df[t_df["risk_state"]==state].iloc[0]
            mnths = row["expected_months_to_default"]
            yrs   = row["expected_years_to_default"]
            col_w.markdown(mc(f"{state} risk account",
                f"{mnths:.1f} months",
                f"{yrs:.1f} years until Default without intervention",
                STATE_COLORS[state]), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(6,3.5))
        sns.heatmap(N_df.astype(float), annot=True, fmt=".1f", cmap="Blues",
                    linewidths=.8, ax=ax, annot_kws={"size":12,"weight":"bold"})
        ax.set_title("Fundamental Matrix N = (I − Q)⁻¹\nExpected months in each state before Default",
                     fontweight="bold", fontsize=10, color=BK_NAVY)
        ax.set_xlabel("State visited"); ax.set_ylabel("Starting state")
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)