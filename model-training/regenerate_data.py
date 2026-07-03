"""
BK Sentinel — Synthetic Panel Dataset Generator (v4 — Markov + Resilience)
============================================================================
Same Markov-banded structure as v3, with one addition: each loan gets a
hidden 'payment_resilience' trait (drawn once, never written to the output)
that genuinely steers the Medium-and-High transition dice rolls, instead of
every account in a DPD band getting an identical, undifferentiated roll.

Resilience leaves a noisy, realistic fingerprint on `all_crb` — exactly like
a real bank, which never observes a customer's true financial resilience
directly, only an imperfect bureau-flag proxy for it.

Why this matters: in v3, EVERY Medium account with DPD 23-29 received the
exact same p=[0.15, 0.60, 0.25, 0.00] regardless of any other column, which
means no feature could ever predict which specific account crosses into
High — the ground truth itself was generated independent of all features.
v4 fixes this at the source while preserving the same population-level
aggregate transition rates (resilience ~ Uniform(0,1) has mean 0.5, and
every modulation formula below reproduces the original v3 base rate
exactly at r=0.5).

Run standalone:
    python3 regenerate_data_v4.py
Then re-run notebooks 01, 02, 03, 04 in order — all four depend on this file.
"""

import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

N_ACCOUNTS = 4244
N_MONTHS   = 16
START_DATE = datetime(2024, 10, 1)

STATES = ['Low', 'Medium', 'High', 'Default']


def next_state(current: str, dpd: int, resilience: float) -> str:
    """
    Sample next risk state given current state, current DPD, AND the
    loan's hidden resilience trait (0 = fragile, 1 = resilient).

    Each band's probabilities are modulated so that:
      - at resilience = 0.5 (the population mean), the result is IDENTICAL
        to the original undifferentiated v3 rates
      - resilient accounts (r -> 1) are pulled toward recovery
      - fragile accounts  (r -> 0) are pulled toward deterioration
    This makes resilience a genuine causal driver of the outcome, not a
    label leaked in after the fact.
    """
    if current == 'Low':
        if dpd == 0:
            base = [0.943, 0.055, 0.002, 0.000]
        elif dpd <= 1:
            base = [0.845, 0.150, 0.005, 0.000]
        else:
            base = [0.690, 0.300, 0.010, 0.000]
        # Low is largely untouched by resilience — the jump out of Low is rare
        # and already well-predicted by DPD itself; keep this band as-is.
        return np.random.choice(STATES, p=base)

    elif current == 'Medium':
        if dpd <= 14:
            base = [0.200, 0.800, 0.000, 0.000]
        elif dpd <= 22:
            base = [0.180, 0.720, 0.100, 0.000]
        else:
            base = [0.150, 0.600, 0.250, 0.000]

        p_low  = min(0.97, base[0] * (2 * resilience))
        p_high = min(0.97, base[2] * (2 * (1 - resilience)))
        p_default = base[3]
        p_med  = max(0.0, 1.0 - p_low - p_high - p_default)
        probs = np.array([p_low, p_med, p_high, p_default])
        probs = probs / probs.sum()  # guard against float drift
        return np.random.choice(STATES, p=probs)

    elif current == 'High':
        if dpd <= 55:
            base = [0.000, 0.080, 0.860, 0.060]
        elif dpd <= 75:
            base = [0.000, 0.050, 0.720, 0.230]
        else:
            base = [0.000, 0.020, 0.420, 0.560]

        p_med_recover = min(0.97, base[1] * (2 * resilience))
        p_default     = min(0.97, base[3] * (2 * (1 - resilience)))
        p_high_stay   = max(0.0, 1.0 - p_med_recover - p_default)
        probs = np.array([0.0, p_med_recover, p_high_stay, p_default])
        probs = probs / probs.sum()
        return np.random.choice(STATES, p=probs)

    else:  # Default — unchanged, matches the documented ~98% persistence
        return np.random.choice(STATES, p=[0.000, 0.000, 0.020, 0.980])


def dpd_for_state(state: str) -> int:
    """Sample a realistic DPD value within the state's BNR range."""
    if state == 'Low':
        # BNR rule: Low = exactly 0 days in arrears. (v3 previously allowed
        # DPD 1-3 here 30% of the time, which by strict BNR rule should be
        # Medium, not Low — fixed.)
        return 0
    elif state == 'Medium':
        return int(np.random.randint(5, 30))
    elif state == 'High':
        return int(np.random.randint(32, 90))
    else:  # Default
        return int(np.random.randint(91, 280))


def dpd_to_state(dpd: int) -> str:
    if dpd == 0:     return 'Low'
    elif dpd <= 30:  return 'Medium'
    elif dpd <= 90:  return 'High'
    else:            return 'Default'


def dpd_to_code(dpd: int) -> int:
    return STATES.index(dpd_to_state(dpd))


def sample_crb(resilience: float) -> int:
    """
    all_crb (credit bureau flag count) as a NOISY, imperfect proxy for the
    hidden resilience trait — exactly like a real bank, which sees bureau
    flags but never the customer's true underlying financial resilience.
    Low resilience shifts probability mass toward more flags, but the
    relationship is deliberately noisy, not deterministic.
    """
    base = np.array([0.75, 0.15, 0.07, 0.03])
    fragility = 1.0 - resilience          # 0 = resilient, 1 = fragile
    shift = fragility * 0.30
    adj = base + np.array([-shift, shift * 0.40, shift * 0.40, shift * 0.20])
    adj = np.clip(adj, 0.01, None)
    adj = adj / adj.sum()
    return int(np.random.choice([0, 1, 2, 3], p=adj))


# ── Initial state distribution (Oct 2024) ─────────────────────────────────────
INIT_DIST = np.array([0.72, 0.10, 0.09, 0.09])

# ── Loan meta distributions ───────────────────────────────────────────────────
SEGMENTS = {
    'RETAIL':      0.887,
    'SME':         0.048,
    'AGRICULTURE': 0.019,
    'CORPORATE':   0.007,
    'UNKNOWN':     0.039,
}
LOAN_TYPES = ['Personal Loan', 'Business Loan', 'Mortgage', 'Agricultural Loan', 'Staff Loan']


def make_hash(val: str) -> str:
    return 'ANON_' + hashlib.sha256(val.encode()).hexdigest()[:8].upper()


# ── Main generation ────────────────────────────────────────────────────────────
print("Generating BK Sentinel synthetic panel dataset v4 (Markov + Resilience)...")
print(f"Accounts: {N_ACCOUNTS} | Months: {N_MONTHS} | Start: {START_DATE.strftime('%Y-%m')}")
print(f"Target transitions: Low→Med 6%, Med→High 11%, High→Default 22%, Default 98%")
print(f"NEW: per-loan resilience trait modulates Medium/High transitions; proxied via all_crb")

rows = []

for acc_idx in range(N_ACCOUNTS):
    if acc_idx % 800 == 0:
        print(f"  Account {acc_idx + 1}/{N_ACCOUNTS}...")

    segment   = np.random.choice(list(SEGMENTS.keys()), p=list(SEGMENTS.values()))
    loan_type = np.random.choice(LOAN_TYPES)
    loan_term = int(np.random.choice([12,24,36,48,60,72,84], p=[0.10,0.20,0.25,0.20,0.15,0.06,0.04]))
    disbursed = float(max(500_000, min(np.random.lognormal(13.5, 1.2), 50_000_000)))

    # Hidden per-loan trait. Never written to output directly — only its
    # noisy fingerprints on all_crb AND interest_rate are observable, same
    # as a real bank: riskier customers get flagged by the bureau AND
    # priced higher at origination, but neither signal alone is perfect.
    resilience = float(np.random.uniform(0, 1))
    crb        = sample_crb(resilience)

    # interest_rate: continuous proxy, much finer resolution than the
    # 4-bucket CRB flag. Lower resilience -> priced higher, with noise.
    rate_center = 0.24 - resilience * 0.12          # 0.12 (resilient) .. 0.24 (fragile)
    rate        = float(np.clip(np.random.normal(rate_center, 0.02), 0.10, 0.26))

    loan_age_at_start = int(np.random.randint(0, max(1, loan_term - N_MONTHS)))
    loan_id     = f"LN{100000 + acc_idx}"
    customer_id = make_hash(f"CUS{200000 + acc_idx}")

    monthly_instalment = (disbursed * rate / 12) / (1 - (1 + rate / 12) ** (-loan_term))

    state = np.random.choice(STATES, p=INIT_DIST)

    for m_idx in range(N_MONTHS):
        y      = START_DATE.year + (START_DATE.month - 1 + m_idx) // 12
        mo     = (START_DATE.month - 1 + m_idx) % 12 + 1
        obs_month = f"{y:04d}-{mo:02d}"

        dpd = dpd_for_state(state)

        instalments_in_arrears = max(0, int(dpd / 30))
        principal_due   = monthly_instalment * instalments_in_arrears
        interest_due    = monthly_instalment * 0.4 * instalments_in_arrears
        penal           = max(0.0, (dpd - 30) * disbursed * 0.0005) if dpd > 30 else 0.0
        suspended       = disbursed * 0.02 if dpd > 90 else 0.0
        accrued         = disbursed * rate / 12
        balance         = max(0.0, disbursed - monthly_instalment * m_idx * 0.7)
        arrears_ratio   = min(1.0, instalments_in_arrears / max(loan_term, 1))
        repayment_ratio = max(0.0, 1.0 - arrears_ratio)
        n_paid          = max(0, loan_age_at_start + m_idx - instalments_in_arrears)

        rows.append({
            'loan_id':                loan_id,
            'customer_id':            customer_id,
            'observation_month':      obs_month,
            'segment':                segment,
            'loan_type':              loan_type,
            'loan_term_months':       loan_term,
            'disbursed_amount_lcy':   round(disbursed, 2),
            'interest_rate':          round(rate, 4),
            'days_in_arrears':        dpd,
            'risk_state':             state,
            'risk_state_code':        STATES.index(state),
            'instalments_in_arrears': instalments_in_arrears,
            'principal_balance':      round(balance, 2),
            'principal_due':          round(principal_due, 2),
            'interest_due':           round(interest_due, 2),
            'penal_interest':         round(penal, 2),
            'suspended_interest':     round(suspended, 2),
            'accrued_interest':       round(accrued, 2),
            'repayment_ratio':        round(repayment_ratio, 4),
            'loan_age_months':        loan_age_at_start + m_idx,
            'arrears_ratio':          round(arrears_ratio, 4),
            'number_instalments_paid': n_paid,
            'all_crb':                crb,
            'account_status':         'ACTIVE' if state != 'Default' else 'NPL',
        })

        state = next_state(state, dpd, resilience)

# ── Build dataframe ────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df = df.sort_values(['loan_id', 'observation_month']).reset_index(drop=True)

df['next_risk_state']      = df.groupby('loan_id')['risk_state'].shift(-1)
df['next_risk_state_code'] = df.groupby('loan_id')['risk_state_code'].shift(-1)

# ── Diagnostics ────────────────────────────────────────────────────────────────
print(f"\nDataset: {len(df):,} rows × {len(df.columns)} columns")
print(f"Unique accounts: {df['loan_id'].nunique():,}  |  Months: {df['observation_month'].nunique()}")

last_month = sorted(df['observation_month'].unique())[-1]
print(f"\n{last_month} snapshot:")
jan = df[df['observation_month'] == last_month]
total_jan = len(jan)
for s in STATES:
    n = int((jan['risk_state'] == s).sum())
    print(f"  {s:<10} {n:>5}  ({n/total_jan*100:.1f}%)")

print("\nEmpirical transition matrix (all 15 pairs):")
trans = df.dropna(subset=['next_risk_state'])
ct    = pd.crosstab(trans['risk_state'], trans['next_risk_state'])
ct    = ct.reindex(index=STATES, columns=STATES, fill_value=0)
prob  = ct.div(ct.sum(axis=1), axis=0).round(4)
print(prob.to_string())

print("\nKey rates:")
print(f"  Low  → Medium:  {prob.loc['Low','Medium']*100:.1f}%   (target ~6%)")
print(f"  Med  → High:    {prob.loc['Medium','High']*100:.1f}%   (target ~11%)")
print(f"  High → Default: {prob.loc['High','Default']*100:.1f}%   (target ~22%)")
print(f"  Default persist:{prob.loc['Default','Default']*100:.1f}%   (target ~98%)")

# ── Save ───────────────────────────────────────────────────────────────────────
df.to_csv('bk_sentinel_verified.csv', index=False)

df_trans = df.dropna(subset=['next_risk_state']).copy()
df_trans.to_csv('bk_sentinel_transitions.csv', index=False)

prob.to_csv('bk_transition_matrix.csv')
ct.to_csv('bk_transition_counts.csv')

print("\nFiles saved:")
print("  bk_sentinel_verified.csv")
print("  bk_sentinel_transitions.csv")
print("  bk_transition_matrix.csv")
print("  bk_transition_counts.csv")
print("\nNext steps: re-run notebooks 01, 02, 03, 04 in order.")
