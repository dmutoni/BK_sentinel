"""
BK Sentinel — Regenerate Synthetic Data
Run this as a cell in Notebook 01, replacing the data generation section.
This version has more realistic deterioration patterns:
  - More accounts transitioning High → Default
  - More Medium → High transitions
  - More diverse January 2026 snapshot
"""

import pandas as pd
import numpy as np
import hashlib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── configuration ─────────────────────────────────────────────
N_ACCOUNTS  = 4244
N_MONTHS    = 16
START_DATE  = datetime(2024, 10, 1)

# ── BNR DPD thresholds ────────────────────────────────────────
def dpd_to_state(dpd):
    if dpd == 0:             return 'Low'
    elif dpd <= 30:          return 'Medium'
    elif dpd <= 90:          return 'High'
    else:                    return 'Default'

def dpd_to_code(dpd):
    return {'Low':0,'Medium':1,'High':2,'Default':3}[dpd_to_state(dpd)]

# ── loan segments ─────────────────────────────────────────────
SEGMENTS = {
    'RETAIL':      0.887,
    'SME':         0.048,
    'AGRICULTURE': 0.019,
    'CORPORATE':   0.007,
    'UNKNOWN':     0.039,
}

LOAN_TYPES = ['Personal Loan','Business Loan','Mortgage','Agricultural Loan','Staff Loan']

# ── risk trajectories — adjusted for more defaults ────────────
# Old: stable_low=45%, deteriorate=15%, recover=10%, slow_degrade=10%
# New: stable_low=20%, deteriorate=30%, recover=5%, slow_degrade=20%
TRAJECTORIES = {
    'stable_low':     0.20,  # stays healthy throughout (was 45%)
    'deteriorate':    0.30,  # clear Low→Medium→High→Default path (was 15%)
    'slow_degrade':   0.20,  # slowly gets worse over 16 months (was 10%)
    'volatile':       0.10,  # bounces between states (was 8%)
    'stable_medium':  0.05,  # stays Medium with minor variation
    'stable_high':    0.05,  # stuck in High risk the whole time (new)
    'stable_default': 0.05,  # already in Default, stays there (was 4%)
    'recover':        0.03,  # improves over time (was 10%)
    'partial_degrade':0.02,  # goes to Medium and stays
}

def generate_dpd_sequence(trajectory, n_months=16):
    """Generate a 16-month DPD sequence based on trajectory type."""
    seq = []

    if trajectory == 'stable_low':
        # Healthy accounts — occasional 1-5 day delays but always catches up
        for _ in range(n_months):
            seq.append(np.random.choice([0, 0, 0, 0, 1, 2, 3], p=[0.7, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03]))

    elif trajectory == 'deteriorate':
        # Classic deterioration: starts fine, deteriorates, ends in Default
        speed = np.random.choice(['fast', 'medium', 'slow'], p=[0.4, 0.4, 0.2])
        if speed == 'fast':
            # Default by month 8-10
            breakpoint = np.random.randint(4, 9)
            for m in range(n_months):
                if m < breakpoint:
                    seq.append(np.random.randint(0, 5))
                elif m < breakpoint + 3:
                    seq.append(np.random.randint(15, 45))
                elif m < breakpoint + 5:
                    seq.append(np.random.randint(45, 91))
                else:
                    seq.append(np.random.randint(91, 200))
        elif speed == 'medium':
            # Default by month 10-13
            breakpoint = np.random.randint(6, 11)
            for m in range(n_months):
                if m < breakpoint:
                    seq.append(np.random.randint(0, 15))
                elif m < breakpoint + 2:
                    seq.append(np.random.randint(20, 60))
                elif m < breakpoint + 3:
                    seq.append(np.random.randint(60, 91))
                else:
                    seq.append(np.random.randint(91, 180))
        else:
            # Default in last 2-3 months
            breakpoint = np.random.randint(10, 14)
            for m in range(n_months):
                if m < breakpoint:
                    seq.append(np.random.randint(0, 30))
                elif m < breakpoint + 1:
                    seq.append(np.random.randint(30, 91))
                else:
                    seq.append(np.random.randint(91, 150))

    elif trajectory == 'slow_degrade':
        # Gradually worsening — may or may not hit Default
        start_dpd = np.random.randint(0, 10)
        increment = np.random.uniform(3, 8)
        for m in range(n_months):
            dpd = start_dpd + m * increment + np.random.normal(0, 3)
            seq.append(max(0, int(dpd)))

    elif trajectory == 'volatile':
        # Bounces between states — pays then misses then pays
        dpd = 0
        for m in range(n_months):
            action = np.random.choice(['pay', 'miss', 'partial'], p=[0.45, 0.35, 0.20])
            if action == 'pay':
                dpd = max(0, dpd - np.random.randint(15, 35))
            elif action == 'miss':
                dpd += np.random.randint(20, 40)
            else:
                dpd = max(0, dpd - np.random.randint(5, 15))
            seq.append(min(dpd, 150))

    elif trajectory == 'stable_medium':
        # Watchlist account — slightly behind but not getting worse
        for _ in range(n_months):
            seq.append(np.random.randint(5, 30))

    elif trajectory == 'stable_high':
        # Stuck in High risk — DPD between 31-90 throughout
        base = np.random.randint(35, 70)
        for _ in range(n_months):
            seq.append(max(31, min(90, base + np.random.randint(-8, 8))))

    elif trajectory == 'stable_default':
        # Already defaulted — DPD > 90 throughout
        base = np.random.randint(95, 250)
        for _ in range(n_months):
            seq.append(base + np.random.randint(0, 20))

    elif trajectory == 'recover':
        # Starts in trouble, gets better
        start_dpd = np.random.randint(60, 150)
        for m in range(n_months):
            dpd = start_dpd - m * np.random.uniform(5, 12) + np.random.normal(0, 3)
            seq.append(max(0, int(dpd)))

    elif trajectory == 'partial_degrade':
        # Goes to Medium/High and stays there
        for m in range(n_months):
            if m < 4:
                seq.append(np.random.randint(0, 10))
            else:
                seq.append(np.random.randint(15, 35))

    return seq

# ── generate loan account features ───────────────────────────
def generate_account_features():
    segment    = np.random.choice(list(SEGMENTS.keys()), p=list(SEGMENTS.values()))
    loan_type  = np.random.choice(LOAN_TYPES)
    loan_term  = np.random.choice([12, 24, 36, 48, 60, 72, 84], p=[0.10, 0.20, 0.25, 0.20, 0.15, 0.06, 0.04])
    disbursed  = np.random.lognormal(mean=13.5, sigma=1.2)
    disbursed  = max(500_000, min(disbursed, 50_000_000))
    rate       = np.random.uniform(0.12, 0.24)
    crb        = np.random.choice([0, 1, 2, 3], p=[0.75, 0.15, 0.07, 0.03])
    return segment, loan_type, loan_term, disbursed, rate, crb

# ── hash function for anonymization ──────────────────────────
def make_hash(val):
    return 'ANON_' + hashlib.sha256(str(val).encode()).hexdigest()[:8].upper()

# ── main generation loop ──────────────────────────────────────
print("Generating BK Sentinel synthetic panel dataset...")
print(f"Accounts: {N_ACCOUNTS} | Months: {N_MONTHS} | Start: {START_DATE.strftime('%Y-%m')}")
print()

traj_names = list(TRAJECTORIES.keys())
traj_probs = list(TRAJECTORIES.values())

rows = []

for acc_idx in range(N_ACCOUNTS):
    if acc_idx % 500 == 0:
        print(f"  Generating account {acc_idx+1}/{N_ACCOUNTS}...")

    # assign trajectory
    traj = np.random.choice(traj_names, p=traj_probs)

    # loan features
    segment, loan_type, loan_term, disbursed, rate, crb = generate_account_features()

    # loan start offset (how many months into the loan at Oct 2024)
    loan_age_at_start = np.random.randint(0, max(1, loan_term - N_MONTHS))

    # anonymized IDs
    raw_loan_id     = f"LN{np.random.randint(100000, 999999)}"
    raw_customer_id = f"CUS{np.random.randint(100000, 999999)}"
    loan_id         = raw_loan_id
    customer_id     = make_hash(raw_customer_id)

    # generate DPD sequence
    dpd_seq = generate_dpd_sequence(traj, N_MONTHS)

    # monthly features
    monthly_balance = disbursed
    monthly_instalment = (disbursed * rate / 12) / (1 - (1 + rate/12)**(-loan_term))

    for m_idx, dpd in enumerate(dpd_seq):
        obs_date  = START_DATE + timedelta(days=30 * m_idx)
        obs_month = obs_date.strftime('%Y-%m')

        state = dpd_to_state(dpd)

        # derived financials
        instalments_in_arrears = max(0, int(dpd / 30))
        principal_due = monthly_instalment * instalments_in_arrears
        interest_due  = monthly_instalment * 0.4 * instalments_in_arrears
        penal         = max(0, (dpd - 30) * monthly_balance * 0.0005) if dpd > 30 else 0
        suspended     = monthly_balance * 0.02 if dpd > 90 else 0
        accrued       = monthly_balance * rate / 12
        repayment     = max(0, 1 - (instalments_in_arrears / max(loan_term, 1)))
        balance       = max(0, monthly_balance - monthly_instalment * m_idx * 0.7)
        arrears_ratio = min(1.0, instalments_in_arrears / max(loan_term, 1))
        n_paid        = max(0, loan_age_at_start + m_idx - instalments_in_arrears)
        loan_age      = loan_age_at_start + m_idx

        rows.append({
            'loan_id':                  loan_id,
            'customer_id':              customer_id,
            'observation_month':        obs_month,
            'segment':                  segment,
            'loan_type':                loan_type,
            'loan_term_months':         loan_term,
            'disbursed_amount_lcy':     round(disbursed, 2),
            'interest_rate':            round(rate, 4),
            'days_in_arrears':          dpd,
            'risk_state':               state,
            'risk_state_code':          dpd_to_code(dpd),
            'instalments_in_arrears':   instalments_in_arrears,
            'principal_balance':        round(balance, 2),
            'principal_due':            round(principal_due, 2),
            'interest_due':             round(interest_due, 2),
            'penal_interest':           round(penal, 2),
            'suspended_interest':       round(suspended, 2),
            'accrued_interest':         round(accrued, 2),
            'repayment_ratio':          round(repayment, 4),
            'loan_age_months':          loan_age,
            'arrears_ratio':            round(arrears_ratio, 4),
            'number_instalments_paid':  n_paid,
            'all_crb':                  crb,
            'account_status':           'ACTIVE' if state != 'Default' else 'NPL',
        })

# ── build dataframe ───────────────────────────────────────────
df = pd.DataFrame(rows)
df = df.sort_values(['loan_id','observation_month']).reset_index(drop=True)

print(f"\nDataset generated: {len(df):,} rows × {len(df.columns)} columns")
print(f"Unique accounts:   {df['loan_id'].nunique():,}")
print(f"Months:            {df['observation_month'].nunique()}")

# ── create next_risk_state (target variable) ──────────────────
df['next_risk_state']      = df.groupby('loan_id')['risk_state'].shift(-1)
df['next_risk_state_code'] = df.groupby('loan_id')['risk_state_code'].shift(-1)

# ── distribution check ────────────────────────────────────────
print("\nRisk state distribution (all months):")
print(df['risk_state'].value_counts().to_string())

print("\nJanuary 2026 snapshot:")
jan = df[df['observation_month'] == '2026-01']
print(jan['risk_state'].value_counts().to_string())

print("\nTransition distribution (next_risk_state where not NaN):")
trans = df.dropna(subset=['next_risk_state'])
ct = pd.crosstab(trans['risk_state'], trans['next_risk_state'])
print(ct)

# ── trajectory proportion check ──────────────────────────────
print("\nTrajectory assignments:")
traj_counts = {t: 0 for t in traj_names}
# Approximate from Default state occurrence
total = len(df['loan_id'].unique())
for t, p in zip(traj_names, traj_probs):
    print(f"  {t:<20} ~{int(p*total):,} accounts ({p*100:.0f}%)")

# ── save ──────────────────────────────────────────────────────
df.to_csv('bk_sentinel_verified.csv', index=False)

# save transitions subset
df_trans = df.dropna(subset=['next_risk_state']).copy()
df_trans.to_csv('bk_sentinel_transitions.csv', index=False)

# save transition matrix
counts = pd.crosstab(df_trans['risk_state'], df_trans['next_risk_state'])
STATES = ['Low', 'Medium', 'High', 'Default']
counts = counts.reindex(index=STATES, columns=STATES, fill_value=0)
matrix = counts.div(counts.sum(axis=1), axis=0)
matrix.to_csv('bk_transition_matrix.csv')
counts.to_csv('bk_transition_counts.csv')

print("\n✓ Files saved:")
print("  bk_sentinel_verified.csv")
print("  bk_sentinel_transitions.csv")
print("  bk_transition_matrix.csv")
print("  bk_transition_counts.csv")
print("\nNow re-run Notebooks 02, 03, and 04 in order.")
