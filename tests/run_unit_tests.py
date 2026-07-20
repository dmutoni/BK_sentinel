"""
BK Sentinel — Data Pipeline Unit Tests (Table 9, section 4.3.3)
Run from the repo root:  python3 tests/run_unit_tests.py
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

CSV = Path(__file__).parent.parent / "model-training" / "Bk_panel_data.csv"
df = pd.read_csv(CSV, low_memory=False)
d, s = df["days_in_arrears"], df["risk_state"]

print("BK Sentinel — Data Pipeline Unit Tests (notebook 01)")
print(f"Run: {datetime.now():%Y-%m-%d %H:%M}  |  Panel: {len(df):,} records, "
      f"{df['loan_id'].nunique():,} accounts, {df['observation_month'].nunique()} months")
print("=" * 74)

def check(name, violations, note=""):
    tag = "PASS   " if violations == 0 else "FLAGGED"
    print(f"[{tag}] {name:<52s} {violations:>6,}{('  ' + note) if note else ''}")

check("DPD = 0  =>  state = Low",      int(((d == 0) & (s != "Low")).sum()))
check("DPD 1-30  =>  state = Medium",  int((d.between(1, 30) & (s != "Medium")).sum()))
check("DPD 31-90  =>  state = High",   int((d.between(31, 90) & (s != "High")).sum()))
check("DPD > 90  =>  state = Default", int(((d > 90) & (s != "Default")).sum()))
check("suspended_interest > 0 only when DPD > 90",
      int(((df["suspended_interest"] > 0) & (d <= 90)).sum()))
check("Negative days_in_arrears", int((d < 0).sum()))
check("Loan term within 1-360 months",
      int((~df["loan_term_months"].between(1, 360)).sum()))

n_acc = df["loan_id"].nunique()
full = int((df.groupby("loan_id")["observation_month"].nunique() == 16).sum())
tag = "PASS   " if full == n_acc else "FLAGGED"
print(f"[{tag}] Panel continuity (present all 16 months)          {full:,} of {n_acc:,}")

dup = int(df.duplicated(subset=["loan_id", "observation_month"]).sum())
check("Duplicate loan-month pairs", dup, "-> documented, retained")

pct_scale = int((df["interest_rate"] > 50).sum())
check("Interest rate on decimal scale (0-50%)", pct_scale, "-> unit fix applied")

critical = ["customer_id", "branch_name", "loan_type", "disbursed_amount_lcy"]
nulls = int(df[critical].isna().any(axis=1).sum())
print("-" * 74)
print(f"Null values in critical columns: {nulls} records removed ({nulls/len(df):.2%})")
print(f"Verified panel: {len(df) - nulls:,} records")
