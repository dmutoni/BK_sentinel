"""
BK Sentinel — Markov Chain Service
Handles all transition matrix and absorption computations.
"""

import numpy as np
import pandas as pd
from numpy.linalg import matrix_power, inv
from config import STATES, TRANSIENT


def compute_forecast(P: np.ndarray, horizon: int) -> np.ndarray:
    """
    Raise the transition matrix P to the power of n.
    Returns P^n — the n-month transition probability matrix.
    """
    return matrix_power(P, horizon)


def compute_portfolio_evolution(
    P: np.ndarray,
    current_distribution: pd.Series,
    max_horizon: int = 12
) -> dict:
    """
    Given the current portfolio distribution (counts per state),
    forecast how the distribution shifts over max_horizon months.

    Returns a dict mapping horizon (int) -> {state: percentage}
    """
    total = current_distribution.sum()
    if total == 0:
        return {}

    pi0 = np.array([current_distribution.get(s, 0) / total for s in STATES])
    result = {}

    for n in range(max_horizon + 1):
        pi_n = pi0 if n == 0 else pi0 @ matrix_power(P, n)
        result[n] = {s: round(float(pi_n[i] * 100), 2) for i, s in enumerate(STATES)}

    return result


def compute_markov_horizon_risk(P: np.ndarray, horizon: int, state: str) -> float:
    """
    Compute the probability of being in High or Default state
    after exactly `horizon` months, starting from `state`.

    Used to generate the horizon risk score for each account.
    """
    if state not in STATES:
        return 0.0
    Pn = matrix_power(P, horizon)
    i = STATES.index(state)
    high_idx    = STATES.index("High")
    default_idx = STATES.index("Default")
    return float(Pn[i, high_idx] + Pn[i, default_idx])


def compute_absorption_analysis(P: np.ndarray) -> dict:
    """
    Perform absorbing Markov chain analysis.
    Computes the fundamental matrix N = (I - Q)^-1 and
    absorption probabilities B = N @ R.

    Returns:
        - N: fundamental matrix (expected months in each transient state)
        - B: absorption probabilities (probability of reaching Default)
        - t: expected total months to Default from each transient state
    """
    t_idx = [STATES.index(s) for s in TRANSIENT]
    a_idx = [STATES.index("Default")]

    Q = P[np.ix_(t_idx, t_idx)]  # transient → transient
    R = P[np.ix_(t_idx, a_idx)]  # transient → absorbing

    I = np.eye(len(TRANSIENT))
    N = inv(I - Q)   # fundamental matrix
    B = N @ R        # absorption probabilities
    t = N.sum(axis=1)  # expected months to absorption

    return {
        "N": N,
        "B": B,
        "t": t,
        "Q": Q,
        "R": R,
        "transient_states": TRANSIENT,
    }


def build_model_transition_matrix(
    df: pd.DataFrame,
    pred_col: str
) -> pd.DataFrame:
    """
    Build a transition matrix from model predictions.
    Used to compare how well each model's predictions
    reproduce the empirical transition dynamics.
    """
    df_t = df.copy()
    df_t["next_pred"] = df_t.groupby("loan_id")[pred_col].shift(-1)
    df_t = df_t.dropna(subset=["next_pred"])

    counts = pd.crosstab(df_t[pred_col], df_t["next_pred"])
    counts = counts.reindex(index=STATES, columns=STATES, fill_value=0)
    matrix = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    return matrix


def compare_model_matrices(
    empirical_P: np.ndarray,
    model_matrices: dict
) -> dict:
    """
    Compare model-derived transition matrices against the empirical ground truth.
    Returns MAE for each model — lower is better.
    """
    results = {}
    for model_name, matrix in model_matrices.items():
        mae = float(np.abs(matrix.values - empirical_P).mean())
        results[model_name] = {
            "mae":             round(mae, 4),
            "high_to_default": round(float(matrix.loc["High", "Default"]), 4),
            "medium_to_high":  round(float(matrix.loc["Medium", "High"]), 4),
            "default_persist": round(float(matrix.loc["Default", "Default"]), 4),
        }
    return results
