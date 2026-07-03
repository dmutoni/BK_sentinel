"""
BK Sentinel — Explanation Service
Generates per-account and global feature-contribution explanations for the
trained XGBoost model.

We compute these using XGBoost's own native Tree SHAP implementation
(Booster.predict(..., pred_contribs=True)) rather than the third-party
`shap` package. `shap`'s TreeExplainer reliably segfaults against this
model on some machines — a hard crash in its numba/llvmlite dependency
that no Python try/except can catch. XGBoost implements the exact same
Tree SHAP algorithm natively in its own C++ core, so the numbers are
identical — we just get them without the fragile dependency.
"""

import numpy as np
import xgboost as xgb
from database.loader import get_model
from config import FEATURE_LABELS, SHAP_TOP_N_FEATURES, STATES

N_CLASSES = len(STATES)


def _tree_shap_contribs(X: np.ndarray) -> np.ndarray:
    """
    Compute Tree SHAP contributions natively via XGBoost.

    XGBoost's pred_contribs output shape for multiclass models isn't
    consistent across versions/call paths — sometimes a proper 3D array,
    sometimes flattened 2D. We handle both explicitly rather than guessing.

    Returns an array of shape (n_samples, n_classes, n_features) — the
    trailing bias/expected-value column XGBoost includes is dropped.
    """
    model, _, features = get_model()
    booster = model.get_booster()
    dmat = xgb.DMatrix(X, feature_names=features)
    raw = np.array(booster.predict(dmat, pred_contribs=True))

    n_samples  = X.shape[0]
    n_features = len(features)

    if raw.ndim == 3:
        # (n_samples, n_classes, n_features + 1)
        return raw[:, :, :-1]

    if raw.ndim == 2 and raw.shape[1] == N_CLASSES * (n_features + 1):
        # Flattened multiclass output — reshape then drop the bias column
        return raw.reshape(n_samples, N_CLASSES, n_features + 1)[:, :, :-1]

    if raw.ndim == 2 and raw.shape[1] == n_features + 1:
        # Single-output model — same contributions apply to every class
        single = raw[:, :-1]
        return np.repeat(single[:, np.newaxis, :], N_CLASSES, axis=1)

    raise ValueError(
        f"Unexpected pred_contribs shape {raw.shape} for "
        f"{n_features} features / {N_CLASSES} classes"
    )


def explain_prediction(X_row: np.ndarray, pred_class_idx: int) -> list:
    """
    Generate a feature-contribution explanation for a single account.

    Args:
        X_row:          feature array for the account (shape: [n_features])
        pred_class_idx: the predicted class index (0=Low, 1=Med, 2=High, 3=Default)

    Returns:
        List of dicts with feature name, actual value, and contribution.
        Positive value = pushes toward pred_class. Negative = pushes away.
    """
    try:
        _, _, features = get_model()
        # X_row arrives as shape (1, n_features) from encode_features() —
        # flatten it so X_flat[i] gives a single value, not the whole row.
        X_flat = np.asarray(X_row).reshape(-1)

        contribs = _tree_shap_contribs(X_flat.reshape(1, -1))
        vals = contribs[0, pred_class_idx]

        # Get top N features by absolute contribution
        top_idx = np.argsort(np.abs(vals))[::-1][:SHAP_TOP_N_FEATURES]

        return [
            {
                "feature": FEATURE_LABELS.get(features[i], features[i]),
                "value":   round(float(X_flat[i]), 2),
                "shap":    round(float(vals[i]), 4),
            }
            for i in top_idx
        ]

    except Exception as e:
        print(f"[Explain] Error computing explanation: {e}")
        return []


def explain_batch_default(X_batch: np.ndarray, n_samples: int = 100) -> list:
    """
    Compute global feature importance for the Default class
    across a batch of accounts.

    Returns list of {feature, importance} sorted by importance descending.
    """
    try:
        _, _, features = get_model()

        # Sample for performance
        if len(X_batch) > n_samples:
            idx = np.random.choice(len(X_batch), n_samples, replace=False)
            X_batch = X_batch[idx]

        contribs = _tree_shap_contribs(X_batch)
        sv = contribs[:, 3, :]  # Default class = index 3

        mean_abs = np.abs(sv).mean(axis=0)
        importance = sorted(
            [{"feature": FEATURE_LABELS.get(features[i], features[i]),
              "importance": round(float(mean_abs[i]), 4)}
             for i in range(len(features))],
            key=lambda x: x["importance"],
            reverse=True
        )
        return importance

    except Exception as e:
        print(f"[Explain] Batch error: {e}")
        return []
