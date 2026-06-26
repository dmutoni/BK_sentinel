"""
BK Sentinel — SHAP Explanation Service
Generates local and global SHAP explanations for the Random Forest model.
"""

import numpy as np
from database.loader import get_model
from config import FEATURE_LABELS, SHAP_TOP_N_FEATURES


def explain_prediction(X_row: np.ndarray, pred_class_idx: int) -> list:
    """
    Generate SHAP explanation for a single account prediction.

    Args:
        X_row:          feature array for the account (shape: [n_features])
        pred_class_idx: the predicted class index (0=Low, 1=Med, 2=High, 3=Default)

    Returns:
        List of dicts with feature name, actual value, and SHAP value.
        Positive SHAP = pushes toward pred_class. Negative = pushes away.
    """
    try:
        import shap
        model, encoders, features = get_model()

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row.reshape(1, -1))

        # Handle both old (list) and new (3D array) SHAP output formats
        if isinstance(shap_values, list):
            vals = shap_values[pred_class_idx][0]
        else:
            arr = np.array(shap_values)
            vals = arr[0, :, pred_class_idx] if arr.ndim == 3 else arr[0]

        # Get top N features by absolute SHAP value
        top_idx = np.argsort(np.abs(vals))[::-1][:SHAP_TOP_N_FEATURES]

        return [
            {
                "feature": FEATURE_LABELS.get(features[i], features[i]),
                "value":   round(float(X_row[i]), 2),
                "shap":    round(float(vals[i]), 4),
            }
            for i in top_idx
        ]

    except ImportError:
        return []
    except Exception as e:
        print(f"[SHAP] Error computing explanation: {e}")
        return []


def explain_batch_default(X_batch: np.ndarray, n_samples: int = 100) -> list:
    """
    Compute global SHAP feature importance for the Default class
    across a batch of accounts.

    Returns list of {feature, importance} sorted by importance descending.
    """
    try:
        import shap
        model, encoders, features = get_model()

        # Sample for performance
        if len(X_batch) > n_samples:
            idx = np.random.choice(len(X_batch), n_samples, replace=False)
            X_batch = X_batch[idx]

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_batch)

        # Get Default class SHAP values (index 3)
        if isinstance(shap_values, list):
            sv = shap_values[3]
        else:
            arr = np.array(shap_values)
            sv  = arr[:, :, 3] if arr.ndim == 3 else arr

        mean_abs = np.abs(sv).mean(axis=0)
        importance = sorted(
            [{"feature": FEATURE_LABELS.get(features[i], features[i]),
              "importance": round(float(mean_abs[i]), 4)}
             for i in range(len(features))],
            key=lambda x: x["importance"],
            reverse=True
        )
        return importance

    except ImportError:
        return []
    except Exception as e:
        print(f"[SHAP] Batch error: {e}")
        return []
