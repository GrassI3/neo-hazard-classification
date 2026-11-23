"""Imbalance-aware evaluation: metrics, curves, threshold analysis and comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

HEADLINE_METRICS = ["recall", "f1", "roc_auc", "pr_auc", "precision"]
# Metrics where a lower value is better (used when highlighting the winner).
LOWER_IS_BETTER = {"false_negative_rate", "false_positive_rate", "false_negatives", "train_time_s"}


def metrics_at_threshold(
    y_true: np.ndarray | pd.Series, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Threshold-dependent metrics for predicted hazard probabilities."""
    y_true = np.asarray(y_true)
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm,
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
) -> dict[str, Any]:
    """Evaluate a fitted classifier on the untouched test set.

    Accuracy is included for completeness but is a secondary metric: with ~8%
    hazardous objects, always predicting "Safe" already scores ~92%.
    """
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    results = metrics_at_threshold(y_test, proba, threshold)
    results.update(
        {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
            "fpr": fpr,
            "tpr": tpr,
            "proba": proba,
            "y_true": np.asarray(y_test),
            "accuracy_is_secondary": True,
        }
    )
    return results


def threshold_analysis(
    y_true: np.ndarray | pd.Series, proba: np.ndarray, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """Recall / precision / F1 for decision thresholds between 0.1 and 0.9."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 0.91, 0.05), 2)
    rows = []
    for t in thresholds:
        m = metrics_at_threshold(y_true, proba, float(t))
        rows.append({k: m[k] for k in ("threshold", "recall", "precision", "f1", "false_negatives")})
    return pd.DataFrame(rows)


def best_f1_threshold(analysis: pd.DataFrame) -> tuple[float, float]:
    """Return the (threshold, F1) pair with the highest F1 from a threshold analysis."""
    best = analysis.loc[analysis["f1"].idxmax()]
    return float(best["threshold"]), float(best["f1"])


def compare_models(
    results_rf: dict[str, Any],
    results_xgb: dict[str, Any],
    train_times: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Tidy comparison table, one row per model."""
    cols = [
        "recall", "precision", "f1", "roc_auc", "pr_auc", "accuracy",
        "false_negatives", "false_negative_rate", "false_positive_rate",
    ]
    rows = []
    for name, res in (("Random Forest", results_rf), ("XGBoost", results_xgb)):
        row = {"model": name, **{c: res[c] for c in cols}}
        if train_times is not None:
            row["train_time_s"] = train_times[name]
        rows.append(row)
    return pd.DataFrame(rows)


def generate_conclusion(results_rf: dict[str, Any], results_xgb: dict[str, Any]) -> str:
    """Plain-language summary of which model did better on the key metrics."""

    def winner(metric: str, lower_better: bool = False) -> str:
        a, b = results_rf[metric], results_xgb[metric]
        if np.isclose(a, b):
            return "Both models tied"
        rf_wins = a < b if lower_better else a > b
        return "Random Forest" if rf_wins else "XGBoost"

    lines = [
        f"**Recall:** {winner('recall')} "
        f"(RF {results_rf['recall']:.3f} vs XGB {results_xgb['recall']:.3f}).",
        f"**Lower false-negative rate:** {winner('false_negative_rate', lower_better=True)} "
        f"(RF {results_rf['false_negative_rate']:.3f} vs XGB {results_xgb['false_negative_rate']:.3f}; "
        f"missed hazards: RF {results_rf['false_negatives']}, XGB {results_xgb['false_negatives']}).",
        f"**F1:** {winner('f1')} (RF {results_rf['f1']:.3f} vs XGB {results_xgb['f1']:.3f}).",
        f"**ROC-AUC:** {winner('roc_auc')} "
        f"(RF {results_rf['roc_auc']:.3f} vs XGB {results_xgb['roc_auc']:.3f}).",
        f"**PR-AUC:** {winner('pr_auc')} "
        f"(RF {results_rf['pr_auc']:.3f} vs XGB {results_xgb['pr_auc']:.3f}).",
    ]
    return "\n\n".join(lines)
