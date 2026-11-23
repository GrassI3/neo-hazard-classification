"""End-to-end pipeline used by the dashboard, the analysis script and the smoke test."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation import compare_models, evaluate_model
from src.models import (
    DEFAULT_RF_PARAMS,
    DEFAULT_XGB_PARAMS,
    MODEL_NAMES,
    feature_importances,
    train_models,
)
from src.preprocessing import IMBALANCE_METHODS, DataBundle, class_counts, prepare_data


def run_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    imbalance_method: str = "smote",
    rf_params: dict[str, Any] | None = None,
    xgb_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preprocess, train both models and evaluate them on the untouched test set."""
    bundle = prepare_data(df, test_size=test_size)
    trained = train_models(bundle, rf_params, xgb_params, imbalance_method)
    results = {
        name: evaluate_model(model, bundle.X_test, bundle.y_test)
        for name, model in trained.models.items()
    }
    return {
        "bundle": bundle,
        "trained": trained,
        "results": results,
        "comparison": compare_models(
            results["Random Forest"], results["XGBoost"], trained.train_times
        ),
        "importances": {
            name: feature_importances(model, bundle.feature_names)
            for name, model in trained.models.items()
        },
        "class_counts_before": class_counts(bundle.y_train),
        "class_counts_after": trained.train_class_counts,
    }


def compare_imbalance_strategies(
    bundle: DataBundle,
    rf_params: dict[str, Any] | None = None,
    xgb_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Train and test both models under each imbalance strategy (none / class_weight / smote)."""
    rf_params = rf_params or DEFAULT_RF_PARAMS
    xgb_params = xgb_params or DEFAULT_XGB_PARAMS
    rows = []
    for method in IMBALANCE_METHODS:
        trained = train_models(bundle, rf_params, xgb_params, method)
        for name in MODEL_NAMES:
            res = evaluate_model(trained.models[name], bundle.X_test, bundle.y_test)
            rows.append(
                {
                    "imbalance_method": method,
                    "model": name,
                    "recall": res["recall"],
                    "precision": res["precision"],
                    "f1": res["f1"],
                    "roc_auc": res["roc_auc"],
                    "pr_auc": res["pr_auc"],
                    "false_negatives": res["false_negatives"],
                }
            )
    return pd.DataFrame(rows)
