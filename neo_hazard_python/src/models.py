"""Model builders, training, cross-validation and feature importance."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from src import RANDOM_STATE
from src.preprocessing import DataBundle, class_weight_params, handle_imbalance, log_transform

DEFAULT_RF_PARAMS: dict[str, Any] = {"n_estimators": 200, "max_depth": 12}
DEFAULT_XGB_PARAMS: dict[str, Any] = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1}

MODEL_NAMES = ("Random Forest", "XGBoost")


def build_random_forest(
    n_estimators: int = 200,
    max_depth: int | None = 12,
    class_weight: str | dict | None = None,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """Bagging ensemble: independent deep trees on bootstrap samples."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgboost(
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    scale_pos_weight: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> XGBClassifier:
    """Boosting ensemble: shallow trees added sequentially to fix earlier errors."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )


def build_models(
    y_train: pd.Series, rf_params: dict[str, Any], xgb_params: dict[str, Any], imbalance_method: str
) -> dict[str, Any]:
    """Build both (unfitted) models, adding class weights when that strategy is chosen."""
    rf_kwargs, xgb_kwargs = dict(rf_params), dict(xgb_params)
    if imbalance_method == "class_weight":
        weights = class_weight_params(y_train)
        rf_kwargs["class_weight"] = weights["class_weight"]
        xgb_kwargs["scale_pos_weight"] = weights["scale_pos_weight"]
    return {
        "Random Forest": build_random_forest(**rf_kwargs),
        "XGBoost": build_xgboost(**xgb_kwargs),
    }


@dataclass
class TrainedModels:
    """Fitted models plus how long each took to train."""

    models: dict[str, Any]
    train_times: dict[str, float]
    imbalance_method: str
    train_class_counts: dict[str, int] = field(default_factory=dict)


def train_models(
    data_bundle: DataBundle,
    rf_params: dict[str, Any] | None = None,
    xgb_params: dict[str, Any] | None = None,
    imbalance_method: str = "smote",
) -> TrainedModels:
    """Resample the training set (if requested) and fit both models on it."""
    rf_params = rf_params or DEFAULT_RF_PARAMS
    xgb_params = xgb_params or DEFAULT_XGB_PARAMS
    X_fit, y_fit = handle_imbalance(data_bundle.X_train, data_bundle.y_train, imbalance_method)
    models = build_models(data_bundle.y_train, rf_params, xgb_params, imbalance_method)
    times: dict[str, float] = {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_fit, y_fit)
        times[name] = time.perf_counter() - start
    counts = {"Safe": int((y_fit == 0).sum()), "Hazardous": int((y_fit == 1).sum())}
    return TrainedModels(models, times, imbalance_method, counts)


def cross_validate_models(
    data_bundle: DataBundle,
    rf_params: dict[str, Any] | None = None,
    xgb_params: dict[str, Any] | None = None,
    imbalance_method: str = "smote",
    n_splits: int = 5,
) -> pd.DataFrame:
    """Stratified k-fold CV on the training data only.

    Log transform, scaling and SMOTE all live inside an imblearn Pipeline, so they
    are re-fitted on each training fold and never see the validation fold.
    """
    rf_params = rf_params or DEFAULT_RF_PARAMS
    xgb_params = xgb_params or DEFAULT_XGB_PARAMS
    models = build_models(data_bundle.y_train, rf_params, xgb_params, imbalance_method)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        steps: list[tuple[str, Any]] = [
            ("log", FunctionTransformer(log_transform)),
            ("scale", StandardScaler()),
        ]
        if imbalance_method == "smote":
            steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        steps.append(("model", model))
        scores = cross_validate(
            Pipeline(steps),
            data_bundle.X_train_raw,
            data_bundle.y_train,
            cv=cv,
            scoring=["recall", "f1", "roc_auc"],
        )
        row: dict[str, Any] = {"model": name}
        for metric in ("recall", "f1", "roc_auc"):
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)
    return pd.DataFrame(rows)


def feature_importances(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Return the model's feature importances sorted from most to least important."""
    return (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
