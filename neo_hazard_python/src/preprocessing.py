"""Leakage-free preprocessing: stratified split, log + standard scaling, imbalance handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import RANDOM_STATE
from src.data import FEATURE_COLUMNS, TARGET_COLUMN

# Diameter and miss distance span several orders of magnitude (heavy right skew).
# log1p compresses them so a few huge values don't dominate the scaler's mean/std
# and so SMOTE interpolates in a sensible space.
LOG_FEATURES: list[str] = ["estimated_diameter", "miss_distance"]

IMBALANCE_METHODS = ("none", "class_weight", "smote")


@dataclass
class DataBundle:
    """Everything the models need after preprocessing."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    feature_names: list[str]
    X_train_raw: pd.DataFrame  # unscaled, used for leakage-free cross-validation
    X_test_raw: pd.DataFrame


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split; the target is cast to int 0/1."""
    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(int)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def log_transform(X: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to the skewed features (returns a copy)."""
    X = X.copy()
    for col in LOG_FEATURES:
        if col in X.columns:
            X[col] = np.log1p(X[col])
    return X


def transform_features(X_raw: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Log-transform then scale raw feature rows with an already fitted scaler."""
    X_log = log_transform(X_raw[FEATURE_COLUMNS])
    return pd.DataFrame(scaler.transform(X_log), columns=FEATURE_COLUMNS, index=X_raw.index)


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    scaler_path: str | Path | None = None,
) -> DataBundle:
    """Split, then fit the scaler on the training set only and transform both sets."""
    X_train_raw, X_test_raw, y_train, y_test = split_data(df, test_size, random_state)
    scaler = StandardScaler().fit(log_transform(X_train_raw))
    if scaler_path is not None:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
    return DataBundle(
        X_train=transform_features(X_train_raw, scaler),
        X_test=transform_features(X_test_raw, scaler),
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        feature_names=list(FEATURE_COLUMNS),
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
    )


def class_counts(y: pd.Series | np.ndarray) -> dict[str, int]:
    """Count Safe (0) and Hazardous (1) labels."""
    y = np.asarray(y)
    return {"Safe": int((y == 0).sum()), "Hazardous": int((y == 1).sum())}


def class_weight_params(y_train: pd.Series | np.ndarray) -> dict[str, object]:
    """Weights for the class-weighting strategy: XGBoost scale_pos_weight and RF class_weight."""
    counts = class_counts(y_train)
    return {
        "scale_pos_weight": counts["Safe"] / max(counts["Hazardous"], 1),
        "class_weight": "balanced",
    }


def handle_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "smote",
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Resample the TRAINING set only.

    ``"smote"`` oversamples the minority class; ``"class_weight"`` and ``"none"``
    return the data unchanged (class weighting happens inside the models).
    """
    if method not in IMBALANCE_METHODS:
        raise ValueError(f"method must be one of {IMBALANCE_METHODS}")
    before = class_counts(y_train)
    if method == "smote":
        X_res, y_res = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
    else:
        X_res, y_res = X_train, y_train
    if verbose:
        print(f"[{method}] before: {before} -> after: {class_counts(y_res)}")
    return X_res, y_res
