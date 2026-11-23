"""Tests for training, evaluation, CV and the end-to-end smoke run."""

import numpy as np
import pytest

from src.data import generate_mock_neo_data
from src.evaluation import best_f1_threshold, compare_models, evaluate_model, threshold_analysis
from src.models import cross_validate_models, feature_importances, train_models
from src.pipeline import compare_imbalance_strategies, run_pipeline
from src.preprocessing import prepare_data

FAST_RF = {"n_estimators": 50, "max_depth": 8}
FAST_XGB = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.2}


@pytest.fixture(scope="module")
def bundle():
    return prepare_data(generate_mock_neo_data())


@pytest.fixture(scope="module")
def trained(bundle):
    return train_models(bundle, FAST_RF, FAST_XGB, "smote")


def test_training_is_fast(trained) -> None:
    assert all(t < 30 for t in trained.train_times.values())


def test_metric_keys_and_confusion_matrix(bundle, trained) -> None:
    for model in trained.models.values():
        res = evaluate_model(model, bundle.X_test, bundle.y_test)
        for key in ["recall", "precision", "f1", "roc_auc", "pr_auc", "accuracy",
                    "confusion_matrix", "false_negative_rate", "false_positive_rate",
                    "fpr", "tpr", "proba"]:
            assert key in res
        assert res["confusion_matrix"].shape == (2, 2)
        assert res["confusion_matrix"].sum() == len(bundle.y_test)
        assert 0.6 < res["roc_auc"] < 0.995  # learnable but not suspiciously perfect


def test_compare_models_table(bundle, trained) -> None:
    rf = evaluate_model(trained.models["Random Forest"], bundle.X_test, bundle.y_test)
    xgb = evaluate_model(trained.models["XGBoost"], bundle.X_test, bundle.y_test)
    table = compare_models(rf, xgb, trained.train_times)
    assert list(table["model"]) == ["Random Forest", "XGBoost"]
    assert "train_time_s" in table.columns


def test_threshold_analysis(bundle, trained) -> None:
    res = evaluate_model(trained.models["XGBoost"], bundle.X_test, bundle.y_test)
    ta = threshold_analysis(res["y_true"], res["proba"])
    assert ta["threshold"].min() == pytest.approx(0.1)
    assert ta["threshold"].max() == pytest.approx(0.9)
    assert ta["recall"].is_monotonic_decreasing  # higher threshold, fewer positives
    t, f1 = best_f1_threshold(ta)
    assert f1 == ta["f1"].max() and 0.1 <= t <= 0.9


def test_class_weight_applied(bundle) -> None:
    trained = train_models(bundle, FAST_RF, FAST_XGB, "class_weight")
    assert trained.models["Random Forest"].class_weight == "balanced"
    assert trained.models["XGBoost"].get_params()["scale_pos_weight"] > 5


def test_cross_validation(bundle) -> None:
    cv = cross_validate_models(bundle, FAST_RF, FAST_XGB, "smote", n_splits=3)
    assert len(cv) == 2
    assert (cv["roc_auc_mean"] > 0.6).all()


def test_feature_importances(bundle, trained) -> None:
    imp = feature_importances(trained.models["XGBoost"], bundle.feature_names)
    assert imp["importance"].is_monotonic_decreasing
    assert np.isclose(imp["importance"].sum(), 1.0, atol=1e-5)


def test_smoke_end_to_end() -> None:
    df = generate_mock_neo_data(n_rows=800)
    out = run_pipeline(df, imbalance_method="smote", rf_params=FAST_RF, xgb_params=FAST_XGB)
    assert len(out["comparison"]) == 2
    assert out["class_counts_after"]["Safe"] == out["class_counts_after"]["Hazardous"]
    strategies = compare_imbalance_strategies(out["bundle"], FAST_RF, FAST_XGB)
    assert len(strategies) == 6
