"""Tests for stratification, leakage-free scaling and SMOTE."""

import numpy as np
import pytest

from src.data import generate_mock_neo_data
from src.preprocessing import class_counts, class_weight_params, handle_imbalance, prepare_data


@pytest.fixture(scope="module")
def bundle():
    return prepare_data(generate_mock_neo_data(), test_size=0.2)


def test_stratification(bundle) -> None:
    assert abs(bundle.y_train.mean() - bundle.y_test.mean()) < 0.01
    assert len(bundle.X_test) == 1000


def test_target_is_int(bundle) -> None:
    assert set(np.unique(bundle.y_train)) == {0, 1}
    assert bundle.y_train.dtype.kind == "i"


def test_scaler_fit_on_train_only(bundle) -> None:
    assert np.allclose(bundle.X_train.mean(), 0, atol=1e-9)
    assert np.allclose(bundle.X_train.std(ddof=0), 1, atol=1e-9)
    # The test set was transformed with the train statistics, so it is not exactly 0/1.
    assert not np.allclose(bundle.X_test.mean(), 0, atol=1e-9)
    assert not np.allclose(bundle.X_test.std(ddof=0), 1, atol=1e-9)
    assert bundle.scaler.n_samples_seen_ == len(bundle.X_train)


def test_smote_balances_train_only(bundle) -> None:
    test_before = class_counts(bundle.y_test)
    X_res, y_res = handle_imbalance(bundle.X_train, bundle.y_train, "smote")
    counts = class_counts(y_res)
    assert counts["Safe"] == counts["Hazardous"]
    assert class_counts(bundle.y_test) == test_before  # test set untouched
    assert bundle.y_test.mean() < 0.15


def test_no_test_rows_in_resampled_train(bundle) -> None:
    X_res, _ = handle_imbalance(bundle.X_train, bundle.y_train, "smote")
    train_rows = {tuple(r) for r in np.round(X_res.to_numpy(), 10)}
    test_rows = {tuple(r) for r in np.round(bundle.X_test.to_numpy(), 10)}
    assert train_rows.isdisjoint(test_rows)


@pytest.mark.parametrize("method", ["none", "class_weight"])
def test_unchanged_methods(bundle, method: str) -> None:
    X_res, y_res = handle_imbalance(bundle.X_train, bundle.y_train, method)
    assert X_res is bundle.X_train and y_res is bundle.y_train


def test_class_weight_params(bundle) -> None:
    params = class_weight_params(bundle.y_train)
    counts = class_counts(bundle.y_train)
    assert params["scale_pos_weight"] == pytest.approx(counts["Safe"] / counts["Hazardous"])
    assert params["class_weight"] == "balanced"


def test_invalid_method(bundle) -> None:
    with pytest.raises(ValueError):
        handle_imbalance(bundle.X_train, bundle.y_train, "magic")
