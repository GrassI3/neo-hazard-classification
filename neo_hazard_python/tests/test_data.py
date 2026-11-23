"""Tests for the mock generator, the CSV loader and the EDA summary."""

import pandas as pd
import pytest

from src.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    generate_mock_neo_data,
    get_dataset,
    load_real_neo_data,
    summarize_dataset,
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return generate_mock_neo_data()


def test_shape_and_columns(df: pd.DataFrame) -> None:
    assert df.shape == (5000, 5)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]


def test_target_dtype_is_bool(df: pd.DataFrame) -> None:
    assert df[TARGET_COLUMN].dtype == bool


def test_imbalance_in_range(df: pd.DataFrame) -> None:
    assert 0.05 <= df[TARGET_COLUMN].mean() <= 0.10


@pytest.mark.parametrize("rate", [0.03, 0.08, 0.15])
def test_hazard_rate_parameter(rate: float) -> None:
    data = generate_mock_neo_data(n_rows=4000, hazard_rate=rate)
    assert data[TARGET_COLUMN].mean() == pytest.approx(rate, abs=0.005)


def test_same_seed_is_reproducible() -> None:
    pd.testing.assert_frame_equal(generate_mock_neo_data(), generate_mock_neo_data())
    assert not generate_mock_neo_data(random_state=1).equals(generate_mock_neo_data())


def test_physical_correlations(df: pd.DataFrame) -> None:
    corr = df[["estimated_diameter", "absolute_magnitude"]].corr(method="spearman").iloc[0, 1]
    assert corr < -0.9
    assert df["relative_velocity"].between(1, 70).all()
    assert df["miss_distance"].between(1e5, 7.5e7).all()
    assert df["absolute_magnitude"].between(14, 32).all()


def test_no_missing_values(df: pd.DataFrame) -> None:
    assert df.isna().sum().sum() == 0


def test_load_real_csv(tmp_path) -> None:
    kaggle = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "est_diameter_min": [0.1, 0.2, 0.3],
            "est_diameter_max": [0.2, 0.4, None],
            "relative_velocity": [10000.0, 20000.0, 30000.0],
            "miss_distance": [1e6, 2e6, 3e6],
            "orbiting_body": ["Earth"] * 3,
            "sentry_object": [False] * 3,
            "absolute_magnitude": [20.0, 21.0, 22.0],
            "hazardous": [True, False, True],
        }
    )
    path = tmp_path / "neo.csv"
    kaggle.to_csv(path, index=False)
    df = load_real_neo_data(path)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert len(df) == 2  # row with a missing value dropped
    assert df[TARGET_COLUMN].dtype == bool
    assert get_dataset("csv", path).equals(df)


def test_summary(df: pd.DataFrame) -> None:
    s = summarize_dataset(df)
    assert s["shape"] == (5000, 5)
    assert sum(s["class_counts"].values()) == 5000
    assert sum(s["class_percentages"].values()) == pytest.approx(100)


def test_build_nasa_dataset_joins_and_dedupes(tmp_path) -> None:
    from src.data import build_nasa_dataset

    asteroids = pd.DataFrame(
        {"pdes": ["A1", "B2", "C3"], "H": [20.0, 25.0, 21.0],
         "diameter_km": [0.3, 0.03, None], "pha": [True, False, True]}
    )
    cneos = pd.DataFrame(  # 1960-2060 format, distance in AU
        {"des": ["A1", "A1", "B2", "C3"],
         "cd": ["2020-Jan-01 00:00", "2030-Jan-01 00:00", "2021-Jan-01 00:00", "2021-Feb-01 00:00"],
         "dist": [0.04, 0.01, 0.02, 0.03], "v_rel": [10.0, 12.0, 8.0, 9.0]}
    )
    other = pd.DataFrame(  # 2015-2035 format, duplicate of A1's 2030 approach
        {"designation": ["A1"], "close_approach_date": ["2030-01-01 00:00"],
         "distance_au": [0.01], "velocity_km_s": [12.0]}
    )
    paths = [tmp_path / "a.csv", tmp_path / "c1.csv", tmp_path / "c2.csv"]
    for frame, path in zip([asteroids, cneos, other], paths):
        frame.to_csv(path, index=False)
    df = build_nasa_dataset(paths[0], paths[1:])
    assert list(df["designation"]) == ["A1", "B2"]  # C3 dropped (missing diameter)
    a1 = df.iloc[0]
    assert a1["relative_velocity"] == 12.0  # closest approach kept
    assert a1["miss_distance"] == pytest.approx(0.01 * 149_597_870.7)
    assert df[TARGET_COLUMN].dtype == bool


def test_real_nasa_dataset_if_present() -> None:
    from src.data import ASTEROIDS_FILE, load_nasa_dataset

    if not ASTEROIDS_FILE.exists():
        pytest.skip("raw NASA files not present")
    df = load_nasa_dataset()
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert len(df) > 20000
    assert 0.03 <= df[TARGET_COLUMN].mean() <= 0.10
    assert df.isna().sum().sum() == 0
