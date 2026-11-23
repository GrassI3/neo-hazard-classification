"""Dataset creation and loading: a realistic mock generator and a Kaggle CSV loader."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from src import RANDOM_STATE

FEATURE_COLUMNS: list[str] = [
    "estimated_diameter",
    "relative_velocity",
    "miss_distance",
    "absolute_magnitude",
]
TARGET_COLUMN = "is_potentially_hazardous_asteroid"

# Kaggle "NASA - Nearest Earth Objects" column -> our column name
KAGGLE_COLUMN_MAP: dict[str, str] = {
    "est_diameter_max": "estimated_diameter",
    "relative_velocity": "relative_velocity",
    "miss_distance": "miss_distance",
    "absolute_magnitude": "absolute_magnitude",
    "hazardous": TARGET_COLUMN,
}

# NASA defines a "large" asteroid as H <= 22 (roughly 140 m or bigger).
LARGE_OBJECT_MAGNITUDE = 22.0


def generate_mock_neo_data(
    n_rows: int = 5000,
    hazard_rate: float = 0.08,
    random_state: int = RANDOM_STATE,
    noise_rate: float = 0.02,
    verbose: bool = False,
) -> pd.DataFrame:
    """Generate a synthetic NEO dataset with physically correlated features.

    The label mostly follows NASA's rule (large object AND close approach). The
    distance cut-off is tuned so the hazardous share equals ``hazard_rate``, then
    ``noise_rate`` of the labels are flipped (equal numbers in each direction, so
    the class balance is preserved) so the task is learnable but not trivial.
    """
    if not 0 < hazard_rate < 0.5:
        raise ValueError("hazard_rate must be between 0 and 0.5")
    rng = np.random.default_rng(random_state)

    absolute_magnitude = np.clip(rng.normal(22.5, 2.8, n_rows), 14.0, 32.0)

    # D(km) = 1329 / sqrt(albedo) * 10^(-H/5): brighter (lower H) means bigger.
    albedo = np.clip(rng.normal(0.14, 0.03, n_rows), 0.05, 0.40)
    estimated_diameter = 1329.0 / np.sqrt(albedo) * 10 ** (-absolute_magnitude / 5)
    estimated_diameter *= rng.lognormal(0.0, 0.05, n_rows)  # measurement noise

    relative_velocity = np.clip(rng.lognormal(np.log(13.0), 0.45, n_rows), 1.0, 70.0)

    # Log-uniform between 0.1 million and 75 million km.
    miss_distance = 10 ** rng.uniform(np.log10(1e5), np.log10(7.5e7), n_rows)

    # NASA-style rule: large AND close. Choose the distance cut-off among large
    # objects so that exactly hazard_rate * n_rows objects are hazardous.
    n_hazardous = int(round(hazard_rate * n_rows))
    is_large = absolute_magnitude <= LARGE_OBJECT_MAGNITUDE
    large_idx = np.flatnonzero(is_large)
    if n_hazardous > len(large_idx):
        raise ValueError("hazard_rate too high for the number of large objects")
    closest_large = large_idx[np.argsort(miss_distance[large_idx])[:n_hazardous]]
    label = np.zeros(n_rows, dtype=bool)
    label[closest_large] = True

    # Symmetric label noise: flip k positives and k negatives.
    n_flips_each = min(int(round(noise_rate * n_rows / 2)), n_hazardous)
    pos_idx = np.flatnonzero(label)
    neg_idx = np.flatnonzero(~label)
    label[rng.choice(pos_idx, n_flips_each, replace=False)] = False
    label[rng.choice(neg_idx, n_flips_each, replace=False)] = True

    df = pd.DataFrame(
        {
            "estimated_diameter": estimated_diameter,
            "relative_velocity": relative_velocity,
            "miss_distance": miss_distance,
            "absolute_magnitude": absolute_magnitude,
            TARGET_COLUMN: label,
        }
    )
    if verbose:
        share = df[TARGET_COLUMN].mean()
        print(f"Generated {n_rows} rows, hazardous share = {share:.2%}")
    return df


def load_real_neo_data(path: str | Path | IO[Any]) -> pd.DataFrame:
    """Load the Kaggle NASA NEO CSV and map it onto the project's 5-column schema."""
    raw = pd.read_csv(path)
    missing = [c for c in KAGGLE_COLUMN_MAP if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    df = raw[list(KAGGLE_COLUMN_MAP)].rename(columns=KAGGLE_COLUMN_MAP).dropna()
    target = df[TARGET_COLUMN]
    if target.dtype == object:
        target = target.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    df[TARGET_COLUMN] = target.astype(bool)
    for col in FEATURE_COLUMNS:
        df[col] = df[col].astype(float)
    return df.reset_index(drop=True)[FEATURE_COLUMNS + [TARGET_COLUMN]]


# ---------------------------------------------------------------- real NASA data
# The three raw NASA files live in data/raw/:
#   near_earth_asteroids_2025.csv   one row per asteroid: H, diameter, and the real
#                                   NASA label `pha` (Potentially Hazardous Asteroid)
#   close_approaches_*.csv          one row per Earth fly-by: distance and speed
# They are joined on the asteroid designation to get our 5-column schema.
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
NASA_DATASET_PATH = PROJECT_DIR / "data" / "neo_nasa.csv"
ASTEROIDS_FILE = RAW_DIR / "near_earth_asteroids_2025.csv"
APPROACH_FILES = [
    RAW_DIR / "close_approaches_1960_2060.csv",
    RAW_DIR / "close_approaches_2015_2035.csv",
]
AU_KM = 149_597_870.7  # 1 astronomical unit in km


def _read_approaches(path: str | Path) -> pd.DataFrame:
    """Read one close-approach file (either of the two NASA formats) into a common shape."""
    raw = pd.read_csv(path)
    if "des" in raw.columns:  # NASA CNEOS format (1960-2060 file)
        designation, date = raw["des"], pd.to_datetime(raw["cd"], format="%Y-%b-%d %H:%M")
        dist_au, velocity = raw["dist"], raw["v_rel"]
    else:  # 2015-2035 file
        designation, date = raw["designation"], pd.to_datetime(raw["close_approach_date"])
        dist_au, velocity = raw["distance_au"], raw["velocity_km_s"]
    return pd.DataFrame(
        {
            "designation": designation.astype(str).str.strip(),
            "date": date,
            "miss_distance": dist_au.astype(float) * AU_KM,  # AU -> km
            "relative_velocity": velocity.astype(float),  # already km/s
        }
    )


def build_nasa_dataset(
    asteroids_path: str | Path = ASTEROIDS_FILE,
    approach_paths: list[str | Path] | None = None,
) -> pd.DataFrame:
    """Join the NASA asteroid catalogue with the close-approach records.

    Steps:
      1. Stack both close-approach files and drop exact duplicates (same object,
         same date) because the two files overlap for 2015-2035.
      2. Keep ONE row per asteroid: its closest recorded approach. An asteroid with
         many fly-bys would otherwise appear in both train and test (leakage).
      3. Attach H, diameter and the real `pha` label from the asteroid catalogue.
    """
    approach_paths = approach_paths or APPROACH_FILES
    approaches = pd.concat([_read_approaches(p) for p in approach_paths], ignore_index=True)
    approaches = approaches.drop_duplicates(["designation", "date"])
    closest = approaches.sort_values("miss_distance").drop_duplicates("designation")

    asteroids = pd.read_csv(asteroids_path, low_memory=False)
    asteroids = pd.DataFrame(
        {
            "designation": asteroids["pdes"].astype(str).str.strip(),
            "estimated_diameter": asteroids["diameter_km"],
            "absolute_magnitude": asteroids["H"],
            TARGET_COLUMN: asteroids["pha"],
        }
    )
    df = closest.merge(asteroids, on="designation", how="inner").dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    )
    target = df[TARGET_COLUMN]
    if target.dtype == object:
        target = target.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    df[TARGET_COLUMN] = target.astype(bool)
    df = df.sort_values("designation").reset_index(drop=True)
    return df[["designation", "date"] + FEATURE_COLUMNS + [TARGET_COLUMN]]


def load_nasa_dataset(path: str | Path = NASA_DATASET_PATH, rebuild: bool = False) -> pd.DataFrame:
    """Return the real NASA dataset (5 columns), building and caching it on first use."""
    path = Path(path)
    if rebuild or not path.exists():
        full = build_nasa_dataset()
        path.parent.mkdir(parents=True, exist_ok=True)
        full.to_csv(path, index=False)
    df = pd.read_csv(path)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(bool)
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def get_dataset(
    source: str = "nasa",
    path: str | Path | IO[Any] | None = None,
    **mock_kwargs: Any,
) -> pd.DataFrame:
    """Return the real NASA data (``"nasa"``), mock data (``"mock"``) or a Kaggle CSV (``"csv"``)."""
    if source == "nasa":
        return load_nasa_dataset()
    if source == "mock":
        return generate_mock_neo_data(**mock_kwargs)
    if source == "csv":
        if path is None:
            raise ValueError("path is required when source='csv'")
        return load_real_neo_data(path)
    raise ValueError(f"Unknown source: {source!r}")


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Quick EDA summary: shape, dtypes, missing values and class balance."""
    counts = df[TARGET_COLUMN].value_counts()
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "class_counts": {
            "Safe": int(counts.get(False, 0)),
            "Hazardous": int(counts.get(True, 0)),
        },
        "class_percentages": {
            "Safe": float(counts.get(False, 0) / len(df) * 100),
            "Hazardous": float(counts.get(True, 0) / len(df) * 100),
        },
    }
