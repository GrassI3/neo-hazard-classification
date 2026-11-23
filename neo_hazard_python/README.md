# Near-Earth Object Hazard Classification (Python)

Predicts whether a Near-Earth Object (asteroid) is **potentially hazardous** and compares two
ensemble models on a heavily imbalanced dataset:

| Model | Ensemble family | Core idea |
|---|---|---|
| Random Forest | Bagging | Many deep trees trained independently on bootstrap samples, votes averaged |
| XGBoost | Boosting | Trees trained sequentially, each correcting the previous trees' errors |

It runs on **real NASA data**: 23,964 asteroids (each at its closest recorded Earth approach)
labelled with NASA's own `pha` (Potentially Hazardous Asteroid) flag. Only 6.6% are hazardous, so
the project focuses on **Recall, F1, ROC-AUC and PR-AUC** and on **false negatives** (missed
hazards) rather than accuracy. A synthetic (mock) generator is kept for tests and experiments.

## Setup

```powershell
cd neo_hazard_python
python -m venv .venv
.\.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py              # interactive dashboard (http://localhost:8501)
python run_analysis.py            # full study on the real NASA data + exports to results/
python run_analysis.py --mock     # same, on 5,000 synthetic rows
pytest                            # 32 tests (data, preprocessing, models, smoke, dashboard)
```

## Data

`data/raw/` holds the three NASA files. `src/data.py → build_nasa_dataset()` joins them into
`data/neo_nasa.csv` (built automatically on first use):

| Raw file | Gives |
|---|---|
| `near_earth_asteroids_2025.csv` | `H`, `diameter_km`, label `pha` |
| `close_approaches_1960_2060.csv` | miss distance (AU), relative velocity (km/s) |
| `close_approaches_2015_2035.csv` | same, other format (overlapping years are de-duplicated) |

One row per asteroid (its closest approach) is kept so the same object cannot appear in both
train and test. A Kaggle-format CSV can still be used via *Upload CSV* in the dashboard.

## Project structure

```
neo_hazard_python/
├── app.py                 # Streamlit dashboard (UI only)
├── run_analysis.py        # CLI: prints every table, writes results/
├── requirements.txt
├── README.md
├── REPORT.md              # written analysis with the real numbers
├── data/
│   ├── raw/               # the 3 original NASA CSVs
│   └── neo_nasa.csv       # joined modelling table (generated)
├── results/               # metrics_summary.csv, CV, strategies, thresholds, PNG figures
├── src/
│   ├── data.py            # NASA dataset builder, mock generator, Kaggle loader, EDA
│   ├── preprocessing.py   # stratified split, log + standard scaling, SMOTE / class weights
│   ├── models.py          # model builders, training, cross-validation, importances
│   ├── evaluation.py      # metrics, ROC/PR, threshold analysis, comparison, conclusion
│   ├── pipeline.py        # end-to-end run + imbalance-strategy comparison
│   └── plots.py           # Plotly figures (app) and matplotlib/seaborn PNGs (report)
└── tests/
    ├── test_data.py
    ├── test_preprocessing.py
    ├── test_models.py     # includes the 800-row end-to-end smoke test
    └── test_app.py        # Streamlit AppTest: loads the app and clicks Train
```

## Dashboard

* **Sidebar:** data source (real NASA data / mock / upload CSV), mock rows and hazardous rate, imbalance method,
  test size, Random Forest and XGBoost hyperparameters, **Train / Re-train Models** button.
* **Tabs:** Overview (metric cards and comparison table with the best values highlighted),
  Risk Matrix (Red = Hazardous, Blue = Safe, size = diameter), side-by-side Confusion Matrices
  on a shared colour scale with false-negative counts, overlapping ROC curves, Feature Importance,
  Threshold slider and a *Try your own asteroid* form, plus an auto-generated Conclusion.

## Screenshots

_Add screenshots of the dashboard here._

| Overview | Risk Matrix | Confusion Matrices |
|---|---|---|
| ![](docs/overview.png) | ![](docs/risk_matrix.png) | ![](docs/confusion.png) |

Static figures from the latest analysis run are in `results/`:
`confusion_matrices.png`, `roc_curves.png`, `feature_importance.png`.

## Reproducibility

`RANDOM_STATE = 42` is used everywhere (data generation, split, SMOTE, models, CV), so every
run of `run_analysis.py` produces the same numbers.
