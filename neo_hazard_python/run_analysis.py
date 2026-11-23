"""Command-line run of the full study: prints every table and exports results/ for the report.

Usage:  python run_analysis.py            (real NASA data, default)
        python run_analysis.py --mock     (synthetic data, 5,000 rows)
        python run_analysis.py data/neo.csv   (a Kaggle-format CSV)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import (
    NASA_DATASET_PATH,
    generate_mock_neo_data,
    load_nasa_dataset,
    load_real_neo_data,
    summarize_dataset,
)
from src.evaluation import best_f1_threshold, threshold_analysis
from src.models import DEFAULT_RF_PARAMS, DEFAULT_XGB_PARAMS, cross_validate_models
from src.pipeline import compare_imbalance_strategies, run_pipeline
from src.plots import save_static_figures
from src.preprocessing import class_counts

RESULTS_DIR = Path(__file__).parent / "results"
IMBALANCE_METHOD = "smote"

pd.set_option("display.width", 160)
pd.set_option("display.precision", 4)


def main() -> None:
    """Run the pipeline, print the results and save tables and figures."""
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "--mock":
        df = generate_mock_neo_data()
        source = "mock (n_rows=5000, hazard_rate=0.08, random_state=42)"
    elif arg:
        df = load_real_neo_data(arg)
        source = arg
    else:
        df = load_nasa_dataset(rebuild=True)
        source = f"real NASA data ({NASA_DATASET_PATH.name}, built from data/raw/)"
    RESULTS_DIR.mkdir(exist_ok=True)

    summary = summarize_dataset(df)
    print(f"=== Dataset: {source} ===")
    print(f"shape: {summary['shape']}")
    print(f"class counts: {summary['class_counts']}")
    print("class %: " + ", ".join(f"{k} {v:.2f}%" for k, v in summary["class_percentages"].items()))
    corr = df[["estimated_diameter", "absolute_magnitude"]].corr(method="spearman").iloc[0, 1]
    print(f"Spearman corr(diameter, absolute_magnitude) = {corr:.3f}")

    out = run_pipeline(df, imbalance_method=IMBALANCE_METHOD)
    # The scaler was fitted on the training split only; save it for reuse.
    joblib.dump(out["bundle"].scaler, RESULTS_DIR / "scaler.joblib")
    print(f"\n=== Imbalance handling ({IMBALANCE_METHOD}) ===")
    print(f"train before: {out['class_counts_before']} -> after: {out['class_counts_after']}")
    print(f"test set (untouched): {class_counts(out['bundle'].y_test)}")

    print("\n=== Test-set comparison ===")
    print(out["comparison"].to_string(index=False))
    out["comparison"].to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    print("\n=== 5-fold stratified CV (training data, SMOTE inside each fold) ===")
    cv = cross_validate_models(out["bundle"], imbalance_method=IMBALANCE_METHOD)
    print(cv.to_string(index=False))
    cv.to_csv(RESULTS_DIR / "cross_validation.csv", index=False)

    print("\n=== Imbalance strategy comparison (test set) ===")
    strategies = compare_imbalance_strategies(out["bundle"])
    print(strategies.to_string(index=False))
    strategies.to_csv(RESULTS_DIR / "imbalance_strategies.csv", index=False)

    print("\n=== Threshold analysis ===")
    best = {}
    frames = []
    for name, res in out["results"].items():
        ta = threshold_analysis(res["y_true"], res["proba"])
        frames.append(ta.assign(model=name))
        best[name] = best_f1_threshold(ta)
        print(f"{name}: best F1 = {best[name][1]:.4f} at threshold {best[name][0]:.2f}")
    thresholds = pd.concat(frames)
    print(thresholds.pivot(index="threshold", columns="model", values=["recall", "precision"]))
    thresholds.to_csv(RESULTS_DIR / "threshold_analysis.csv", index=False)

    print("\n=== Feature importance ===")
    for name, imp in out["importances"].items():
        print(f"{name}:\n{imp.to_string(index=False)}")
        imp.to_csv(RESULTS_DIR / f"feature_importance_{name.lower().replace(' ', '_')}.csv",
                   index=False)

    figures = save_static_figures(out, RESULTS_DIR)
    run_info = {
        "source": source,
        "imbalance_method": IMBALANCE_METHOD,
        "rf_params": DEFAULT_RF_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "best_f1_threshold": best,
        "train_times_s": out["trained"].train_times,
        "spearman_diameter_vs_magnitude": float(np.round(corr, 4)),
    }
    (RESULTS_DIR / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"\nSaved tables and figures to {RESULTS_DIR}: {[p.name for p in figures]}")


if __name__ == "__main__":
    main()
