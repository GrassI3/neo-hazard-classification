"""Streamlit dashboard: Near-Earth Object hazard classification (UI only; logic lives in src/)."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.data import (
    FEATURE_COLUMNS,
    generate_mock_neo_data,
    load_nasa_dataset,
    load_real_neo_data,
    summarize_dataset,
)
from src.evaluation import LOWER_IS_BETTER, generate_conclusion, metrics_at_threshold
from src.pipeline import run_pipeline
from src.plots import (
    confusion_matrix_figure,
    feature_importance_figure,
    risk_matrix_figure,
    roc_curve_figure,
)
from src.preprocessing import transform_features

st.set_page_config(page_title="NEO Hazard Classification", page_icon="☄️", layout="wide")

# Dark text on the green highlight so it stays readable in both light and dark themes.
BEST_CELL_STYLE = "background-color: #b7e4c7; color: #0b3d20; font-weight: bold;"

IMBALANCE_OPTIONS = {"SMOTE": "smote", "Class weighting": "class_weight", "None": "none"}


@st.cache_data(show_spinner=False)
def load_nasa() -> pd.DataFrame:
    """Cached loading of the real NASA dataset (built from data/raw/ on first use)."""
    return load_nasa_dataset()


@st.cache_data(show_spinner=False)
def load_mock(n_rows: int, hazard_rate: float) -> pd.DataFrame:
    """Cached mock-data generation."""
    return generate_mock_neo_data(n_rows=n_rows, hazard_rate=hazard_rate)


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Cached parsing of an uploaded Kaggle CSV."""
    return load_real_neo_data(io.BytesIO(file_bytes))


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Data")
source = st.sidebar.radio("Data source", ["Real NASA data", "Mock", "Upload CSV"])
df: pd.DataFrame | None = None
if source == "Real NASA data":
    df = load_nasa()
    st.sidebar.caption(f"{len(df):,} real asteroids (closest approach each), NASA `pha` label.")
elif source == "Mock":
    n_rows = st.sidebar.slider("Number of rows", 1000, 20000, 5000, step=500)
    hazard_pct = st.sidebar.slider("Hazardous rate (%)", 3, 15, 8)
    df = load_mock(n_rows, hazard_pct / 100)
else:
    uploaded = st.sidebar.file_uploader("Kaggle NASA NEO CSV (neo.csv)", type="csv")
    if uploaded is not None:
        try:
            df = load_csv(uploaded.getvalue())
        except ValueError as exc:
            st.sidebar.error(str(exc))

st.sidebar.header("Preprocessing")
imbalance_label = st.sidebar.selectbox("Imbalance method", list(IMBALANCE_OPTIONS))
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, step=0.05)

st.sidebar.header("Random Forest (bagging)")
rf_n = st.sidebar.slider("RF n_estimators", 50, 500, 200, step=10)
rf_depth = st.sidebar.slider("RF max_depth", 3, 30, 12)

st.sidebar.header("XGBoost (boosting)")
xgb_n = st.sidebar.slider("XGB n_estimators", 50, 500, 200, step=10)
xgb_depth = st.sidebar.slider("XGB max_depth", 2, 12, 4)
xgb_lr = st.sidebar.slider("XGB learning_rate", 0.01, 0.5, 0.1, step=0.01)

train_clicked = st.sidebar.button("Train / Re-train Models", type="primary", disabled=df is None)

# ---------------------------------------------------------------- header
st.title("☄️ Near-Earth Object Hazard Classification")
st.caption(
    "Comparing **Random Forest** (bagging: many independent deep trees, votes averaged) with "
    "**XGBoost** (boosting: shallow trees built one after another, each fixing the previous "
    "trees' errors) on a heavily imbalanced dataset."
)

if train_clicked and df is not None:
    with st.spinner("Training Random Forest and XGBoost..."):
        st.session_state["output"] = run_pipeline(
            df,
            test_size=test_size,
            imbalance_method=IMBALANCE_OPTIONS[imbalance_label],
            rf_params={"n_estimators": rf_n, "max_depth": rf_depth},
            xgb_params={"n_estimators": xgb_n, "max_depth": xgb_depth, "learning_rate": xgb_lr},
        )
        st.session_state["df"] = df
        st.session_state["imbalance_label"] = imbalance_label

if "output" not in st.session_state:
    if df is None:
        st.info("Upload a CSV in the sidebar, or switch back to the NASA data.")
    else:
        summary = summarize_dataset(df)
        st.info(
            f"👋 Dataset ready: **{summary['shape'][0]:,} objects**, "
            f"**{summary['class_percentages']['Hazardous']:.1f}% hazardous**. "
            "Choose your settings in the sidebar and click **Train / Re-train Models** to start."
        )
        st.dataframe(df.head(10), use_container_width=True)
    st.stop()

output = st.session_state["output"]
trained_df: pd.DataFrame = st.session_state["df"]
results = output["results"]
rf_res, xgb_res = results["Random Forest"], results["XGBoost"]
summary = summarize_dataset(trained_df)

tabs = st.tabs(
    ["Overview", "Risk Matrix", "Confusion Matrices", "ROC Curves",
     "Feature Importance", "Threshold & Try It", "Conclusion"]
)

# ---------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(8)
    c[0].metric("Dataset size", f"{summary['shape'][0]:,}")
    c[1].metric("Hazardous", f"{summary['class_percentages']['Hazardous']:.1f}%")
    c[2].metric("RF Recall", f"{rf_res['recall']:.3f}")
    c[3].metric("RF F1", f"{rf_res['f1']:.3f}")
    c[4].metric("RF ROC-AUC", f"{rf_res['roc_auc']:.3f}")
    c[5].metric("XGB Recall", f"{xgb_res['recall']:.3f}")
    c[6].metric("XGB F1", f"{xgb_res['f1']:.3f}")
    c[7].metric("XGB ROC-AUC", f"{xgb_res['roc_auc']:.3f}")

    comparison = output["comparison"].set_index("model")
    higher = [col for col in comparison.columns if col not in LOWER_IS_BETTER]
    lower = [col for col in comparison.columns if col in LOWER_IS_BETTER]
    styled = (
        comparison.style.format("{:.3f}")
        .format("{:d}", subset=["false_negatives"])
        .highlight_max(subset=higher, props=BEST_CELL_STYLE)
        .highlight_min(subset=lower, props=BEST_CELL_STYLE)
    )
    st.subheader("Model comparison (test set)")
    st.dataframe(styled, use_container_width=True)
    st.caption("Green = better value for that metric.")
    st.warning(
        "**Why accuracy is not the headline metric:** only about "
        f"{summary['class_percentages']['Hazardous']:.0f}% of objects are hazardous, so a model "
        f"that always says *Safe* would already score ~{summary['class_percentages']['Safe']:.0f}% "
        "accuracy while missing every dangerous asteroid. Recall, F1, ROC-AUC and PR-AUC tell us "
        "far more."
    )
    before, after = output["class_counts_before"], output["class_counts_after"]
    st.caption(
        f"Imbalance method: **{st.session_state['imbalance_label']}**. Training class counts "
        f"before: {before} → after: {after}. The test set is never resampled."
    )

# ---------------------------------------------------------------- risk matrix
with tabs[1]:
    st.plotly_chart(risk_matrix_figure(trained_df), use_container_width=True)
    n_haz = int(trained_df["is_potentially_hazardous_asteroid"].sum())
    st.caption(
        f"Red = Hazardous, Blue = Safe. **All {len(trained_df):,} objects are plotted** "
        f"({len(trained_df) - n_haz:,} safe, {n_haz:,} hazardous). Marker size uses a log scale "
        "of diameter so small asteroids stay visible. Safe asteroids are mostly small (median "
        "~30 m) while hazardous ones are large (median ~260 m), which is why the big markers are red. "
        "Note that distance and speed barely separate the classes: size does."
    )

# ---------------------------------------------------------------- confusion matrices
with tabs[2]:
    zmax = max(rf_res["confusion_matrix"].max(), xgb_res["confusion_matrix"].max())
    cols = st.columns(2)
    for col, (name, res) in zip(cols, results.items()):
        with col:
            st.plotly_chart(
                confusion_matrix_figure(res["confusion_matrix"], name, zmax),
                use_container_width=True,
            )
            st.markdown(
                f"**False negatives (missed hazards): {res['false_negatives']} — "
                f"FN rate {res['false_negative_rate']:.1%}**"
            )

# ---------------------------------------------------------------- ROC
with tabs[3]:
    st.plotly_chart(roc_curve_figure(results), use_container_width=True)

# ---------------------------------------------------------------- feature importance
with tabs[4]:
    show_rf = st.toggle("Also show Random Forest importances", value=False)
    st.plotly_chart(feature_importance_figure(output["importances"], show_rf),
                    use_container_width=True)

# ---------------------------------------------------------------- threshold + try it
with tabs[5]:
    st.subheader("Decision threshold")
    threshold = st.slider("Hazard probability threshold", 0.05, 0.95, 0.5, step=0.05)
    st.caption("Lowering the threshold catches more true hazards (higher recall) at the cost of "
               "more false alarms. For planetary defence, a missed hazard is the costly error.")
    rows = []
    for name, res in results.items():
        m = metrics_at_threshold(res["y_true"], res["proba"], threshold)
        rows.append({"model": name, "recall": m["recall"], "precision": m["precision"],
                     "f1": m["f1"], "false_negatives": m["false_negatives"],
                     "false_positives": m["false_positives"]})
    st.dataframe(pd.DataFrame(rows).set_index("model").style.format(
        {"recall": "{:.3f}", "precision": "{:.3f}", "f1": "{:.3f}"}), use_container_width=True)

    st.subheader("Try your own asteroid")
    with st.form("asteroid"):
        fc = st.columns(4)
        diameter = fc[0].number_input("Estimated diameter (km)", 0.001, 50.0, 0.5, format="%.3f")
        velocity = fc[1].number_input("Relative velocity (km/s)", 0.5, 80.0, 15.0)
        miss = fc[2].number_input("Miss distance (km)", 1e4, 1e8, 2e6, step=1e5, format="%.0f")
        magnitude = fc[3].number_input("Absolute magnitude (H)", 10.0, 35.0, 20.0)
        submitted = st.form_submit_button("Predict")
    if submitted:
        X_new = pd.DataFrame([[diameter, velocity, miss, magnitude]], columns=FEATURE_COLUMNS)
        X_scaled = transform_features(X_new, output["bundle"].scaler)
        pc = st.columns(2)
        for col, (name, model) in zip(pc, output["trained"].models.items()):
            p = float(model.predict_proba(X_scaled)[0, 1])
            col.metric(f"{name} hazard probability", f"{p:.1%}",
                       "Hazardous" if p >= threshold else "Safe",
                       delta_color="inverse" if p >= threshold else "normal")

# ---------------------------------------------------------------- conclusion
with tabs[6]:
    st.success(generate_conclusion(rf_res, xgb_res))
