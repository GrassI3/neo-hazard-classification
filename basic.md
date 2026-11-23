# Near-Earth Object Hazard Classification
## A Comparative Study of Random Forest (Bagging) vs XGBoost (Boosting)

> **How to use this file:** Give this whole document to Claude Code and tell it:
> *"Read this brief and complete it phase by phase. Finish and verify each phase before starting the next. After each phase, summarize what you built and how you tested it."*

---

## 0. Project Context

**Goal:** Build an end-to-end data analytics / machine learning project that predicts whether a Near-Earth Object (asteroid) is *potentially hazardous*, and **compares two ensemble models**:

| Model | Ensemble family | Core idea |
|---|---|---|
| Random Forest | Bagging | Many deep trees trained independently on bootstrap samples, votes averaged |
| XGBoost | Boosting | Trees trained sequentially, each correcting the previous trees' errors |

**Why this is an interesting comparison:** The data is heavily imbalanced (only ~5-10% of objects are hazardous), so plain accuracy is misleading. A model that predicts "Safe" every time would score ~92% accuracy and be useless. The project therefore focuses on **Recall, F1-Score, ROC-AUC (and PR-AUC)**, and especially on **false negatives** (missing a truly hazardous asteroid is the costly error).

**Deliverable:** A runnable Python project with a Streamlit dashboard, plus a short written report suitable for submission to a teacher.

**Tech stack:** Python 3.10+, pandas, numpy, scikit-learn, imbalanced-learn, xgboost, plotly, seaborn, matplotlib, streamlit, pytest, joblib.

**Recommended project structure:**

```
neo-hazard-classification/
├── app.py                      # Streamlit entry point
├── requirements.txt
├── README.md
├── REPORT.md                   # Written findings (Phase 5)
├── data/
│   └── (optional) neo.csv      # Real Kaggle data if the user adds it
├── src/
│   ├── __init__.py
│   ├── data.py                 # Mock generator + real-data loader
│   ├── preprocessing.py        # Split, scaling, imbalance handling
│   ├── models.py               # Model builders and training
│   ├── evaluation.py           # Metrics, curves, confusion matrices
│   └── plots.py                # Plotly/Seaborn figure builders
└── tests/
    ├── test_data.py
    ├── test_preprocessing.py
    └── test_models.py
```

**Global rules for Claude Code (apply to every phase):**

1. Use a fixed `RANDOM_STATE = 42` everywhere so results are reproducible.
2. Keep logic in `src/` modules; `app.py` should only handle UI and call functions.
3. Add type hints and short docstrings to every function.
4. Never fit scalers, SMOTE, or any transformer on test data (**no data leakage**).
5. Run the code after writing it. Do not claim something works without executing it.

---

## PHASE 1 - Project Setup & Data Ingestion

**Objective:** A working environment and a dataset the whole app can run on immediately.

### Tasks

1. Create the folder structure above and a `requirements.txt` with pinned-minimum versions:
   `pandas, numpy, scikit-learn, imbalanced-learn, xgboost, plotly, seaborn, matplotlib, streamlit, pytest, joblib`.
2. In `src/data.py`, write `generate_mock_neo_data(n_rows=5000, hazard_rate=0.08, random_state=42) -> pd.DataFrame`.
   - Columns: `estimated_diameter` (km), `relative_velocity` (km/s), `miss_distance` (km), `absolute_magnitude` (H), `is_potentially_hazardous_asteroid` (bool).
   - Use realistic distributions:
     - `absolute_magnitude`: roughly normal around 22-23, clipped to ~[14, 32].
     - `estimated_diameter`: **derive it from absolute magnitude** using the standard formula `D(km) = 1329 / sqrt(albedo) * 10^(-H/5)` with albedo ~ 0.14 plus small noise (bigger = lower H). This makes the features physically correlated, as in the real dataset.
     - `relative_velocity`: log-normal, roughly 5-40 km/s.
     - `miss_distance`: log-uniform or log-normal, roughly 0.1M to 75M km.
   - **Label logic (important):** Make hazard status *mostly* follow NASA's real rule (large object AND close approach, i.e. low `absolute_magnitude` and small `miss_distance`), then flip a small percentage (~2-3%) of labels randomly as noise. This gives the models a real signal to learn **without** being trivially perfect. Tune thresholds so the final hazardous rate lands in the 5-10% range.
   - Expose the target rate as a parameter and assert/print the resulting class balance.
3. In `src/data.py`, also write `load_real_neo_data(path) -> pd.DataFrame` that reads the Kaggle NASA NEO CSV, renames/selects the columns to the same five names (map from Kaggle's `est_diameter_max`, `relative_velocity`, `miss_distance`, `absolute_magnitude`, `hazardous`), drops rows with missing values, and returns the same schema. Add a `get_dataset(source="mock" | "csv", path=None)` wrapper so the app can switch between them.
4. Create a quick EDA helper `summarize_dataset(df)` returning shape, dtypes, missing values, class counts, and class percentages.

### Deliverables
- `requirements.txt`, project skeleton, `src/data.py`
- A short script or notebook cell proving the mock data loads and showing the class imbalance

### Acceptance criteria
- `generate_mock_neo_data()` returns 5,000 rows, 5 columns, hazardous rate between 5% and 10%
- Same seed gives identical data on repeated calls
- Feature columns are correlated sensibly (e.g. diameter vs absolute magnitude strongly negative correlation)
- `tests/test_data.py` checks shape, columns, dtype of target, and imbalance range

---

## PHASE 2 - Preprocessing & Imbalance Handling

**Objective:** A leakage-free preprocessing pipeline that handles heavy class imbalance.

### Tasks

1. In `src/preprocessing.py`, write `split_data(df, test_size=0.2, random_state=42)`:
   - Separate features `X` and target `y` (cast target to int 0/1).
   - Use **stratified** `train_test_split` so both splits keep the same hazardous ratio.
2. Scaling: use `StandardScaler`, **fit on the training set only**, then transform both sets. Keep the fitted scaler (return it, and save with joblib) so the dashboard can reuse it.
   - Optional improvement: apply `np.log1p` to the heavily skewed features (`miss_distance`, `estimated_diameter`) before scaling and explain why in a comment.
3. Imbalance handling - implement **both** strategies behind one function `handle_imbalance(X_train, y_train, method)`:
   - `"smote"`: `imblearn.over_sampling.SMOTE`, applied **to the training set only**.
   - `"class_weight"`: return data unchanged, and provide a helper that computes `scale_pos_weight = n_negative / n_positive` for XGBoost and `class_weight="balanced"` for Random Forest.
   - `"none"`: baseline with no handling (useful for the comparison story).
4. Return a small dict/dataclass bundling: `X_train, X_test, y_train, y_test, scaler, feature_names`.
5. Print / return before-and-after class counts so the effect of SMOTE is visible.

### Deliverables
- `src/preprocessing.py` with the functions above
- `tests/test_preprocessing.py`

### Acceptance criteria
- Train/test hazardous ratios match within ~1 percentage point (stratification works)
- After SMOTE the training classes are balanced (~50/50), while the **test set is untouched and still imbalanced**
- A test confirms the scaler's mean/std come from training data only (e.g. train columns have mean ~0, std ~1; test columns generally do not)
- No test-set rows ever appear in the resampled training data

---

## PHASE 3 - Model Training & Evaluation (the Core Comparison)

**Objective:** Train Random Forest and XGBoost fairly and compare them with imbalance-appropriate metrics.

### Tasks

1. In `src/models.py`:
   - `build_random_forest(n_estimators, max_depth, class_weight=None, random_state=42)` -> `RandomForestClassifier`
   - `build_xgboost(n_estimators, max_depth, learning_rate, scale_pos_weight=1.0, random_state=42)` -> `XGBClassifier` (use `eval_metric="logloss"`)
   - `train_models(data_bundle, rf_params, xgb_params, imbalance_method)` -> returns fitted models plus training time for each.
2. In `src/evaluation.py`, write `evaluate_model(model, X_test, y_test)` returning a dict with:
   - `recall` (sensitivity), `precision`, `f1`, `roc_auc`, `pr_auc` (average precision), `accuracy` (reported but **flagged as secondary**)
   - `confusion_matrix` (2x2 array), and derived `false_negative_rate` and `false_positive_rate`
   - ROC curve points (`fpr`, `tpr`) and predicted probabilities
3. Add `compare_models(results_rf, results_xgb)` producing a tidy DataFrame (one row per model) so it can be shown as a table and exported.
4. **Fair comparison extras (these earn marks):**
   - 5-fold **stratified cross-validation** on the training data reporting mean +/- std for Recall, F1, ROC-AUC for both models. Do the resampling *inside* each fold (use `imblearn.pipeline.Pipeline`) to avoid leakage.
   - **Threshold analysis:** show how Recall/Precision change as the decision threshold moves from 0.1 to 0.9, and report the threshold that maximizes F1 for each model. Explain that for hazard detection you may deliberately lower the threshold to catch more true hazards.
   - Compare the three imbalance strategies (`none` vs `class_weight` vs `smote`) for both models in one summary table.
5. Feature importance: extract `feature_importances_` from XGBoost (and Random Forest) into a sorted DataFrame.

### Deliverables
- `src/models.py`, `src/evaluation.py`, `tests/test_models.py`
- A printed results table with both models' metrics

### Acceptance criteria
- Both models train in a reasonable time on 5,000 rows (seconds, not minutes)
- All metrics are computed on the **untouched test set**
- ROC-AUC for both models is clearly above 0.5 and the numbers are plausible (not suspiciously perfect 1.00 - if so, revisit the label logic from Phase 1)
- Cross-validation results are produced without leakage
- Tests confirm metric dict keys exist and confusion matrix sums to the test-set size

---

## PHASE 4 - Streamlit Dashboard

**Objective:** A professional, interactive web app (`app.py`) that exposes the whole pipeline.

### Layout

**Page config:** wide layout, title "Near-Earth Object Hazard Classification", a short subtitle explaining Bagging vs Boosting.

**Sidebar controls:**
- Data source: Mock (default) / Upload CSV
- Mock-data settings: number of rows (1,000-20,000), hazardous rate slider (3-15%)
- Imbalance method: `SMOTE` / `Class weighting` / `None`
- Random Forest: `n_estimators` (50-500), `max_depth` (3-30)
- XGBoost: `n_estimators` (50-500), `max_depth` (2-12), `learning_rate` (0.01-0.5)
- Test size slider
- **"Train / Re-train Models" button**

**Behavior requirements:**
- Use `st.session_state` to store trained models and results so the page does not retrain on every interaction.
- Use `@st.cache_data` for data generation. Show `st.spinner` while training.
- Show a friendly message before the first training run.

**Main page sections (use `st.tabs` or clearly separated sections):**

1. **Overview / metrics header** - `st.metric` cards: dataset size, hazardous %, and for each model Recall, F1, ROC-AUC. Show a comparison table beneath, highlighting the best value per metric. Add a small note explaining why accuracy is not the headline metric.

2. **Visual 1 - Risk Matrix** (Plotly scatter)
   - x = `relative_velocity`, y = `miss_distance` (log-scale y-axis is recommended because of the wide range)
   - Marker size = `estimated_diameter`
   - **Strict semantic colours: Red = Hazardous, Blue = Safe.** Use `color_discrete_map={"Hazardous": "red", "Safe": "blue"}` with a text label column, not a numeric one
   - Hover shows all four features; legend labelled clearly; sample down to ~2,000 points for speed if needed, but always keep **all** hazardous points

3. **Visual 2 - Confusion Matrices** (side by side, `st.columns(2)`)
   - Heatmaps for Random Forest and XGBoost with counts annotated and labels "Safe"/"Hazardous"
   - Use the same colour scale on both so they are visually comparable
   - Under each, show the **false negative count and rate** in bold - this is the headline comparison

4. **Visual 3 - ROC Curves** (single Plotly chart)
   - Both models overlapped, AUC in the legend names, dashed diagonal "random guess" reference line, axis titles and a title

5. **Visual 4 - Feature Importance** (horizontal bar chart)
   - XGBoost feature importances sorted ascending so the largest is on top; optional toggle to also show Random Forest

6. **Bonus (recommended):** a threshold slider that updates precision/recall live, and a "Try your own asteroid" form where the user enters the four feature values and sees each model's hazard probability.

7. **Conclusion box** - auto-generated text stating which model had the higher recall / lower false-negative rate / higher ROC-AUC for the current run.

### Deliverables
- `app.py` and `src/plots.py` (one function per figure)

### Acceptance criteria
- `streamlit run app.py` launches with no errors and the app is usable with default settings
- Changing sidebar values and clicking the button retrains and updates **all** visuals
- Colours are strictly Red = Hazardous, Blue = Safe on the risk matrix
- Both confusion matrices appear side by side with a shared colour scale
- Claude Code starts the app in the background (or uses Streamlit's `AppTest`) to confirm there are no runtime exceptions

---

## PHASE 5 - Testing, Documentation & Final Report

**Objective:** Make the project submission-ready and defensible in front of a teacher.

### Tasks

1. **Tests:** run `pytest` and fix any failures. Add a smoke test that runs the full pipeline end to end on a small mock dataset (e.g. 800 rows).
2. **`README.md`:** project overview, setup (`python -m venv`, `pip install -r requirements.txt`), how to run (`streamlit run app.py`), project structure, and screenshots placeholder section.
3. **`REPORT.md`** (the written analysis - about 2 pages) with these sections:
   - **Introduction:** what NEOs are, why hazard classification matters
   - **Dataset:** source, features, class imbalance (include the actual numbers from the run)
   - **Methodology:** preprocessing, SMOTE vs class weights, why stratified split, why no leakage
   - **Models:** plain-language explanation of Bagging (Random Forest) vs Boosting (XGBoost) and the main hyperparameters
   - **Evaluation metrics:** why Recall, F1, ROC-AUC, PR-AUC instead of accuracy; why false negatives are the costly error
   - **Results:** table of metrics, the cross-validation results, the imbalance-strategy comparison, and which model wins **and why**
   - **Limitations:** synthetic data caveat (real NASA labels depend on orbital parameters like MOID not fully captured here), possible overfitting, limited features
   - **Conclusion & future work:** e.g. hyperparameter search, adding orbital features, trying LightGBM, probability calibration
   - Fill in the real numbers from an actual run - **do not invent results**
4. **Reproducibility:** save a sample results table to `results/metrics_summary.csv` and a `results/` folder with exported figures (PNG) of the confusion matrices, ROC curve and feature importance for use in slides/report.
5. **Final checklist run:** fresh virtual environment -> install -> run tests -> launch app -> confirm every item in the checklist below.

### Final checklist
- [ ] Mock data: 5,000 rows, 5-10% hazardous
- [ ] Scaling fitted on train only; SMOTE on train only
- [ ] Random Forest and XGBoost both trained and evaluated on the untouched test set
- [ ] Recall, F1, ROC-AUC (and PR-AUC) reported; accuracy not the headline
- [ ] Sidebar hyperparameters + retrain button work live
- [ ] Visual 1: Red = Hazardous, Blue = Safe, size = diameter
- [ ] Visual 2: side-by-side confusion matrices with FN rates shown
- [ ] Visual 3: overlapping ROC curves with AUC in legend
- [ ] Visual 4: horizontal XGBoost feature importance
- [ ] Tests pass; README and REPORT complete with real numbers

---

## Notes to Claude Code on Quality and Honesty

- If a model scores near-perfect (AUC ~ 1.0), treat that as a warning sign of leakage or a trivially separable synthetic label, and fix the data generator rather than celebrating.
- Report the results you actually get. If Random Forest beats XGBoost in a given run, say so and explain plausible reasons (e.g. small tabular dataset, default hyperparameters).
- Keep the code simple and readable; this is a student project and the author must be able to explain every part of it.