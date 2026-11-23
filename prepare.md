# prepare.md: Everything You Need to Present This Project

Read this once tonight, then use the **Demo script** (section 9) and the **Teacher Q&A** (section 10) tomorrow.

---

## 1. The project in 30 seconds

> "I built a machine-learning project that predicts whether an asteroid is **potentially hazardous**
> to Earth, using **real NASA data** on about 24,000 asteroids. I compare two ensemble models:
> **Random Forest**, which uses *bagging*, and **XGBoost**, which uses *boosting*. Only 6.6% of
> asteroids are hazardous, so accuracy is misleading. I judge the models on **recall** (how many
> real hazards we catch), F1 and ROC-AUC, and I focus on **false negatives**, the hazards we miss.
> Both models catch about **97% of hazardous asteroids**. They are basically tied, and I can
> explain why. There is also an interactive **Streamlit dashboard**."

---

## 2. How to run it (practise this once before class)

```powershell
cd C:\Users\jaiga\OneDrive\Desktop\DA_Project\neo_hazard_python
.\.venv\Scripts\activate

python run_analysis.py     # prints every result table, saves CSVs + PNG charts in results/  (~15 s)
streamlit run app.py       # opens the dashboard in the browser at http://localhost:8501
pytest                     # runs 32 automated tests, all should pass
```

In the dashboard: keep **"Real NASA data"** selected and click **Train / Re-train Models** (it takes a few seconds).

If the venv ever breaks: `python -m venv .venv`, then `.\.venv\Scripts\activate`, then `pip install -r requirements.txt`.

---

## 3. The data (where it comes from and how it's built)

You were given 3 real NASA CSV files. They are now in `neo_hazard_python/data/raw/`:

| File | Rows | What's inside |
|---|---|---|
| `near_earth_asteroids_2025.csv` | 41,281 asteroids | size (**H**, diameter) and NASA's official label **`pha`** = Potentially Hazardous Asteroid (True/False) |
| `close_approaches_1960_2060.csv` | 25,586 fly-bys | for each time an asteroid passes Earth: **distance** and **speed** |
| `close_approaches_2015_2035.csv` | 27,430 fly-bys | same, different format, overlapping years |

**No single file has everything**, so the code (`src/data.py → build_nasa_dataset`) joins them:

1. Put both fly-by files together (53,016 rows) and **remove duplicates** (the files overlap for 2015–2035). That leaves 39,719.
2. Keep **one row per asteroid**: its **closest** fly-by.
   *Why?* An asteroid with 5 fly-bys would appear 5 times. Copies could land in both train and test, so the model would be tested on things it already saw (**data leakage**).
3. **Join** with the asteroid file using the asteroid's ID (designation) to get size and the label.
4. Drop rows with missing values. The result is saved as `data/neo_nasa.csv`.

**Final dataset: 23,964 asteroids, of which 22,379 are Safe (93.4%) and 1,585 are Hazardous (6.6%).** This is heavily imbalanced.

### The 4 features (inputs) + 1 target (output)

| Column | Meaning | Unit |
|---|---|---|
| `estimated_diameter` | how big the asteroid is | km |
| `relative_velocity` | how fast it passes Earth | km/s |
| `miss_distance` | how close it comes to Earth | km |
| `absolute_magnitude` (H) | brightness. **Lower H = bigger asteroid** | – |
| `is_potentially_hazardous_asteroid` | **target**: hazardous or not (NASA's label) | True/False |

### NASA's real definition of "hazardous" (know this!)
An asteroid is a PHA if **both** are true:
- **H ≤ 22** (roughly bigger than 140 m), **and**
- **MOID ≤ 0.05 AU** (its orbit comes within about 7.5 million km of Earth's orbit).
  MOID = *Minimum Orbit Intersection Distance*. 1 AU = Earth–Sun distance ≈ 150 million km.

---

## 4. The pipeline, step by step

```
3 raw CSVs ─► join + clean ─► stratified 80/20 split ─► log + scale (fit on TRAIN only)
           ─► handle imbalance (SMOTE on TRAIN only) ─► train RF + XGBoost
           ─► evaluate on untouched TEST set ─► charts, tables, dashboard
```

1. **Stratified split (80% train, 20% test).** "Stratified" means both parts keep the same 6.6% hazardous ratio.
   Train: 17,903 safe / 1,268 hazardous. Test: 4,476 safe / 317 hazardous.
2. **Log transform.** Distance and diameter range from tiny to huge (diameter goes from 0.0005 km to 7 km), so we take `log` to squash them.
3. **StandardScaler.** Puts every feature on the same scale (mean 0, std 1). It is **fitted on training data only**, then applied to test. If we fitted it on test too, information from the test set would leak into training.
4. **Imbalance handling.** 3 options are compared:
   - **SMOTE**: creates *synthetic* hazardous examples by drawing points between real hazardous neighbours. Training hazardous goes 1,268 → 17,903 (50/50). **Only on train, never on test.**
   - **Class weighting**: no new data. Mistakes on hazardous asteroids just **count ~14× more** (17,903 / 1,268 ≈ 14.1).
   - **None**: the baseline.
5. **Train both models** on the same data.
6. **Evaluate on the untouched test set.**
7. **5-fold cross-validation**: the training data is split 5 ways. Each part takes a turn as the validation set, which gives mean ± std (a more reliable result than one split). SMOTE and scaling happen *inside* each fold, so there is still no leakage.

---

## 5. The two models (explain like this)

| | **Random Forest** | **XGBoost** |
|---|---|---|
| Family | **Bagging** (Bootstrap AGGregatING) | **Boosting** |
| How trees are built | **In parallel**, independently | **One after another** (sequential) |
| Each tree sees | a random sample of rows (with replacement) + random subset of features | all data, but focuses on the **errors** of the previous trees |
| Final answer | **vote / average** of all trees | **weighted sum** of all trees |
| Mainly reduces | **variance** (overfitting) | **bias** (underfitting) |
| Our settings | 200 trees, max_depth 12 (deep) | 200 trees, max_depth 4 (shallow), learning_rate 0.1 |

**Analogy:**
- *Random Forest* = ask 200 people independently and take the majority vote.
- *XGBoost* = 200 students in a row. Each one studies **only the questions the previous ones got wrong**.

**Learning rate** (XGBoost): how much each new tree is allowed to change the answer. Small = slower but safer against overfitting.

---

## 6. Metrics: why NOT accuracy

**Key argument:** a dumb model that always says "Safe" gets **93.4% accuracy** but catches **0 hazards**. Accuracy is useless here.

**Confusion matrix:**

| | Predicted Safe | Predicted Hazardous |
|---|---|---|
| **Actually Safe** | True Negative (TN) | False Positive (FP) = false alarm |
| **Actually Hazardous** | **False Negative (FN) = MISSED HAZARD, the worst error** | True Positive (TP) |

| Metric | Formula | Plain meaning |
|---|---|---|
| **Recall** | TP / (TP + FN) | Of all real hazards, how many did we catch? **Most important.** |
| Precision | TP / (TP + FP) | Of everything we flagged, how much was really hazardous? |
| **F1** | harmonic mean of precision & recall | balance of the two |
| **ROC-AUC** | area under the ROC curve (0.5 = random, 1 = perfect) | how well the model *ranks* hazardous above safe |
| **PR-AUC** | area under the precision-recall curve | like ROC-AUC but focused on the rare class, so better for imbalanced data |

**Why false negatives matter most:** a false alarm just means a telescope takes another look.
A missed hazard could be an asteroid hitting Earth.

---

## 7. Results (real numbers, memorise the bold ones)

### Main result: test set of 4,793 asteroids (317 hazardous), using SMOTE

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC | Accuracy | Missed hazards (FN) |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.969 | 0.766 | 0.855 | 0.997 | 0.950 | 0.978 | **10 of 317** |
| XGBoost | **0.975** | 0.771 | 0.861 | 0.997 | 0.952 | 0.979 | **8 of 317** |

**Both catch about 97% of hazardous asteroids. XGBoost is very slightly ahead in this run.**

### Cross-validation (5-fold, on training data)

| Model | Recall | F1 | ROC-AUC |
|---|---|---|---|
| Random Forest | 0.968 ± 0.019 | 0.865 ± 0.018 | 0.996 ± 0.001 |
| XGBoost | 0.968 ± 0.012 | 0.868 ± 0.015 | 0.997 ± 0.001 |

The recall is the **same**, and the differences are smaller than the ± spread. So the two models are **statistically tied**.

### Imbalance strategies (the most interesting table)

| Strategy | RF missed hazards | XGB missed hazards | Precision |
|---|---|---|---|
| None | **43** | **44** | ~0.91 |
| Class weight | **6** | 12 | ~0.76 |
| SMOTE | 10 | **8** | ~0.77 |

**Main takeaway:** handling imbalance cut missed hazards from **~44 to 6–12**. The price is more
false alarms (precision drops from 0.91 to 0.77). That's a good trade for planetary defence.

### Threshold
The model outputs a probability. By default, "hazardous" means probability ≥ 0.5.
If we **lower the threshold to 0.1**, recall goes to **0.997 (RF)**, i.e. only 1 of 317 hazards is missed,
and precision only drops to about 0.71. For safety we'd use a **low threshold**. (Shown live in the dashboard's "Threshold" tab.)

### Feature importance
- `absolute_magnitude` (size) is by far the most important (XGBoost 0.92, RF 0.51).
- Random Forest splits the importance between H (0.51) and diameter (0.44) because they carry the same information (correlation −0.999).
- Distance and speed barely matter (~0.01–0.03).

---

## 8. The KEY insight (this is what impresses a teacher)

**"Why are the scores so high (ROC-AUC 0.997)? Is it cheating or leakage?"**

No. It comes from **how NASA defines the label**:
- PHA = **H ≤ 22** AND **MOID ≤ 0.05 AU**.
- The fly-by files only contain asteroids that came within about 0.1 AU of Earth, so nearly all of them already (almost) pass the distance part.
- That means the label is **mostly just "H ≤ 22"**, and H is one of our features.

**Proof:** the one-line rule `if H <= 22 → hazardous` gets **recall 0.987, F1 0.859** on the same test set, about the same as the ML models.

| | Hazardous | Safe |
|---|---|---|
| Big (H ≤ 22) **and** came within 0.05 AU | 1,211 | 22 |
| Big (H ≤ 22) but only came within 0.05–0.1 AU | 347 | 467 |
| Small (H > 22) | 27 | 21,890 |

The **only hard cases** are big asteroids whose closest approach was 0.05–0.1 AU. For those, the answer
depends on **MOID**, an orbital value that is **not** in our 4 features. That's where nearly all mistakes come from.

**I deliberately did NOT use MOID as a feature:** it's half of the label's definition, so the model would just copy the rule (that *would* be leakage).

**Why are the models tied?** The problem is almost one threshold on one feature, so there's little left
for boosting to fix or for bagging to smooth out. Both models find the same rule.

---

## 9. Demo script (≈5 minutes)

1. **Intro (30 s):** say section 1.
2. **Data (1 min):** open `neo_hazard_python/data/raw/`. "3 NASA files, joined on asteroid ID, one row per asteroid, 23,964 rows, 6.6% hazardous."
3. **Run the dashboard:** `streamlit run app.py`, keep "Real NASA data", click **Train**.
4. Walk through the tabs:
   - **Overview:** metric cards. "Accuracy isn't the headline, recall is."
   - **Risk Matrix:** red = hazardous, blue = safe, bubble size = diameter. "The red ones are the big ones."
   - **Confusion Matrices:** side by side. Point at the **false negatives** (RF 10, XGB 8).
   - **ROC Curves:** both hug the top-left corner. AUC 0.997.
   - **Feature Importance:** absolute magnitude dominates. "This matches NASA's definition."
   - **Threshold & Try It:** drag the threshold down and recall goes up. Type in a big asteroid (low H) and watch the probability go high.
   - **Conclusion:** auto-generated summary.
5. Change the imbalance method to **None**, retrain, and show that **false negatives jump to ~44**. *(This is the best live moment.)*
6. **Wrap up:** say section 8 (why the scores are high and why the models tie) and the limitations.

---

## 10. Teacher Q&A (likely questions + answers)

**Q: What is bagging vs boosting?**
Bagging trains many models independently on random bootstrap samples and averages them (reduces variance). Boosting trains models in sequence, each fixing the last one's errors (reduces bias).

**Q: Why not use accuracy?**
93.4% of asteroids are safe, so "always safe" gives 93.4% accuracy with zero hazards found.

**Q: What is SMOTE?**
Synthetic Minority Over-sampling Technique. It creates new fake hazardous examples by interpolating between real hazardous neighbours. It is only applied to training data.

**Q: Why apply SMOTE / scaling only on training data?**
The test set must represent new, unseen, real-world data. If we touched it, the results would be over-optimistic (data leakage).

**Q: What is stratified splitting?**
It keeps the same class ratio (6.6% hazardous) in train and test. Without it, the test set might get too few hazardous asteroids.

**Q: What is cross-validation and why use it?**
Split the training data into 5 folds and train/validate 5 times, each fold taking a turn as validation. It gives mean ± std, so we know the result isn't luck from one split.

**Q: Your AUC is 0.997, isn't that suspicious?**
See section 8: the label is mostly "H ≤ 22", and the simple rule gets the same F1. It's not leakage. I deliberately excluded MOID, which *would* be leakage.

**Q: Which model is better?**
On this data they are statistically tied: same CV recall (0.968), AUC difference 0.001. XGBoost was slightly ahead with SMOTE (8 vs 10 misses), RF with class weights (6 vs 12). With a richer problem (more features), boosting usually has the edge.

**Q: Why log-transform?**
Distance and diameter span several orders of magnitude. Log stops a few huge values from dominating the scaler, and it gives SMOTE a sensible space to interpolate in.

**Q: Do tree models even need scaling?**
Not really, since trees split on thresholds. But SMOTE uses distances between points, so it needs scaled features. Scaling also keeps the pipeline consistent.

**Q: What's the decision threshold?**
The probability cut-off above which we call something hazardous. Default 0.5. Lowering it catches more hazards but gives more false alarms. For asteroids, a low threshold is safer.

**Q: Why one row per asteroid?**
To avoid the same asteroid appearing in both train and test (leakage). We keep its closest approach.

**Q: Limitations?**
(1) The label is mostly a rule on one feature, so the task is easy. (2) There are no orbital features (MOID, eccentricity, inclination). (3) Diameter is often estimated from H, so the two size features are nearly redundant. (4) No hyperparameter tuning. (5) Only the closest fly-by is kept.

**Q: Future work?**
Add orbital features (without MOID), tune hyperparameters with RandomizedSearchCV, try LightGBM/CatBoost, calibrate probabilities, and choose the threshold from a cost of a miss vs a false alarm.

**Q: How do you know your code works?**
32 automated tests with `pytest`. They check the data shape and ratio, that the scaler is fitted only on train, that SMOTE never touches test, that the metrics are correct, a full end-to-end run, and that the dashboard loads and trains.

**Q: What is RANDOM_STATE = 42?**
A fixed random seed, so every run gives exactly the same numbers (reproducibility).

---

## 11. File-by-file guide (if the teacher opens the code)

```
DA_Project/
├── prepare.md                 ← this file
├── basic.md                   ← original project brief
└── neo_hazard_python/
    ├── app.py                 ← Streamlit dashboard (UI only)
    ├── run_analysis.py        ← runs everything, prints tables, saves results/
    ├── REPORT.md              ← the written report with all numbers
    ├── README.md              ← setup instructions
    ├── requirements.txt       ← libraries
    ├── data/raw/              ← the 3 NASA CSVs
    ├── data/neo_nasa.csv      ← joined dataset (auto-generated)
    ├── results/               ← output tables (CSV) + charts (PNG) for slides
    ├── src/
    │   ├── data.py            ← loads/joins NASA data (+ mock generator for tests)
    │   ├── preprocessing.py   ← split, log, scaling, SMOTE / class weights
    │   ├── models.py          ← builds RF & XGBoost, trains, cross-validation
    │   ├── evaluation.py      ← recall, F1, ROC-AUC, confusion matrix, thresholds
    │   ├── pipeline.py        ← connects everything end-to-end
    │   └── plots.py           ← all charts
    └── tests/                 ← 32 pytest tests
```

**Charts ready for slides:** `results/confusion_matrices.png`, `results/roc_curves.png`, `results/feature_importance.png`.

**Libraries used:** pandas, numpy (data), scikit-learn (Random Forest, metrics, scaling, CV), xgboost,
imbalanced-learn (SMOTE), matplotlib/seaborn/plotly (charts), streamlit (dashboard), pytest (tests).

---

## 12. One-line answers to memorise

- **Dataset:** 23,964 real NASA asteroids, 6.6% hazardous.
- **Best result:** about 97% of hazards caught, ROC-AUC 0.997.
- **Missed hazards:** RF 10, XGBoost 8 (out of 317), vs 44 without imbalance handling.
- **Winner:** statistically tied. Explain why with H ≤ 22.
- **Most important feature:** absolute magnitude (size).
- **Biggest lesson:** imbalance handling + recall matter more than which ensemble you pick.
