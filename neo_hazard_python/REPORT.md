# Near-Earth Object Hazard Classification
## A Comparative Study of Random Forest (Bagging) vs XGBoost (Boosting) on Real NASA Data

*All numbers below come from `python run_analysis.py` (real NASA data, seed 42, SMOTE, default
hyperparameters). The full tables are in `results/`.*

---

### 1. Introduction

Near-Earth Objects (NEOs) are asteroids and comets whose orbits bring them close to Earth. NASA
labels an asteroid a **Potentially Hazardous Asteroid (PHA)** when it is

* large enough to cause regional damage: absolute magnitude **H ≤ 22** (about 140 m or bigger), **and**
* its orbit passes close to Earth's: **MOID ≤ 0.05 AU** (Minimum Orbit Intersection Distance, about 7.5 million km).

Telescopes discover thousands of NEOs a year, so an automatic classifier that flags likely hazards
helps astronomers decide what to follow up first. Missing a real hazard (a false negative) is far
more costly than a false alarm, and that drives how the models are judged.

### 2. Dataset

Three real NASA files (in `data/raw/`) are combined:

| File | Rows | What it gives us |
|---|---|---|
| `near_earth_asteroids_2025.csv` | 41,281 asteroids | `H`, `diameter_km`, and the real NASA label `pha` (2,539 PHAs) |
| `close_approaches_1960_2060.csv` | 25,586 fly-bys | distance (AU) and relative speed (km/s) of each Earth approach |
| `close_approaches_2015_2035.csv` | 27,430 fly-bys | the same, in a different format |

**Building the modelling table** (`src/data.py → build_nasa_dataset`):

1. Stack both close-approach files (53,016 rows) and drop duplicates, because the files overlap for
   2015–2035. That leaves 39,719 unique approaches by 24,044 asteroids.
2. Keep **one row per asteroid**, its *closest* recorded approach. If an asteroid with 5 fly-bys
   stayed in 5 times, near-identical copies could land in both train and test (leakage).
3. Join on the asteroid designation to attach `H`, diameter and the `pha` label, and drop rows
   with missing values.

**Result: 23,964 real asteroids, 22,379 Safe (93.4%) and 1,585 Hazardous (6.6%).** Approach dates
run from 1960 to 2059.

| Feature | Unit | Source |
|---|---|---|
| `estimated_diameter` | km | asteroid catalogue (`diameter_km`) |
| `relative_velocity` | km/s | close-approach file (`v_rel`) |
| `miss_distance` | km | close-approach file (AU × 149,597,870.7) |
| `absolute_magnitude` (H) | – | asteroid catalogue (`H`) |
| `is_potentially_hazardous_asteroid` | bool | asteroid catalogue (`pha`), the target |

Diameter and H are almost the same information (Spearman ρ = −0.999), because NASA usually
estimates diameter *from* H when it has not been measured directly.

After the stratified 80/20 split: train 17,903 Safe / 1,268 Hazardous; test 4,476 Safe / 317 Hazardous.

### 3. Methodology

1. **Stratified split.** Both sets keep the 6.6% hazardous ratio.
2. **Log transform and scaling.** `miss_distance` and `estimated_diameter` span several orders
   of magnitude, so `log1p` is applied first. A `StandardScaler` is then **fitted on the training
   set only** and applied to both sets.
3. **Imbalance handling**, compared three ways:
   * **SMOTE** creates synthetic hazardous examples by interpolating between neighbouring
     hazardous objects. It is applied to the training set only (1,268 → 17,903 hazardous); the test
     set is never touched.
   * **Class weighting** keeps the data as it is, but mistakes on the rare class cost more
     (`class_weight="balanced"` for RF, `scale_pos_weight = 17903/1268 ≈ 14.1` for XGBoost).
   * **None**, as a baseline.
4. **No leakage.** In 5-fold cross-validation the log transform, scaler and SMOTE sit inside an
   `imblearn` Pipeline, so they are re-fitted on each training fold. Unit tests check that the scaler
   only saw training rows and that no test row appears in the resampled training data. MOID is
   deliberately **not** used as a feature: it is half of the label's definition, so including it
   would make the task trivial.

### 4. Models

* **Random Forest (bagging).** 200 deep trees (max_depth 12). Each tree is trained on a bootstrap
  sample and looks at a random subset of features at each split. Averaging many different trees
  mainly **reduces variance**.
* **XGBoost (boosting).** 200 shallow trees (max_depth 4, learning_rate 0.1) are added **one after
  another**, and each new tree fits the errors of the current ensemble. This mainly **reduces bias**.
  The learning rate shrinks each tree's contribution to limit overfitting.

### 5. Evaluation metrics

With 93.4% Safe objects, a model that always says "Safe" gets **93.4% accuracy** and catches
**zero** hazards, so accuracy is only a secondary metric. The headline metrics are:

* **Recall:** the share of real hazards caught. A **false negative** is a hazardous asteroid called safe.
* **Precision** and **F1:** F1 balances catching hazards against false alarms.
* **ROC-AUC:** how well the model ranks hazardous above safe objects, at any threshold.
* **PR-AUC:** like ROC-AUC but focused on the rare class, so it is more honest on imbalanced data.

### 6. Results

**Test set (4,793 asteroids, 317 hazardous), SMOTE, threshold 0.5**

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC | Accuracy | FN | FN rate | Train time |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.969 | 0.766 | 0.855 | 0.997 | 0.950 | 0.978 | 10 | 3.2% | 0.85 s |
| XGBoost | **0.975** | **0.771** | **0.861** | **0.997** | **0.952** | **0.979** | **8** | **2.5%** | 2.25 s* |

\*XGBoost's first fit includes one-off library start-up time.

**5-fold stratified cross-validation (training data, SMOTE inside each fold)**

| Model | Recall | F1 | ROC-AUC |
|---|---|---|---|
| Random Forest | 0.968 ± 0.019 | 0.865 ± 0.018 | 0.996 ± 0.001 |
| XGBoost | 0.968 ± 0.012 | 0.868 ± 0.015 | 0.997 ± 0.001 |

**Imbalance strategies (test set)**

| Strategy | Model | Recall | Precision | F1 | ROC-AUC | PR-AUC | FN |
|---|---|---|---|---|---|---|---|
| none | Random Forest | 0.864 | 0.913 | **0.888** | 0.996 | 0.959 | 43 |
| none | XGBoost | 0.861 | 0.916 | **0.888** | 0.997 | 0.957 | 44 |
| class_weight | Random Forest | **0.981** | 0.764 | 0.859 | 0.997 | 0.957 | **6** |
| class_weight | XGBoost | 0.962 | 0.763 | 0.851 | 0.997 | 0.954 | 12 |
| smote | Random Forest | 0.969 | 0.766 | 0.855 | 0.997 | 0.950 | 10 |
| smote | XGBoost | 0.975 | 0.771 | 0.861 | 0.997 | 0.952 | 8 |

Imbalance handling makes a big difference: **without it, the models miss 43–44 hazards; with
SMOTE or class weights they miss only 6–12.** In exchange, precision falls from about 0.91 to about 0.77 (more false
alarms). F1 is highest with "none", which is exactly why F1 alone is not the right objective here:
missing a hazard costs far more than a false alarm.

**Threshold analysis.** At threshold 0.10, recall rises to 0.997 (RF) and 0.994 (XGB), i.e. only 1–2
of 317 hazards are missed, while precision stays about 0.71–0.73. The F1-optimal thresholds are
0.80 (RF, F1 0.872) and 0.90 (XGB, F1 0.878). For planetary defence we would choose a
**low** threshold and accept more false alarms.

**Feature importance.** `absolute_magnitude` dominates (XGB 0.92, RF 0.51). RF splits the "size"
signal between H (0.51) and diameter (0.44) because the two are nearly identical, while XGBoost puts
almost all of it on H. `miss_distance` (≈0.02) and `relative_velocity` (≈0.01–0.03) matter
little. This matches NASA's definition, which depends on size and orbit, not speed.

**Why are the scores so high, and is it leakage?** It is not leakage. It follows from how the label
is defined. The close-approach files only list approaches within ~0.1 AU, so almost every object
already passes or nearly passes the MOID part of the rule, and the label mostly reduces to "H ≤ 22".
As a check, the **one-line rule `H ≤ 22` alone** gets recall 0.987, precision 0.760 and F1 0.859 on the
same test set, about the same as the ML models. Breaking the data down:

| | Hazardous | Safe |
|---|---|---|
| H ≤ 22 and closest approach ≤ 0.05 AU | 1,211 | 22 |
| H ≤ 22 and closest approach 0.05–0.1 AU | 347 | 467 |
| H > 22 | 27 | 21,890 |

The only genuinely hard cases are large asteroids whose closest *recorded* approach is
0.05–0.1 AU. Their label depends on MOID, an orbital quantity that is not among our four features.
This is where almost all false positives come from.

**Which model wins?** In this run **XGBoost is slightly ahead** with SMOTE (8 vs 10 missed hazards, and
marginally higher F1, ROC-AUC and PR-AUC). With class weighting, **Random Forest** is ahead (6 vs 12 missed).
Cross-validation shows identical mean recall (0.968) and ROC-AUC differences of 0.001, well inside
the fold-to-fold spread. The honest conclusion: **the two ensembles are statistically tied on this
data.** The problem is close to a single threshold on one feature, so there is little left for
boosting's bias-reduction or bagging's variance-reduction to improve.

### 7. Limitations

* **The label is (mostly) a rule on one of our features.** That makes the task easy and the high
  scores expected. A plain `H ≤ 22` rule is a strong baseline that the models barely beat.
* **Missing orbital features.** MOID, eccentricity and inclination exist in the asteroid file, but they
  are left out on purpose (MOID is part of the label definition). Adding orbit features would make
  this an orbit-classification problem rather than a "close approach" one.
* **Diameter is often estimated from H**, so the two size features are almost redundant.
* **Closest approach only.** One row per asteroid avoids leakage, but drops the other fly-bys.
* **Default hyperparameters, single split.** No tuning was done. With 317 test hazards, each missed
  object changes recall by about 0.3 points.

### 8. Conclusion and future work

On 23,964 real NASA asteroids, both ensembles catch about **97% of hazardous objects** with ROC-AUC
≈ 0.997, against 0% recall for the "always Safe" baseline that has 93% accuracy. Handling class
imbalance cut missed hazards from about 44 to 6–12. Random Forest and XGBoost are statistically tied. The data
explains why: NASA's PHA label is essentially "large AND orbit close to Earth's", and size is directly
in our features. Next steps:
* add orbital features (e, i, q, a) **without MOID** to see whether the models can learn the orbit-closeness part;
* hyperparameter search (`RandomizedSearchCV` inside the leakage-free pipeline);
* LightGBM / CatBoost, and probability calibration;
* choose the decision threshold from an explicit cost of a missed hazard versus a false alarm.
