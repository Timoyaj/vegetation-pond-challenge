# Zindi GeoAI Aquaculture Pond Identification — ML Plan

## Background

**Task**: Binary classification — aquaculture pond (1) vs. other land cover (0) from tabular satellite features.  
**Data**: 146 columns — `ID`, `label`, then 144 feature columns (12 time steps × 12 bands each: `VH`, `VV`, `blue`, `green`, `nir`, `nira`, `re1`, `re2`, `re3`, `red`, `swir1`, `swir2`). 1,821 training samples, ~40.4% pond.  
**Metric**: Weighted composite = **F1 × 0.6 + ROC-AUC × 0.4**  
**Core difficulty**: Test data has only 4–6 valid months; the rest are masked as **-9999**. Models must generalize across this temporal domain shift.

---

## ✅ Decisions Confirmed

| Question | Answer |
|----------|--------|
| Notebook format | `.ipynb` Jupyter notebooks |
| Model scope | **Tree ensembles only** (no 1D-CNN / LSTM) |
| Goal | **Top leaderboard rank** + reproducible solution |

> [!NOTE]
> For top-rank performance we add: **Optuna hyperparameter tuning** for LightGBM + XGBoost, **10× ROS augmentation** (more diverse masking), **pseudo-labeling** on high-confidence test predictions, and a **stacking meta-learner** instead of simple rank averaging.

---

## Proposed Changes

### Phase 1 — Environment & Data Exploration

#### [NEW] `01_eda.ipynb`

Exploratory Data Analysis notebook:
- Load `Train.csv` and `Test.csv`
- Replace `-9999` → `NaN`
- Compute `n_valid_optical`, `n_valid_sar` per sample
- Class balance check and visualization (bar + pie)
- Missingness analysis: which time steps / bands are most masked in test?
- Histograms: VH, VV, NDWI (computed on t=01) by class
- Correlation heatmap of raw bands
- **Analysis markdown cells** after every plot explaining what we see

---

### Phase 2 — Feature Engineering

#### [NEW] `02_feature_engineering.py`

Standalone module — importable by all downstream notebooks:

```
Raw 144 columns → Feature matrix
```

**Step 1 — Preprocessing**
- Parse band/timestep structure from column names (regex: `{band}_{ts}`)
- Replace `-9999` → `NaN`
- Add `n_valid_optical` (count non-NaN across 10 optical bands × 12 steps)
- Add `n_valid_sar` (count non-NaN across VH, VV × 12 steps)

**Step 2 — Spectral Indices (per valid time step)**

| Index | Formula |
|-------|---------|
| NDWI | `(green − nir) / (green + nir)` |
| MNDWI | `(green − swir1) / (green + swir1)` |
| NDVI | `(nir − red) / (nir + red)` |
| EVI | `2.5 × (nir−red) / (nir + 6·red − 7.5·blue + 1)` |
| LSWI | `(nir − swir1) / (nir + swir1)` |
| NDRE | `(re1 − red) / (re1 + red)` |
| SAR_ratio | `VH − VV` (in dB space) |
| SAR_RVI | `4·VH / (VV + VH)` |

**Step 3 — Temporal Aggregation (NaN-aware, over valid months only)**

Per raw band and per derived index:
- mean, std, min, max, amplitude (max−min), p25, p75, skewness, linear_trend_slope

This yields: `(12 raw bands + 8 derived indices) × 10 stats = ~200 aggregate features` + 2 validity counts ≈ **~202 engineered features**

**Step 4 — ROS Augmentation (training only)**
- For each training sample, generate 3 augmented copies
- Each copy: randomly mask 6–8 of the 12 months (set to NaN)
- Recompute all aggregate features on the masked data
- Final augmented training set: ~1,821 × 4 ≈ 7,284 samples

> [!IMPORTANT]
> ROS augmentation is applied **after** the train/val split. The validation fold uses **unaugmented** original data to measure clean performance. The augmentation is only in the training fold.

---

### Phase 3 — Model Training & Evaluation

#### [NEW] `03_models.ipynb`

**ML Best Practices Classification Workflow:**

1. **Split before fitting**
   - Stratified 5-fold CV (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`)
   - All preprocessing fitted on training fold, applied to validation fold

2. **Handle missing values**
   - After aggregation, any remaining NaN (e.g. all months masked) → impute with median (fitted on train fold)
   - Document imputation counts

3. **Handle class imbalance**
   - Use `scale_pos_weight = n_neg / n_pos ≈ 1.47` for gradient boosting
   - Evaluate with class-weighted F1

4. **Models trained and compared** (identical CV folds for all):

| Model | Role | Key Config |
|-------|------|-----------|
| Logistic Regression | Baseline | `class_weight='balanced'` |
| Random Forest | Strong baseline | `n_estimators=300, class_weight='balanced'` |
| LightGBM | Primary | `scale_pos_weight=1.47, early_stopping` + **Optuna 50 trials** |
| XGBoost | Primary | `scale_pos_weight=1.47, early_stopping` + **Optuna 50 trials** |
| CatBoost | Diversity | `auto_class_weights='Balanced', eval_metric='F1'` + Optuna |

5.5. **Pseudo-labeling round** (top-rank addition)
   - After initial training, predict test set probabilities
   - Add high-confidence test samples (prob > 0.92 or < 0.08) to training with pseudo-labels
   - Retrain LightGBM + XGBoost on augmented set; expect +1–2% F1

5. **Evaluation per model (on out-of-fold validation)**:
   - F1-score, ROC-AUC, Precision-Recall AUC
   - Confusion matrix
   - Composite score = F1×0.6 + AUC×0.4

6. **Probability calibration**
   - Apply `CalibratedClassifierCV(method='isotonic')` to the best model
   - Verify calibration curve is well-aligned

7. **SHAP analysis** on best model:
   - Feature importance bar chart (top 30)
   - SHAP beeswarm plot
   - Identify which bands / time steps / indices drive predictions

8. **Analysis markdown cells** after every model's results

---

### Phase 4 — Ensemble & Submission

#### [NEW] `04_submission.ipynb`

- **Stacking meta-learner**: train a Logistic Regression on out-of-fold probabilities from LightGBM + XGBoost + CatBoost + RF → learns optimal blend weights per model
- Apply isotonic calibration to stacked output
- **Test-time ROS**: generate 10 masked copies of each test sample, average their probabilities → reduces variance from partial observation
- Threshold = 0.5 for `TargetF1` (no tuning per rules)
- `TargetRAUC` = raw calibrated stacked probability
- Output `submission.csv` matching `SampleSubmission.csv` format
- Validation: assert row count matches, no NaN, all `TargetF1` ∈ {0,1}, all `TargetRAUC` ∈ [0,1]

---

## File Layout

```
vegetation-pond-challenge/
├── Train.csv
├── Test.csv
├── SampleSubmission.csv
├── challenge data disc.pdf
├── literature_review.md          # existing
├── prompt_draft.md               # existing
├── 02_feature_engineering.py     # [NEW] importable feature module
├── 01_eda.ipynb                  # [NEW] EDA notebook
├── 03_models.ipynb               # [NEW] training + eval notebook
├── 04_submission.ipynb           # [NEW] ensemble + submission notebook
└── submission.csv                # [NEW] final output
```

---

## Verification Plan

### Automated Tests
- `assert df_sub.shape[0] == df_sample.shape[0]` — row count matches
- `assert set(df_sub['TargetF1'].unique()).issubset({0,1})` — binary check
- `assert df_sub['TargetRAUC'].between(0,1).all()` — probability range check
- `assert df_sub.isna().sum().sum() == 0` — no NaN in submission

### Performance Targets
- Best model composite CV score ≥ 0.80 (F1×0.6 + AUC×0.4)
- LightGBM expected: F1 ~0.85–0.90, AUC ~0.92–0.95

### Manual Verification
- Review SHAP plots — MNDWI and SAR features should rank highly
- Review confusion matrix — false positive rate should be low (water bodies misclassified)
- Check EDA for any unexpected data quality issues before training
