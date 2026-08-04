# Vegetation Pond Challenge — Task List

## Phase 0 — Environment
- [ ] Check Python + package availability (lightgbm, xgboost, catboost, optuna, shap)
- [ ] Install missing packages

## Phase 1 — EDA
- [ ] Create `01_eda.ipynb`
  - [ ] Load Train.csv + Test.csv
  - [ ] Replace -9999 → NaN, compute n_valid counts
  - [ ] Class balance analysis + chart
  - [ ] Missingness analysis by time step / band
  - [ ] Distribution plots by class
  - [ ] Analysis markdown cells

## Phase 2 — Feature Engineering
- [ ] Create `02_feature_engineering.ipynb`
  - [ ] Band/timestep parser from column names
  - [ ] 8 spectral indices (NDWI, MNDWI, NDVI, EVI, LSWI, NDRE, SAR_ratio, SAR_RVI)
  - [ ] 10 temporal aggregation stats (mean, std, min, max, amplitude, p25, p75, skew, slope)
  - [ ] n_valid_optical, n_valid_sar features
  - [ ] ROS augmentation function (10x, mask 6-8 months)
  - [ ] Save engineered feature DataFrames

## Phase 3 — Model Training
- [ ] Create `03_models.ipynb`
  - [ ] Stratified 5-fold CV setup
  - [ ] Logistic Regression baseline
  - [ ] Random Forest baseline
  - [ ] LightGBM + Optuna (50 trials)
  - [ ] XGBoost + Optuna (50 trials)
  - [ ] CatBoost + Optuna
  - [ ] Pseudo-labeling round
  - [ ] Probability calibration
  - [ ] Confusion matrices
  - [ ] SHAP analysis + feature importance
  - [ ] Model comparison table
  - [ ] Analysis markdown cells

## Phase 4 — Submission
- [ ] Create `04_submission.ipynb`
  - [ ] Stacking meta-learner (LR on OOF probabilities)
  - [ ] Test-time ROS (10 masked copies, average)
  - [ ] Isotonic calibration on stacked output
  - [ ] Generate submission.csv
  - [ ] Validate format against SampleSubmission.csv
  - [ ] All assertions pass
