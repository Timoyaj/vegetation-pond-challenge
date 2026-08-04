# Teamwork Project Prompt — Draft

> Status: **Step 9 — Ready for launch — awaiting user approval**  
> Goal: Craft prompt → get user approval → delegate to teamwork_preview

Build a complete, competition-grade ML solution for the **Zindi GeoAI Aquaculture Pond Identification Challenge** (FAO & ITU). The system must perform binary classification — aquaculture pond vs. other land cover — using tabular features extracted from multi-temporal Sentinel-1 (SAR) and Sentinel-2 (optical) satellite imagery, and must generalize across time periods (trained on one date range, tested on another).

Working directory: `c:/Users/USER/Downloads/vegetation-pond-challenge`

Integrity mode: **development**

Reference: https://zindi.africa/competitions/geoai-aquaculture-pond-identification-challenge

---

## Context

The provided data consists of:
- **Train.csv** (1,821 samples): 144 features — 12 time steps × 12 per-step bands (VH, VV from Sentinel-1 SAR; blue, green, nir, nira, re1, re2, re3, red, swir1, swir2 from Sentinel-2) + ID + label (binary: 1=pond, 0=other)
- **Test.csv**: same feature structure, no label column; **critically, only 4–6 consecutive months contain real values — the remaining months are masked with -9999.** This simulates real-world partial-observation scenarios.
- **SampleSubmission.csv**: expected format with columns: ID, TargetF1, TargetRAUC
- **challenge data disc.pdf**: additional data description in the working directory

Class distribution: ~40.4% pond, ~59.6% non-pond (moderate imbalance).

The core challenge is a **partial-observation temporal domain shift**: the training data has all 12 months populated, while the test data has only 4–6 valid months (the rest are -9999). Any model that uses month-specific raw features directly will fail; the solution must compute features that are meaningful and stable when computed over any subset of months.

---

## Requirements

### R1. Literature-Informed Feature Engineering
Engineer features from the raw 144-column tabular data that are **robust to partial observations** (i.e., work correctly when only 4–6 of the 12 months are valid). This must include:
- **Preprocessing**: Replace -9999 with NaN; track `n_valid_optical` and `n_valid_sar` per sample as features
- **Spectral water/vegetation indices** computed per valid time step: NDWI = (green−nir)/(green+nir), MNDWI = (green−swir1)/(green+swir1) (preferred over NDWI for pond edges), NDVI, EVI, LSWI, NDRE, SAR ratio (VH−VV), SAR-RVI
- **Temporal aggregation statistics** computed over **valid months only** (ignoring NaN): mean, std, min, max, amplitude (max−min), p25, p75, skewness, linear trend slope per raw band and per derived index
- Justify each feature family with its physical meaning for aquaculture pond discrimination

### R2. Temporal Augmentation + Multi-Model Ensemble
The critical strategy for handling the partial-observation test set is **Random Observation Selection (ROS)**: during training, randomly mask 6–8 of the 12 months (set to NaN) for each sample, recompute the aggregate features, and train on this augmented version. This forces the model to learn from any subset of months (matching Thünen Institute 2023, which showed +8–12% cross-temporal generalization improvement).

Train and compare at least three model types:
- At least one gradient boosting model (LightGBM, XGBoost, or CatBoost), with `scale_pos_weight` for class imbalance
- At least one model that explicitly exploits the time-series structure (1D-CNN over the 12 time steps with NaN masking, or LSTM/GRU, or PSE-TAE temporal attention)
- A baseline (Random Forest or Logistic Regression)

Use stratified k-fold cross-validation. Apply calibration (Platt scaling or isotonic regression) to ensure well-calibrated probability outputs.

### R3. Rigorous Evaluation and Model Comparison
Follow the ML Best Practices classification workflow:
- Always split data before fitting preprocessing pipelines
- Handle missing/null values explicitly (analyze, decide, document)
- Check for and address class imbalance (e.g., class weights, oversampling)
- Evaluate each model with: F1-score, ROC-AUC, precision-recall curves, and confusion matrices
- Final composite score: weighted avg of F1 (60%) and AUC (40%), matching competition metric
- Report SHAP feature importance for the best model

### R4. Competition-Format Submission File
Generate a valid `submission.csv` with three columns: `ID`, `TargetF1` (binary 0/1 using threshold 0.5 — do NOT tune threshold), `TargetRAUC` (raw probability between 0 and 1). The file must match the format of `SampleSubmission.csv`.

---

## Acceptance Criteria

### Feature Engineering Quality
- [ ] -9999 values replaced with NaN and valid-month counts (`n_valid_optical`, `n_valid_sar`) computed as features
- [ ] At least 6 distinct spectral indices computed from the raw bands (NDWI, MNDWI, NDVI, EVI, SAR ratio, plus one more)
- [ ] Temporal aggregation features computed over **valid months only** (NaN-aware: mean, std, min, max, amplitude per band)
- [ ] Feature engineering pipeline is fitted on training data only (no leakage from test set)
- [ ] ROS temporal augmentation applied during training (randomly mask 6–8 months, recompute features, augment training set 3× or more)

### Model Performance (Cross-Validation)
- [ ] Best model achieves composite score ≥ 0.80 on stratified out-of-fold CV (weighted avg: F1×0.6 + AUC×0.4)
- [ ] At least 3 models trained and compared with identical CV folds
- [ ] Probability calibration applied to the final ensemble

### Evaluation Rigor
- [ ] Missing value analysis completed and documented
- [ ] Confusion matrix plotted for best model
- [ ] SHAP or feature importance analysis completed and visualized
- [ ] All preprocessing fitted on training fold, applied to validation fold (no leakage)

### Submission
- [ ] `submission.csv` generated with correct format: columns ID, TargetF1, TargetRAUC
- [ ] TargetF1 contains only 0 or 1 (binary, threshold=0.5 on raw probability)
- [ ] TargetRAUC contains raw probabilities in [0, 1] range
- [ ] Row count matches `SampleSubmission.csv`
- [ ] No NaN values in submission

### Documentation
- [ ] Each notebook section followed by a markdown analysis cell explaining the results
- [ ] Final summary cell comprehensively answers: which model works best and why it generalizes temporally

---
*Next: when approved → delegate via invoke_subagent (see Delegation Protocol)*
