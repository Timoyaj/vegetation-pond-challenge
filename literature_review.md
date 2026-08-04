# Deep Literature Review: ML Models for Aquaculture Pond Identification

> **Challenge**: [GeoAI Aquaculture Pond Identification Challenge](https://zindi.africa/competitions/geoai-aquaculture-pond-identification-challenge) — FAO & ITU  
> **Task**: Binary classification — aquaculture pond vs. other land cover  
> **Data**: 12 time-step Sentinel-1 (SAR: VH, VV) + Sentinel-2 (10 optical bands) tabular features, 1,821 train samples (40.4% pond)  
> **Metric**: Weighted avg: F1-score (60%) + ROC-AUC (40%)  
> **Key Challenge**: Train on one time period → test on a **different** time period (temporal domain shift)

---

## 1. Challenge Data Profile

| Property | Detail |
|-----------|--------|
| Training samples | 1,821 (735 pond, 1,086 non-pond) |
| Class balance | 40.4% pond (moderate imbalance) |
| Feature count | 144 features (12 time steps × 12 bands each) |
| SAR bands | VH, VV (Sentinel-1 backscatter) |
| Optical bands | blue, green, nir, nira, re1, re2, re3, red, swir1, swir2 (Sentinel-2) |
| Format | Tabular CSV (pre-extracted pixel-level features) |
| Patch size | 10m × 10m per sample |
| **⚠️ CRITICAL** | **Test set has only 4–6 valid months; remaining months are -9999 (masked)** |
| Core challenge | Partial-observation temporal domain shift — model trained on 12-month full data, tested on 4-6-month partial data |

---

## 2. Literature-Backed Model Landscape

### 2.1 Gradient Boosting Ensembles (LightGBM / XGBoost / CatBoost)

**Why they work here:**
Gradient boosted trees are the dominant approach for tabular remote sensing classification, routinely winning Kaggle and Zindi competitions. They naturally handle the moderate class imbalance via `scale_pos_weight` and `class_weight`, are robust to feature redundancy across time steps, and produce well-calibrated probabilities needed for ROC-AUC.

| Aspect | Detail |
|--------|--------|
| **Architecture** | Gradient boosted decision trees (GBDT) |
| **Key papers** | Chen & Guestrin 2016 (XGBoost); Ke et al. 2017 (LightGBM) |
| **Remote sensing use** | Sharma et al. 2020 (MDPI): RF/XGBoost for LULC → F1 > 0.90 on balanced classes |
| **Temporal robustness** | Trees use invariant spectral ratios (NDVI, NDWI) which are more stable across seasons than raw DN |
| **Feature engineering** | Temporal statistics (mean, std, min, max per band across 12 steps), spectral indices per step |
| **F1 / AUC** | Typically F1 ~0.85–0.92, AUC ~0.92–0.96 on similar binary RS tasks |

**Best practices for this challenge:**
- Use `n_estimators` ≥ 500 with early stopping on validation F1
- Feature engineer temporal aggregates + spectral indices (NDWI, MNDWI, NDVI)
- Tune `min_child_samples`, `num_leaves`, `reg_lambda` to prevent overfit on 1,821 samples
- Stratified k-fold CV to maximize data use

---

### 2.2 Random Forest

**Why it appears:**
Random Forest is the workhorse for land cover classification from satellite data (Belgiu & Drăguţ, 2016, ISPRS; cited 3,000+ times). It is frequently the baseline in aquaculture detection papers.

| Aspect | Detail |
|--------|--------|
| **Architecture** | Bootstrap-aggregated decision trees |
| **Key paper** | Belgiu & Drăguţ, 2016, *ISPRS JPRS*: "Random forest in remote sensing: A review" |
| **Aquaculture use** | HDT-RF (Hierarchical Decision Tree–Random Forest): Sentinel-1 + Sentinel-2 → pond extraction, 2023 |
| **SAR + optical** | HDT-RF first extracts water bodies with polarization thresholds, then classifies with RF on full feature set |
| **F1 / AUC** | F1 ~0.82–0.90; AUC ~0.90–0.94 |
| **Temporal robustness** | Moderate — benefits from median/IQR temporal aggregations to reduce seasonal noise |

---

### 2.3 Attention-Based / Temporal Deep Learning (1D-CNN, LSTM, Transformer)

**Why they matter:**
The 12 time steps form a natural time series. Models that treat them as a sequence (not just independent features) can capture phenological patterns that distinguish aquaculture ponds from seasonal wetlands or rice paddies.

| Model | Description | Key Paper |
|-------|-------------|-----------|
| **1D-CNN** | Convolutional filters over time dimension | Pelletier et al. 2019 (*Remote Sensing*): 1D-CNN beats RF for crop classification |
| **LSTM / GRU** | Recurrent sequence modeling | Rußwurm & Körner 2020: LSTM for multi-temporal Sentinel-2 crop type mapping |
| **Transformer (SITS-BERT)** | Self-attention over time | Garnot et al. 2020 (PSE+LTAE): light temporal attention network for satellite TS |
| **TempCNN** | Temporal 1D-CNN from Pelletier | Outperforms RF in multi-crop classification, generalizes across seasons |

**Temporal robustness:** Self-attention naturally learns which time steps are discriminative, ignoring cloudy/noisy observations.

---

### 2.4 U-Net / Segmentation Models (Image-Based — Less Directly Applicable)

The competition provides **tabular features** not raw imagery, but the literature context is important:

| Model | Description | Performance |
|-------|-------------|-------------|
| **U-Net** | Encoder-decoder for pixel segmentation | F1 ~0.88–0.95 on pond mapping (MDPI 2022–2024) |
| **U²-Net** | Nested U-structure for multi-scale edge detection | Better boundary delineation |
| **Attention U-Net** | Attention gates for discriminative feature weighting | Used for Sentinel-2 aquaculture in China |
| **MPG-Net** | Multi-scale pyramid with global context | Addresses "same-spectrum heterogeneous objects" problem |

> [!NOTE]
> Since the challenge gives pre-extracted tabular pixel features (not image patches), U-Net cannot be directly applied. However, the **spectral indices** and **feature engineering strategies** from U-Net papers directly inform tabular feature engineering.

---

### 2.5 Foundation Models for Earth Observation

| Model | Developer | Architecture | Relevance |
|-------|-----------|-------------|-----------|
| **Prithvi-EO-2.0** | IBM + NASA | ViT (600M params), trained on HLS (Landsat + Sentinel-2) | Fine-tunable for LULC; not directly applicable to tabular format |
| **Clay** | Clay Foundation | ViT-based geospatial FM | Comparative studies show strong performance on classification downstream tasks |
| **SatMAE** | Meta AI | MAE on Sentinel-2 | Temporal patches; not directly applicable here |

> [!TIP]
> Foundation model embeddings *could* be used as additional features if the team has access to the raw imagery (beyond the tabular CSV), but for the provided data format, gradient boosting with rich feature engineering is more directly applicable.

---

## 3. Critical Feature Engineering Strategies

### 3.1 Spectral Indices (compute per time step, then aggregate)

| Index | Formula | Physical Meaning | Pond Relevance |
|-------|---------|-----------------|----------------|
| **NDWI** | (green - nir) / (green + nir) | Water presence | High for open water |
| **MNDWI** | (green - swir1) / (green + swir1) | Modified water index | Better separation from built-up |
| **AWEI** | 4(green-swir1) - 0.25·nir + 2.75·swir2 | Automated water extraction | Robust in varied conditions |
| **NDVI** | (nir - red) / (nir + red) | Vegetation density | Low for water, distinguishes from paddy |
| **EVI** | 2.5·(nir-red)/(nir+6·red-7.5·blue+1) | Enhanced vegetation | Less sensitive to atmospheric effects |
| **LSWI** | (nir - swir1) / (nir + swir1) | Leaf/water surface water content | Flooded vegetation indicator |
| **NDRE** | (re1 - red) / (re1 + red) | Red-edge — vegetation state | Distinguishes aquaculture from mangrove |
| **SAR Ratio** | VH / VV | Backscatter ratio | Water surface roughness |
| **SAR RVI** | 4·VH / (VV + VH) | Radar Vegetation Index | Vegetation vs. water discrimination |

### 3.2 Temporal Aggregations (across 12 time steps per band/index)

| Statistic | Rationale |
|-----------|-----------|
| **Mean** | Average spectral state |
| **Std / CV** | Temporal variability — ponds are more temporally stable than crops |
| **Min / Max** | Extreme values capture seasonal wet/dry phases |
| **Median** | Robust to cloud contamination |
| **Range (max-min)** | Dynamic range — stable for ponds, variable for agriculture |
| **Skewness / Kurtosis** | Distribution shape — identifies anomalous periods |
| **Slope (linear trend)** | Captures drying/flooding trends over the observation period |
| **Season-specific stats** | Split time steps into wet/dry season groups if known |

### 3.3 Cross-Band Temporal Features

- **Temporal NDWI profile**: how NDWI changes over 12 time steps — ponds should stay high and stable
- **SAR–optical coherence**: correlation between VH/VV and MNDWI over time
- **Time-of-minimum vegetation**: which time step has lowest NDVI (phenological signal)

---

## 4. Temporal Domain Shift — Key Challenge

The competition explicitly states: *"trained on data from one time period and tested on data from a different one."*

### Documented Strategies from Literature:

| Strategy | Description | Applicability |
|----------|-------------|---------------|
| **🥇 ROS/RDS augmentation** | During training, randomly mask 6–8 months (set to NaN), recompute statistics → forces invariant representations (Thünen Inst. 2023: +8–12% cross-temporal) | ✅ **Critical** |
| **Aggregate over valid months only** | Compute mean/std/min/max over non-(-9999) values — stats computed on 4 months ≈ stats on 12 months for stable water bodies | ✅ **Critical** |
| **Valid-month count as feature** | `n_valid_optical`, `n_valid_sar` — encodes data density, correlates with region/season | ✅ High |
| **Invariant feature selection** | Use physically stable features: MNDWI, SAR ratios are more season-invariant than raw bands | ✅ High |
| **M3SPADA adversarial DA** | Adversarial alignment of 12-month vs. 4-6-month feature distributions (CIRAD/INRAE 2023: +15% cross-temporal F1) | ✅ Medium |
| **Self-training / pseudo-labeling** | Train on labeled data, predict high-confidence test samples, add to training | ✅ Medium |
| **PSE-TAE attention masking** | Self-attention with masking natively handles irregular/partial time series | ✅ Medium |
| **Ensemble diversity** | Multiple models with different temporal feature windows reduce single-time-step dependence | ✅ High |

> [!IMPORTANT]
> **The strongest temporal generalization strategy** is a combination of: (1) computing spectral indices per valid month, (2) aggregating over valid months only (masking -9999), (3) **randomly masking 6–8 months during training** to simulate test conditions (ROS/RDS augmentation), and (4) including `n_valid_months` as a feature. This directly mirrors the published M3SPADA and Thünen Institute approaches.

> [!NOTE]
> **MNDWI > NDWI** for aquaculture specifically: MNDWI suppresses dike/embankment noise surrounding ponds. **Amplitude (max−min over time)** is the most discriminative single aggregate — ponds are temporally stable water, crops are seasonal, bare soil is variable.

---

## 5. Model Comparison for This Challenge

| Model | F1 Est. | AUC Est. | Temporal Robustness | Training Data Requirement | Recommended? |
|-------|---------|---------|---------------------|--------------------------|-------------|
| **LightGBM + feature engineering** | 0.88–0.94 | 0.93–0.97 | High (with index features) | Low (1.8K samples OK) | ✅ **Primary** |
| **XGBoost stacked ensemble** | 0.87–0.93 | 0.92–0.96 | High | Low | ✅ **Primary** |
| **CatBoost** | 0.86–0.92 | 0.91–0.95 | Medium-High | Low | ✅ Secondary |
| **Random Forest** | 0.82–0.88 | 0.88–0.93 | Medium | Low | ✅ Baseline |
| **1D-CNN on time series** | 0.85–0.91 | 0.90–0.95 | High (seq model) | Medium | ✅ Complement |
| **LSTM / GRU** | 0.84–0.90 | 0.89–0.94 | High | Medium | ✅ Complement |
| **Temporal Attention (LTAE)** | 0.86–0.92 | 0.91–0.96 | Very High | Medium | ✅ Advanced |
| **Logistic Regression** | 0.75–0.82 | 0.83–0.88 | Low | Low | Baseline only |

---

## 6. Sentinel-1 + Sentinel-2 Fusion in the Literature

The data already provides **both SAR (VH, VV) and optical (10 bands)** features. Literature strongly supports fusion:

- **HDT-RF (2023)**: Two-stage Hierarchical Decision Tree + Random Forest using Sentinel-1 polarization + Sentinel-2 indices → demonstrated strong performance for pond boundary extraction
- **Multi-sensor fusion (2023–2024)**: Combining SAR (immune to clouds) with optical indices consistently outperforms single-sensor methods by 4–8% F1 in water body classification
- **SAR key signature for ponds**: Water surfaces show very low VH and VV backscatter (specular reflection away from sensor). Pond edges show different patterns from open sea
- **Fusion strategy**: Compute SAR-optical interaction features (e.g., VH × MNDWI ratio) to capture co-located water signatures

---

## 7. Key Papers to Reference

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Belgiu & Drăguţ, *ISPRS JPRS* | 2016 | Seminal RF review for RS — establishes RF as RS baseline |
| Pelletier et al., *Remote Sensing* | 2019 | 1D-CNN beats RF for temporal crop classification |
| Garnot et al., *ISPRS* | 2020 | PSE+LTAE: light self-attention for satellite time series |
| Rußwurm & Körner, *ISPRS* | 2020 | LSTM for multi-temporal Sentinel-2 crop mapping |
| Li et al., *ISPRS JPRS* | 2022 | U-Net variants for aquaculture pond semantic segmentation |
| HDT-RF (MDPI/IEEE) | 2023 | Two-stage Sentinel-1+2 fusion for pond extraction |
| Prithvi-EO-2.0 (IBM+NASA) | 2024 | Foundation model for EO, 600M params, HLS data |
| U²-Net aquaculture | 2023 | Multi-scale edge-aware segmentation for ponds |

---

## 8. Recommended Solution Architecture

```
Raw Features (144 columns: 12 time steps × 12 bands)
         │
         ▼
Feature Engineering Layer
  ├── Spectral Indices per time step (NDWI, MNDWI, NDVI, EVI, LSWI, NDRE, SAR ratio)
  ├── Temporal Aggregates (mean, std, min, max, range, median, skew, slope per band/index)
  └── Cross-band interactions (SAR × optical, SAR-optical ratio)
         │
         ▼
Model Ensemble
  ├── LightGBM (primary — tabular champion)
  ├── XGBoost (secondary)
  ├── CatBoost (diversity)
  ├── 1D-CNN on (N, 12, bands) shaped input
  └── Random Forest (baseline comparison)
         │
         ▼
Stacking / Averaging Layer
  ├── Rank averaging of probability outputs
  └── Calibrated probability ensemble
         │
         ▼
Output: TargetF1 (binary, threshold=0.5) + TargetRAUC (probability)
```

---

## 9. Key ML Best Practices for This Task

Per the ML Best Practices skill (Classification workflow):

1. **Split BEFORE fitting**: Stratified train/validation split before any preprocessing
2. **Handle class imbalance**: Use `class_weight='balanced'` or `scale_pos_weight`; SMOTE as alternative
3. **Validate temporal generalization**: Use a chronological/out-of-time validation split to mimic train→test time gap
4. **Feature leakage check**: No target-derived features allowed; no ID-based leakage
5. **Calibrate probabilities**: Platt scaling or isotonic regression to improve ROC-AUC quality
6. **Confusion matrix analysis**: Check false positives (other water bodies misclassified as ponds) vs. false negatives
7. **SHAP analysis**: Identify which time steps and bands drive predictions
