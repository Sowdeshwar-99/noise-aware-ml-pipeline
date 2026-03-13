# Noise-Aware Machine Learning Pipeline for Large-Scale Agricultural Yield Prediction

This project investigates how data noise affects machine learning performance in agricultural yield prediction and demonstrates how a **noise-aware ML pipeline** can restore model accuracy and stability.

The pipeline simulates realistic feature and label noise in agronomic datasets and applies statistical mitigation techniques to recover predictive performance at scale using **PySpark and distributed LightGBM**.

The system processes **19 million agricultural records** and demonstrates that noise-aware preprocessing significantly improves model reliability in large-scale precision agriculture analytics.

---

# Project Workflow

![Pipeline Workflow](images/pipeline_workflow.png)

The framework uses a **two-phase pipeline**:

### Phase 1 — Sample Pipeline (100k rows)
Used for controlled experimentation and model selection.

Steps:
1. Data ingestion
2. Preprocessing and feature engineering
3. Noise simulation (50%)
4. Noise mitigation
5. Model training and evaluation

Four regression models were tested:

- Random Forest
- XGBoost
- LightGBM
- Bayesian Ridge Regression

LightGBM demonstrated the best balance of **accuracy, robustness, and scalability**.

---

### Phase 2 — Distributed Pipeline (19M rows)

The selected model was scaled to the full dataset using:

- **PySpark 3.5**
- **SynapseML LightGBM**

Noise was simulated at:

- 20%
- 30%
- 50%

The pipeline applied dual-stage mitigation before distributed training.

---

# Feature Correlation Analysis

![Feature Correlation](images/feature_correlation.png)

The most influential predictors for crop yield include:

- Biomass
- Cumulative thermal time (cumTT)
- Zadok growth stage
- Day of year (DOY)
- Leaf weight

These agronomic variables capture seasonal growth patterns and plant physiological development.

---

# Noise Simulation

Realistic noise was injected into the dataset to simulate real-world agricultural data issues.

Feature noise included:

- Gaussian additive noise
- Multiplicative jitter
- Logit-space perturbations
- Random missingness (MCAR)

Label noise included:

- Yield zeroing
- Outlier scaling
- Log-normal perturbations

This allowed evaluation of model robustness under controlled corruption scenarios.

---

# Noise Mitigation Framework

The pipeline applies **two-stage noise mitigation**:

### Feature Mitigation
- Bayesian Ridge Iterative Imputation
- PCA-based denoising

### Label Mitigation
- Random Forest residual analysis
- Median Absolute Deviation (MAD) filtering

This process reconstructs corrupted signals while preserving real agronomic variability.

---

# Model Performance

![RMSE Comparison](images/rmse_comparison.png)

Noise severely degraded model performance:

| Condition | RMSE |
|----------|------|
| Clean | 412 |
| 50% Noise | 1604 |
| Mitigated | 513 |

The mitigation pipeline recovered **over 90% of lost predictive accuracy**.

---

# Yield Variability Recovery

![Yield Variability](images/yield_variability.png)

Noise increased yield variance dramatically.

Mitigation reduced the variance from:

**≈2010 → 1258 kg/ha**

bringing it close to the clean data baseline.

---

# Scalability Performance

![Runtime Scaling](images/runtime_scaling.png)

The distributed Spark pipeline scaled efficiently:

| Dataset Size | Runtime |
|--------------|--------|
| 100k rows | ~9 minutes |
| 19M rows | ~64 minutes |

This represents **near-linear scalability** for large agricultural datasets.

---

# Technology Stack

- Python
- PySpark 3.5
- SynapseML
- LightGBM
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

# Key Contributions

- Large-scale **noise simulation framework for agricultural datasets**
- Dual-stage **feature and label denoising pipeline**
- Distributed **Spark + LightGBM ML architecture**
- Demonstrated **model robustness recovery under 50% noise**
- Scalable processing for **19M+ records**

---

# Author

Sowdeshwar Survesha Kumaar  
Master of Data Science  
University of Queensland
