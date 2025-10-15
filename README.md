# Wind Turbine Maintenance — ML + Anomaly Detection Project

This project predicts **maintenance needs** and detects **unusual sensor behavior** in wind turbines using both **supervised** and **unsupervised** machine learning.

---

## 📘 Overview

The project explores two complementary modeling paths:

### Supervised (Classification)
Predict whether a turbine requires maintenance (`Maintenance_Label = 1`)  
using operational sensor data.

Models used:
- **Random Forest (RF)**
- **Gradient Boosted Trees (GBDT)** — XGBoost → LightGBM → sklearn fallback  
- **SMOTE** balancing to handle class imbalance  
- Threshold tuning (0.05–0.95) to optimize F1-score

### Unsupervised (Anomaly Detection)
Detect turbines showing **abnormal patterns** even without labeled failures.

Models used:
- **IsolationForest** — isolates outliers by random partitioning
- **One-Class SVM** — defines a boundary around normal behavior
- Evaluated via **ROC-AUC** and **PR-AUC** using known maintenance labels (if available)

---

## Goals

1. Identify key sensor patterns leading to maintenance.
2. Detect early anomalies indicating potential failure.
3. Generate explainable and reproducible visual insights.
4. Produce a clean **one-page summary report (HTML/PDF)**.

---

## Project Structure

```
Final Project/
├── wind_turbine_maintenance_data.csv           # Main dataset
│
├── train_wind_maintenance_models_balanced.py   # Supervised models (SMOTE + threshold tuning)
├── train_wind_maintenance_models_balanced_anomaly.py
│                                               # Adds anomaly detection (IsolationForest + OneClassSVM)
│
├── wind_maintenance_balanced_thresholds.ipynb  # Notebook for SMOTE + threshold analysis
├── wind_maintenance_balanced_anomaly.ipynb     # Notebook incl. anomaly detection
├── wind_maintenance_balanced_anomaly_explained.ipynb
│                                               # Fully explained version (step-by-step)
│
├── make_final_onepager.py                      # Builds one-page HTML + PDF report with captions
├── build_final_onepager.ipynb                  # Notebook interface for the report builder
│
├── outputs_balanced/                           # Results of supervised models
├── outputs_balanced_anomaly/                   # Anomaly detection outputs
└── Final_Project_Report/                       # Final one-pager (HTML/PDF + assets)
```

---

The report (`final_onepager.html` + `final_onepager.pdf`) includes:
- Tables of model metrics  
- ROC/PR/F1 plots with captions  
- Feature importance graphs  
- Top 20 anomalies with Turbine IDs and scores

---

## Key Insights

- **Class imbalance** was addressed with SMOTE and threshold tuning.  
- Both models achieved high recall but modest precision → useful for *early warnings*.  
- **Anomaly detection** helps flag subtle deviations even without labels.  
- **Feature importances** show which sensors (e.g., vibration, temperature) drive maintenance predictions.

---

## Next Steps

- Add rolling statistics and temporal features.
- Tune anomaly detectors with domain thresholds.
- Integrate explainability tools (e.g., SHAP) for per-prediction insights.
- Deploy models in a monitoring dashboard (Streamlit / Power BI).

---

## Author

**Roberto David Palacios Guerra**  
_MCSBT — Intro to AI and ML (Final Project)_  
© 2025 — All rights reserved.
