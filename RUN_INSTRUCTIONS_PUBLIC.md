# 🚀 Wind Turbine Maintenance — Public Run Instructions

This guide explains how to:
1) Set up a Python environment and install dependencies  
2) Run the **training script** (supervised + anomaly + exports)  
3) Build the **executive one‑pager report** (HTML/PDF)  
4) Generate **SMS alert recipients** from the supervised alerts

> These are **portable** instructions for any machine. Replace the example paths with your own project locations.

---

## 0) Project Layout (recommended)

```
Final Project/
├── wind_turbine_maintenance_data.csv
├── train_wind_maintenance_models_balanced_anomaly_final.py
├── make_final_onepager.py
├── generate_sms_alerts.py
├── outputs_balanced/                 # (created by scripts)
├── outputs_balanced_anomaly/         # (created by scripts)
└── Final_Project_Report/             # (created by one-pager builder)
```

> Make sure your dataset includes a `Maintenance_Label` column (0/1 or numeric convertible to binary).  
> If available, `Turbine_ID` will be included in alert exports and SMS messages.

---

## 1) Create & Activate a Virtual Environment

### macOS / Linux
```bash
cd "/path/to/Final Project"

python3.11 -m venv venv
source venv/bin/activate
```

### Windows (PowerShell)
```powershell
cd "C:\path	o\Final Project"

py -3.11 -m venv venv
.env\Scripts\Activate
```

### Install dependencies
```bash
pip install -U pip setuptools wheel
pip install pandas numpy scikit-learn matplotlib xgboost lightgbm imbalanced-learn reportlab
```
> If `xgboost` or `lightgbm` fails to install, the training script will **gracefully fall back** to sklearn’s GradientBoosting.

---

## 2) Train Models (Supervised + Anomaly) & Export Results

Run the final script:
```bash
python "train_wind_maintenance_models_balanced_anomaly_final.py"   --data "wind_turbine_maintenance_data.csv"   --out  "outputs_balanced_anomaly"   --supervised_alert_policy min_precision --min_precision 0.5
```

**What it produces (in `outputs_balanced_anomaly/`):**
- **Supervised results**
  - `results_balanced_thresholds.csv`
  - `threshold_sweep_rf.csv`, `threshold_sweep_gbdt.csv`
  - `top_thresholds_supervised.csv`
  - **Alerts**: `supervised_alerts_RF.csv`, `supervised_alerts_GBDT.csv`, `supervised_alerts_union.csv`
  - Plots: `roc_both.png`, `pr_both.png`, `pr_vs_threshold_*.png`, `feature_importance_*.png`
- **Anomaly results**
  - `anomaly_results.csv`, `top_anomalies_*.csv`, `anomaly_scores_test.csv`
  - `top_thresholds_anomaly.csv`
  - Plots: `anomaly_roc_*.png`, `anomaly_pr_*.png`, `anomaly_pr_vs_threshold_*.png`, `anomaly_hist_*.png`

**Threshold policy options:**
- `--supervised_alert_policy best_f1` (default) — pick the threshold that maximizes F1
- `--supervised_alert_policy min_precision --min_precision 0.5` — pick the highest‑recall threshold with precision ≥ 0.5

> Tip: Choose **min_precision** to control false alarms; choose **best_f1** for a balanced policy.

---

## 3) Build the Executive One‑Pager (HTML/PDF)

```bash
python "make_final_onepager.py"   --supervised_dir "outputs_balanced"   --anomaly_dir    "outputs_balanced_anomaly"   --out            "Final_Project_Report"
```

**Outputs:**
- `Final_Project_Report/final_onepager.html` (always)
- `Final_Project_Report/final_onepager.pdf` (if `reportlab` is installed)

Captions are **decision‑oriented** (how to act on each figure/table).

---

## 4) Generate SMS Alert Recipients (from supervised alerts)

```bash
python "generate_sms_alerts.py"   --alerts "outputs_balanced_anomaly/supervised_alerts_union.csv"   --out    "outputs_balanced_anomaly/sms_alert_recipients.csv"   --site_name "Your Wind Farm Name"
```

**Output file:** `outputs_balanced_anomaly/sms_alert_recipients.csv` with columns:
- `Turbine_ID`
- `Message` (pre‑formatted alert text)

> The SMS utility requires `Turbine_ID` in the union file. If your dataset lacks it, adapt the script to use `row_index` or join with an asset registry.


---

## 5) Troubleshooting

- **File not found (paths with spaces):** Wrap paths in quotes `"..."` on all OSes.
- **ImportError (`imbalanced-learn`):** `pip install imbalanced-learn`
- **PDF not created:** `pip install reportlab`
- **xgboost/lightgbm issues:** The pipeline still runs—sklearn fallback is automatic.
- **`np` UnboundLocalError:** Ensure `import numpy as np` is present and never reassign `np` as a variable.
- **No `Turbine_ID`:** The SMS script expects it. Add an identifier column or modify the script to use indices.

---

## 6) Interpreting Results (quick guide)

- **PR vs Threshold:** Select cutoffs aligned to your **precision target** (false‑alarm tolerance) and **recall target** (risk tolerance).
- **Top thresholds tables:** Pre‑computed operating points (best‑F1; precision‑constrained ≥ 0.3/0.5).
- **Top anomalies tables:** Start inspections with highest scores; verify with SCADA trends and technician feedback.
- **Feature importances:** Highest‑ranked sensors deserve monitoring focus and data quality checks.

---

## 7) Reproducibility & Versioning

- Pin versions in `requirements.txt` for long‑term reproducibility.
- Save trained model artifacts if you plan to deploy (e.g., `joblib.dump(model, ...)`).
- Keep a changelog of threshold policies used in production.

---

**That’s it!** You now have a portable workflow to train, report, and trigger SMS‑ready alerts from wind turbine data.
