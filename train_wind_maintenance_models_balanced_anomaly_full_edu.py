#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wind Turbine Maintenance — FULL Pipeline (Supervised + Anomaly)
====================================================================================

What this script does (end-to-end)
----------------------------------
1) Loads a CSV containing numeric sensor features and a target column `Maintenance_Label`.
2) Splits data into train/test and balances the training set with SMOTE (to handle class imbalance).
3) Trains TWO supervised models:
      - RandomForestClassifier (robust baseline, feature importances)
      - Gradient-Boosted Trees (XGBoost → LightGBM → sklearn fallback)
4) Evaluates the ranking quality using ROC and Precision–Recall (PR) curves.
5) Plots **Precision–Recall vs Threshold** to help you pick an operating point.
6) Scans many thresholds to produce "top picks" tables (best F1; precision-constrained).
7) Exports **supervised alerts** for each model AND a UNION of both.
8) Trains TWO unsupervised anomaly detectors (IsolationForest + One-Class SVM)
   using only NORMAL training data (y==0), then evaluates and exports results.
9) Saves plots, CSVs, and quick HTML/CSV summaries into the output directory.

Minimal usage example (adjust paths as needed)
----------------------------------------------
python train_wind_maintenance_models_balanced_anomaly_full_edu.py \
  --data wind_turbine_maintenance_data.csv \
  --out outputs_balanced_anomaly \
  --supervised_alert_policy min_precision --min_precision 0.5

Key assumptions
---------------
- The dataset has a column named **Maintenance_Label**. Any value > 0 is treated as "needs maintenance" (1).
- All other **numeric** columns are treated as features (strings/timestamps are ignored).
- If a `Turbine_ID` column exists, it will be preserved in alert CSVs for human-friendly outputs.
"""

# ----------------------------
# 1) Imports & basic settings
# ----------------------------
import argparse                      # read --data, --out, etc. from the command line
import warnings                      # show non-fatal warnings when optional libs are missing
from pathlib import Path             # robust cross-platform filesystem paths

import numpy as np                   # fast numeric arrays, math utilities
import pandas as pd                  # tabular data loading and manipulation
import matplotlib.pyplot as plt      # plotting library (kept simple for portability)

from sklearn.model_selection import train_test_split                 # split dataset into train/test
from sklearn.metrics import (                                        # evaluation metrics + plotting helpers
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest  # RF (supervised), IF (unsupervised)
from sklearn.svm import OneClassSVM                                   # OCSVM (unsupervised)

# We PREFER fast external gradient boosters if available, else we fall back to sklearn:
XGB_AVAILABLE = False
LGBM_AVAILABLE = False
SKLEARN_GB_AVAILABLE = True
try:
    from xgboost import XGBClassifier   # best speed/accuracy for many tabular tasks
    XGB_AVAILABLE = True
except Exception:
    warnings.warn("xgboost not available; will try lightgbm or sklearn's GradientBoosting instead.")
if not XGB_AVAILABLE:
    try:
        from lightgbm import LGBMClassifier  # also fast and strong on tabular data
        LGBM_AVAILABLE = True
    except Exception:
        warnings.warn("lightgbm not available; will fall back to sklearn.")
try:
    from sklearn.ensemble import GradientBoostingClassifier  # built-in fallback
except Exception:
    SKLEARN_GB_AVAILABLE = False

# SMOTE = Synthetic Minority Oversampling Technique (balances classes in TRAINING ONLY)
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    raise SystemExit("Missing dependency: imbalanced-learn. Install with: pip install imbalanced-learn")


# ---------------------------------
# 2) Utility / helper functions
# ---------------------------------
def ensure_outdir(outdir: Path) -> None:
    """Create the output folder if it does not exist (so we can save plots/CSVs)."""
    outdir.mkdir(parents=True, exist_ok=True)  # parents=True creates any missing intermediate folders


def load_data(csv_path: Path):
    """
    Load the dataset and prepare:
      df : original DataFrame (we keep this around for Turbine_ID joins later)
      X  : numeric feature matrix (all numeric columns EXCEPT Maintenance_Label)
      y  : binary target (1 if Maintenance_Label > 0 else 0)

    Why numeric only?
    - Most ML models expect numbers. Strings/timestamps should be preprocessed,
      but here we keep things simple and ignore non-numeric columns.
    """
    df = pd.read_csv(csv_path)  # read CSV into a DataFrame
    if "Maintenance_Label" not in df.columns:  # validate expected target column
        raise ValueError("Expected a 'Maintenance_Label' column in the dataset.")

    # Convert labels to numeric; any non-numeric becomes NaN → fill with 0, then map >0 to 1 (binary).
    y_raw = pd.to_numeric(df["Maintenance_Label"], errors="coerce").fillna(0)
    y = (y_raw > 0).astype(int)  # final binary vector

    # Features = all numeric columns EXCEPT the label
    X = df.drop(columns=["Maintenance_Label"]).select_dtypes(include=[np.number])

    # Helpful print for class imbalance awareness
    print("Label value counts (entire dataset):")
    print(pd.Series(y).value_counts(dropna=False).to_string())

    return df, X, y


def make_rf() -> RandomForestClassifier:
    """Build a Random Forest with solid defaults (good baseline; handles noisy features)."""
    return RandomForestClassifier(
        n_estimators=400,  # number of trees (more trees → more stable, slower)
        n_jobs=-1,        # use all CPU cores
        random_state=42   # reproducibility
    )


def make_gbdt():
    """
    Build a Gradient-Boosted Trees classifier.
    Preference order: XGBoost → LightGBM → sklearn GradientBoosting.
    These are conservative defaults; tune later for maximum performance.
    """
    if XGB_AVAILABLE:
        return XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            tree_method="hist",   # fast histogram-based algorithm
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=1.0  # class weighting (we use SMOTE so keep this at 1.0)
        )
    if LGBM_AVAILABLE:
        return LGBMClassifier(
            n_estimators=700,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary",
            random_state=42,
            n_jobs=-1
        )
    if SKLEARN_GB_AVAILABLE:
        return GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=3,
            random_state=42
        )
    raise RuntimeError("No gradient boosting backend available (xgboost, lightgbm, or sklearn).")


def get_proba(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return the POSITIVE-CLASS probability for each row in X.
    - If the model has predict_proba (most classifiers) → use it.
    - Else, if it has decision_function (scores) → min-max normalize to [0,1].
    - Else, fall back to predict() (rare).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]  # column 1 = prob of class "1"
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)  # arbitrary real-valued scores
        d_min, d_max = d.min(), d.max()
        return (d - d_min) / (d_max - d_min + 1e-9)  # scale scores to probabilities-ish
    return model.predict(X)  # last resort (not ideal)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """
    Compute common classification metrics.
    If we have probabilities, also compute ROC-AUC (ranking) and PR-AUC (good under imbalance).
    """
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),  # zero_division avoids crashes on rare cases
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_proba)                 # AUC of ROC curve
        except Exception:
            m["roc_auc"] = np.nan
        try:
            m["pr_auc"] = average_precision_score(y_true, y_proba)        # area under PR curve
        except Exception:
            m["pr_auc"] = np.nan
    return m


def sup_thr_sweep(y_true: np.ndarray, y_prob: np.ndarray, out_csv: Path):
    """
    Supervised threshold sweep (probabilities → labels):
      - Try many thresholds between 0.05 and 0.95.
      - At each threshold, compute metrics and save the table.
      - Return the threshold with the BEST F1 (balanced precision/recall).
    """
    rows = []
    thresholds = np.linspace(0.05, 0.95, 19)  # 0.05, 0.10, ..., 0.95
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        rows.append({
            "threshold": thr,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })
    df_thr = pd.DataFrame(rows)
    df_thr.to_csv(out_csv, index=False)  # save the sweep for transparency/reproducibility
    best_row = df_thr.iloc[df_thr["f1"].values.argmax()]  # row with max F1
    return float(best_row["threshold"]), df_thr


def threshold_sweep_scores(y_true: np.ndarray, scores: np.ndarray, out_csv: Path):
    """
    Unsupervised threshold sweep (scores → labels):
      - For anomaly scores (higher = more anomalous), scan cutoffs across 5th..95th percentiles.
      - Compute metrics and save the table.
      - Return the cutoff with the BEST F1.
    """
    rows = []
    lo, hi = np.percentile(scores, 5), np.percentile(scores, 95)  # avoid extreme tails
    thr_values = np.linspace(lo, hi, 31)  # 31 cutoffs
    for thr in thr_values:
        y_pred = (scores >= thr).astype(int)
        rows.append({
            "threshold": thr,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })
    df_thr = pd.DataFrame(rows)
    df_thr.to_csv(out_csv, index=False)
    best_row = df_thr.iloc[df_thr["f1"].values.argmax()]
    return float(best_row["threshold"]), df_thr


def top_threshold_summary(df_thr: pd.DataFrame, min_precisions=(0.3, 0.5)) -> dict:
    """
    Summarize a threshold sweep with 3 actionable picks:
      - best_f1_threshold        : the overall F1 winner (balanced)
      - prec>=0.3_threshold      : highest recall while keeping precision ≥ 0.30
      - prec>=0.5_threshold      : highest recall while keeping precision ≥ 0.50
    This lets product/ops pick based on risk tolerance.
    """
    out: dict[str, float | None] = {}
    if df_thr is None or df_thr.empty:
        return out

    # Best F1
    i = df_thr["f1"].values.argmax()
    out["best_f1_threshold"] = float(df_thr.iloc[i]["threshold"])
    out["best_f1"] = float(df_thr.iloc[i]["f1"])
    out["best_f1_precision"] = float(df_thr.iloc[i]["precision"])
    out["best_f1_recall"] = float(df_thr.iloc[i]["recall"])

    # Precision-constrained picks
    for pmin in min_precisions:
        df_ok = df_thr[df_thr["precision"] >= pmin]
        if len(df_ok):
            j = df_ok["recall"].values.argmax()
            row = df_ok.iloc[j]
            out[f"prec>={pmin}_threshold"] = float(row["threshold"])
            out[f"prec>={pmin}_recall"] = float(row["recall"])
            out[f"prec>={pmin}_f1"] = float(row["f1"])
        else:
            out[f"prec>={pmin}_threshold"] = None
            out[f"prec>={pmin}_recall"] = None
            out[f"prec>={pmin}_f1"] = None
    return out


def plot_importance(model, feature_names, title: str, out_path: Path, top_k: int = 15):
    """
    Plot top-k feature importances for tree models.
    NOTE: Importances reflect how often/successfully features are used in splits,
    not causal effects. Use domain knowledge for interpretation.
    """
    if not hasattr(model, "feature_importances_"):
        warnings.warn(f"No feature_importances_ for {title}.")
        return None
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top_k]  # indices of top-k features
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(order))[::-1], imp[order][::-1])  # horizontal bars; most important on top
    ax.set_yticks(range(len(order))[::-1])
    ax.set_yticklabels([feature_names[i] for i in order][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)  # save to disk
    plt.close(fig)  # free memory
    return out_path


# ----------------------
# 3) Main program logic
# ----------------------
def main():
    # Parse command-line arguments so the script is flexible and automatable.
    parser = argparse.ArgumentParser(description="Wind maintenance pipeline (supervised + anomaly) — EDU")
    parser.add_argument("--data", required=True, type=str, help="Path to CSV with numeric features + Maintenance_Label")
    parser.add_argument("--out", default="outputs_balanced_anomaly", type=str, help="Folder to write plots/CSVs")
    parser.add_argument("--test_size", default=0.2, type=float, help="Fraction of data for test split (default 0.2)")
    parser.add_argument("--contamination", default=0.05, type=float, help="Approx anomaly fraction for IF/OCSVM")
    parser.add_argument("--supervised_alert_policy", default="best_f1", choices=["best_f1", "min_precision"],
                        help="How to choose alert thresholds from sweeps")
    parser.add_argument("--min_precision", default=0.5, type=float,
                        help="Used only when --supervised_alert_policy=min_precision")
    args = parser.parse_args()

    # Resolve paths and make sure output directory exists
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found at: {data_path}. Tip: wrap spaced paths in quotes.")
    outdir = Path(args.out)
    ensure_outdir(outdir)

    # ---- Load dataset and split into train/test ----
    print("Loading data...")
    df, X, y = load_data(data_path)
    feature_names = X.columns.tolist()

    # Stratified split keeps the same positive/negative ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print("Label counts (train):", np.unique(y_train, return_counts=True))
    print("Label counts (test) :", np.unique(y_test, return_counts=True))

    # ---- Balance TRAINING ONLY with SMOTE ----
    # We NEVER oversample the test set; it's supposed to simulate real data distribution.
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("Resampled label counts:", np.unique(y_res, return_counts=True))

    # ---- Train supervised models on the balanced training set ----
    print("Training RandomForest...")
    rf = make_rf().fit(X_res, y_res)  # fit = learn from data

    print("Training Gradient Boosted Trees...")
    gb = make_gbdt().fit(X_res, y_res)

    # ---- Get probabilities on the untouched test set ----
    print("Evaluating supervised models...")
    y_proba_rf = get_proba(rf, X_test)
    y_proba_gb = get_proba(gb, X_test)

    # ---- Plot ROC and PR curves (ranking quality) ----
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba_rf, name="RandomForest", ax=ax)
    RocCurveDisplay.from_predictions(y_test, y_proba_gb, name="GBDT", ax=ax)
    ax.set_title("ROC — RF vs GBDT")
    fig.tight_layout()
    fig.savefig(outdir / "roc_both.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba_rf, name="RandomForest", ax=ax)
    PrecisionRecallDisplay.from_predictions(y_test, y_proba_gb, name="GBDT", ax=ax)
    ax.set_title("Precision–Recall — RF vs GBDT")
    fig.tight_layout()
    fig.savefig(outdir / "pr_both.png", dpi=150)
    plt.close(fig)

    # ---- Plot Precision & Recall vs Threshold (visual aid for choosing operating point) ----
    for name, probs in [("RandomForest", y_proba_rf), ("GBDT", y_proba_gb)]:
        p_vals, r_vals, thr_vals = precision_recall_curve(y_test, probs)  # returns arrays of p/r & thresholds
        # thresholds array is len-1 vs p/r; pad to align for plotting convenience
        thr_plot = np.concatenate([thr_vals, [thr_vals[-1] if thr_vals.size else 0.5]]) if thr_vals.size else np.array([0.5])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thr_plot, p_vals, label="Precision")  # precision typically decreases as threshold lowers
        ax.plot(thr_plot, r_vals, label="Recall")     # recall typically increases as threshold lowers
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"Precision–Recall vs Threshold — {name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"pr_vs_threshold_{name}.png", dpi=150)
        plt.close(fig)

    # ---- Sweep thresholds and save "top picks" (for business-friendly choices) ----
    best_thr_rf, df_thr_rf = sup_thr_sweep(y_test, y_proba_rf, outdir / "threshold_sweep_rf.csv")
    best_thr_gb, df_thr_gb = sup_thr_sweep(y_test, y_proba_gb, outdir / "threshold_sweep_gbdt.csv")
    top_sup_df = pd.DataFrame([
        {"model": "RandomForest", **top_threshold_summary(df_thr_rf)},
        {"model": "GBDT", **top_threshold_summary(df_thr_gb)},
    ]).set_index("model")
    top_sup_df.to_csv(outdir / "top_thresholds_supervised.csv")

    # ---- Evaluate at default threshold 0.50 and at best-F1 (for a compact snapshot table) ----
    def eval_at(thr: float, proba: np.ndarray) -> dict:
        y_pred = (proba >= thr).astype(int)
        return compute_metrics(y_test, y_pred, proba)

    results_sup = pd.DataFrame([
        {"model": "RandomForest@0.50", **eval_at(0.50, y_proba_rf)},
        {"model": f"RandomForest@{best_thr_rf:.2f}", **eval_at(best_thr_rf, y_proba_rf)},
        {"model": "GBDT@0.50", **eval_at(0.50, y_proba_gb)},
        {"model": f"GBDT@{best_thr_gb:.2f}", **eval_at(best_thr_gb, y_proba_gb)},
    ]).set_index("model").sort_values("f1", ascending=False)
    results_sup.to_csv(outdir / "results_balanced_thresholds.csv")

    # ---- Decide which threshold to use for ALERT exports ----
    def pick_alert_threshold(df_thr: pd.DataFrame, policy: str, pmin: float) -> tuple[float, str]:
        """
        policy == "best_f1":
            pick the highest F1 threshold overall
        policy == "min_precision":
            among thresholds with precision >= pmin, pick the one with highest recall
            if none meet the precision constraint → fall back to best F1
        """
        if policy == "best_f1":
            i = df_thr["f1"].values.argmax()
            return float(df_thr.iloc[i]["threshold"]), "best_f1"
        # precision-constrained
        df_ok = df_thr[df_thr["precision"] >= pmin]
        if len(df_ok):
            j = df_ok["recall"].values.argmax()
            return float(df_ok.iloc[j]["threshold"]), f"min_precision>={pmin}"
        # fallback
        i = df_thr["f1"].values.argmax()
        return float(df_thr.iloc[i]["threshold"]), "fallback_best_f1"

    thr_rf_alert, policy_rf = pick_alert_threshold(df_thr_rf, args.supervised_alert_policy, args.min_precision)
    thr_gb_alert, policy_gb = pick_alert_threshold(df_thr_gb, args.supervised_alert_policy, args.min_precision)

    # ---- Export supervised alerts (per model) ----
    # We attach Turbine_ID if available for operational readability.
    sup_rf = pd.DataFrame(index=X_test.index, data={
        "predicted_prob": y_proba_rf,
        "predicted_label": (y_proba_rf >= thr_rf_alert).astype(int),
        "true_label": y_test.values,
    })
    if "Turbine_ID" in df.columns:
        sup_rf["Turbine_ID"] = df.loc[X_test.index, "Turbine_ID"].values
    rf_alerts = sup_rf[sup_rf["predicted_label"] == 1].copy()  # only rows that triggered
    rf_alerts.to_csv(outdir / "supervised_alerts_RF.csv", index=False)

    sup_gb = pd.DataFrame(index=X_test.index, data={
        "predicted_prob": y_proba_gb,
        "predicted_label": (y_proba_gb >= thr_gb_alert).astype(int),
        "true_label": y_test.values,
    })
    if "Turbine_ID" in df.columns:
        sup_gb["Turbine_ID"] = df.loc[X_test.index, "Turbine_ID"].values
    gb_alerts = sup_gb[sup_gb["predicted_label"] == 1].copy()
    gb_alerts.to_csv(outdir / "supervised_alerts_GBDT.csv", index=False)

    # ---- Export UNION of supervised alerts (triggered by RF OR GBDT) ----
    union = pd.DataFrame({
        "row_index": X_test.index,
        "true_label": y_test.values,
        "prob_rf": y_proba_rf,
        "prob_gbdt": y_proba_gb,
        "rf_trigger": (y_proba_rf >= thr_rf_alert).astype(int),
        "gbdt_trigger": (y_proba_gb >= thr_gb_alert).astype(int),
    })
    if "Turbine_ID" in df.columns:
        union["Turbine_ID"] = df.loc[X_test.index, "Turbine_ID"].values
    union["triggered_by"] = union.apply(
        lambda r: ",".join([m for m, b in [("RF", r["rf_trigger"]), ("GBDT", r["gbdt_trigger"])] if b]), axis=1
    )
    union_alerts = union[(union["rf_trigger"] == 1) | (union["gbdt_trigger"] == 1)].copy()
    union_alerts["rf_threshold_used"] = thr_rf_alert
    union_alerts["gbdt_threshold_used"] = thr_gb_alert
    union_alerts["rf_policy"] = policy_rf
    union_alerts["gbdt_policy"] = policy_gb
    union_alerts.to_csv(outdir / "supervised_alerts_union.csv", index=False)

    print(f"Supervised alerts exported: RF={len(rf_alerts)}, GBDT={len(gb_alerts)}, Union={len(union_alerts)}")

    # ---- Plot feature importances (which features influence decisions most) ----
    _ = plot_importance(rf, feature_names, "Feature Importance — RF (SMOTE)", outdir / "feature_importance_rf_smote.png")
    _ = plot_importance(gb, feature_names, "Feature Importance — GBDT (SMOTE)", outdir / "feature_importance_gbdt_smote.png")

    # ---- Unsupervised anomaly detection: train on NORMAL training data only ----
    print("Training anomaly detectors on normal-only training data...")
    X_train_norm = X_train[y_train == 0]  # keep only y==0 for learning "normal" behavior

    # IsolationForest: points that are "isolated" from the bulk of data become anomalies
    iso = IsolationForest(
        n_estimators=300,
        contamination=args.contamination,  # expected fraction of anomalies (approx)
        random_state=42
    ).fit(X_train_norm)

    # One-Class SVM: learn a boundary around normal data; points outside are anomalies
    oc = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=max(min(args.contamination, 0.49), 0.01)  # nu ∈ (0,1]; rough anomaly fraction bound
    ).fit(X_train_norm)

    # Convert model outputs to "anomaly scores" (higher = more anomalous)
    iso_scores = -iso.score_samples(X_test)     # IF returns higher score for inliers → we flip sign
    oc_scores = -oc.decision_function(X_test)   # OCSVM decision_function: positive=inlier → flip

    # ---- Evaluate anomaly detectors, sweep thresholds, and plot ----
    anom_results = []     # hold metrics at best-F1 cutoff
    anom_top_rows = []    # hold top-pick threshold summaries

    for name, scores in [("IsolationForest", iso_scores), ("OneClassSVM", oc_scores)]:
        # Overall ranking quality (higher is better)
        roc = roc_auc_score(y_test, scores)
        pr = average_precision_score(y_test, scores)

        # Sweep score thresholds to obtain best-F1 and summary table
        thr_csv = outdir / f"anomaly_threshold_sweep_{name}.csv"
        best_thr, df_thr = threshold_sweep_scores(y_test, scores, thr_csv)
        top = top_threshold_summary(df_thr)
        anom_top_rows.append({"model": name, **top})

        # Metrics at the chosen cutoff
        y_pred = (scores >= best_thr).astype(int)
        m = compute_metrics(y_test, y_pred, scores)
        m["roc_auc"] = roc
        m["pr_auc"] = pr
        m["best_thr"] = best_thr
        m["model"] = name
        anom_results.append(m)

        # --- Plots for anomaly models ---
        # ROC curve
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, scores, name=name, ax=ax)
        ax.set_title(f"Anomaly ROC — {name}")
        fig.tight_layout()
        fig.savefig(outdir / f"anomaly_roc_{name}.png", dpi=150)
        plt.close(fig)

        # PR curve
        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_test, scores, name=name, ax=ax)
        ax.set_title(f"Anomaly PR — {name}")
        fig.tight_layout()
        fig.savefig(outdir / f"anomaly_pr_{name}.png", dpi=150)
        plt.close(fig)

        # Precision/Recall vs score threshold
        p_vals, r_vals, thr_vals = precision_recall_curve(y_test, scores)
        thr_plot = np.concatenate([thr_vals, [thr_vals[-1] if thr_vals.size else scores.mean()]]) if thr_vals.size else np.array([scores.mean()])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thr_plot, p_vals, label="Precision")
        ax.plot(thr_plot, r_vals, label="Recall")
        ax.set_xlabel("Score threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"PR vs Threshold — {name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"anomaly_pr_vs_threshold_{name}.png", dpi=150)
        plt.close(fig)

        # Histogram of anomaly scores (helps visualize separation)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(scores, bins=40)
        ax.set_title(f"Anomaly Score Histogram — {name}")
        ax.set_xlabel("Score (higher = more anomalous)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(outdir / f"anomaly_hist_{name}.png", dpi=150)
        plt.close(fig)

        # Export top-50 anomalies for inspection (join Turbine_ID if available)
        top_idx = np.argsort(scores)[::-1][:50]  # indices of top scores
        cols_to_save = ["Turbine_ID"] if "Turbine_ID" in df.columns else []
        top_df = pd.DataFrame({
            "row_index": X_test.index[top_idx],
            "score": scores[top_idx],
            "true_label": y_test.iloc[top_idx].values
        })
        for c in cols_to_save:
            top_df[c] = df.loc[X_test.index[top_idx], c].values
        top_df.to_csv(outdir / f"top_anomalies_{name}.csv", index=False)

    # ---- Consolidate and save anomaly results ----
    anom_df = pd.DataFrame(anom_results).set_index("model").sort_values("f1", ascending=False)
    anom_df.to_csv(outdir / "anomaly_results.csv")

    anom_top_df = pd.DataFrame(anom_top_rows).set_index("model")
    anom_top_df.to_csv(outdir / "top_thresholds_anomaly.csv")

    # Save per-row anomaly scores for transparency
    combined_scores = pd.DataFrame({
        "row_index": X_test.index,
        "true_label": y_test.values,
        "iso_score": iso_scores,
        "ocsvm_score": oc_scores
    }).sort_values("iso_score", ascending=False)
    if "Turbine_ID" in df.columns:
        combined_scores["Turbine_ID"] = df.loc[X_test.index, "Turbine_ID"]
    combined_scores.to_csv(outdir / "anomaly_scores_test.csv", index=False)

    # Optional tiny HTML summary for anomaly section (easy copy/paste into docs)
    try:
        html = outdir / "summary_anomaly.html"
        with open(html, "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>Anomaly Summary</title></head><body>")
            f.write("<h1>Unsupervised Anomaly Detection — Summary</h1>")
            f.write("<h2>Results</h2>")
            f.write(anom_df.round(4).to_html())
            f.write("<h2>Top Threshold Picks</h2>")
            f.write(anom_top_df.round(4).to_html())
            f.write("</body></html>")
        print(f"- HTML summary: {html.name}")
    except Exception as e:
        warnings.warn(f"Could not write anomaly HTML summary: {e}")

    # ---- Final console summary ----
    print("\n=== SUPERVISED (SMOTE) SNAPSHOT ===")
    print(results_sup.round(4))
    print("\nTop thresholds (Supervised):")
    print(top_sup_df.round(4))
    print("\n=== ANOMALY RESULTS ===")
    print(anom_df.round(4))
    print("\nTop thresholds (Anomaly):")
    print(anom_top_df.round(4))
    print("\nArtifacts saved to:", outdir.resolve())


# ----------------------
# 4) Script entry point
# ----------------------
if __name__ == "__main__":  # this ensures main() runs only when you execute this file directly
    main()
