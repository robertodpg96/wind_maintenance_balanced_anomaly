#!/usr/bin/env python3
"""
Wind Turbine Maintenance — FINAL SCRIPT (with supervised alerts exports)
Supervised (SMOTE + threshold tuning) + Unsupervised Anomaly Detection

Adds:
- Exports per-model supervised alerts: supervised_alerts_RF.csv, supervised_alerts_GBDT.csv
- Exports union of alerts (either model triggers): supervised_alerts_union.csv
- Threshold policy configurable: --supervised_alert_policy [best_f1|min_precision], --min_precision 0.5

Usage:
  python train_wind_maintenance_models_balanced_anomaly_final.py \
      --data /path/to/wind_turbine_maintenance_data.csv \
      --out outputs_balanced_anomaly \
      --contamination 0.05 \
      --supervised_alert_policy best_f1
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay,
    precision_recall_curve, classification_report
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM

# Optional boosters
XGB_AVAILABLE = False
LGBM_AVAILABLE = False
SKLEARN_GB_AVAILABLE = True

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    warnings.warn("xgboost not available; will try lightgbm or sklearn's GradientBoosting instead.")
if not XGB_AVAILABLE:
    try:
        from lightgbm import LGBMClassifier
        LGBM_AVAILABLE = True
    except Exception:
        warnings.warn("lightgbm not available; will fall back to sklearn.")
try:
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:
    SKLEARN_GB_AVAILABLE = False

# SMOTE (class imbalance handling)
try:
    from imblearn.over_sampling import SMOTE
except Exception as e:
    raise SystemExit("Missing dependency: imbalanced-learn. Install with: pip install imbalanced-learn")


# -------------------------- Utilities --------------------------
def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    if 'Maintenance_Label' not in df.columns:
        raise ValueError("Expected a 'Maintenance_Label' column in the dataset.")

    # Coerce and force binary target
    y_raw = pd.to_numeric(df['Maintenance_Label'], errors='coerce').fillna(0)
    y = (y_raw > 0).astype(int)

    # Numeric features only
    X = df.drop(columns=['Maintenance_Label']).select_dtypes(include=[np.number])

    print("Label value counts (entire dataset):")
    print(pd.Series(y).value_counts(dropna=False).to_string())

    return df, X, y


def make_rf():
    return RandomForestClassifier(
        n_estimators=400,
        n_jobs=-1,
        random_state=42
    )


def make_gbdt():
    if XGB_AVAILABLE:
        return XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=1.0
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


def get_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        d_min, d_max = d.min(), d.max()
        return (d - d_min) / (d_max - d_min + 1e-9)
    return model.predict(X)


def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics["roc_auc"] = np.nan
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        except Exception:
            metrics["pr_auc"] = np.nan
    return metrics


def sup_thr_sweep(y_true, y_prob, out_csv: Path):
    rows = []
    thresholds = np.linspace(0.05, 0.95, 19)
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        rows.append({
            "threshold": thr,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        })
    df_thr = pd.DataFrame(rows); df_thr.to_csv(out_csv, index=False)
    best = df_thr.iloc[df_thr['f1'].values.argmax()]
    return float(best['threshold']), df_thr


def threshold_sweep_scores(y_true, scores, out_csv: Path):
    rows = []
    thr_values = np.linspace(np.percentile(scores, 5), np.percentile(scores, 95), 31)
    for thr in thr_values:
        y_pred = (scores >= thr).astype(int)
        rows.append({
            "threshold": thr,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        })
    df_thr = pd.DataFrame(rows)
    df_thr.to_csv(out_csv, index=False)
    best_row = df_thr.iloc[df_thr['f1'].values.argmax()]
    return float(best_row['threshold']), df_thr


def _top3_thresholds(df_thr: pd.DataFrame, min_precisions=(0.3, 0.5)):
    out = {}
    if df_thr is None or df_thr.empty:
        return out
    i = df_thr['f1'].values.argmax()
    out['best_f1_threshold'] = float(df_thr.iloc[i]['threshold'])
    out['best_f1'] = float(df_thr.iloc[i]['f1'])
    out['best_f1_precision'] = float(df_thr.iloc[i]['precision'])
    out['best_f1_recall'] = float(df_thr.iloc[i]['recall'])
    for pmin in min_precisions:
        df_ok = df_thr[df_thr['precision'] >= pmin]
        if len(df_ok):
            j = df_ok['recall'].values.argmax()
            row = df_ok.iloc[j]
            out[f'prec>={pmin}_threshold'] = float(row['threshold'])
            out[f'prec>={pmin}_recall'] = float(row['recall'])
            out[f'prec>={pmin}_f1'] = float(row['f1'])
        else:
            out[f'prec>={pmin}_threshold'] = None
            out[f'prec>={pmin}_recall'] = None
            out[f'prec>={pmin}_f1'] = None
    return out


def plot_importance(model, feature_names, title, out_path: Path, top_k: int = 15):
    if not hasattr(model, "feature_importances_"):
        warnings.warn(f"No feature_importances_ for {title}.")
        return None
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.barh(range(len(order))[::-1], imp[order][::-1])
    ax.set_yticks(range(len(order))[::-1])
    ax.set_yticklabels([feature_names[i] for i in order][::-1])
    ax.set_xlabel("Importance"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# -------------------------- Main --------------------------
def main():
    parser = argparse.ArgumentParser(description="Balanced RF+GBDT with SMOTE + Anomaly Detection (FINAL + supervised alerts export)")
    parser.add_argument("--data", required=True, type=str, help="Path to CSV with Maintenance_Label")
    parser.add_argument("--out", default="outputs_balanced_anomaly", type=str, help="Output directory")
    parser.add_argument("--test_size", default=0.2, type=float, help="Test size fraction")
    parser.add_argument("--contamination", default=0.05, type=float, help="Assumed outlier proportion for unsupervised models")
    parser.add_argument("--supervised_alert_policy", default="best_f1", choices=["best_f1", "min_precision"],
                        help="How to pick alert thresholds for supervised exports")
    parser.add_argument("--min_precision", default=0.5, type=float,
                        help="Minimum precision when --supervised_alert_policy=min_precision")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found at: {data_path}. Tip: wrap spaced paths in quotes.")

    outdir = Path(args.out); ensure_outdir(outdir)

    print("Loading data...")
    df, X, y = load_data(data_path)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    print("Label counts (train):", np.unique(y_train, return_counts=True))
    print("Label counts (test) :", np.unique(y_test, return_counts=True))

    # ---------------- Supervised (SMOTE) ----------------
    print("Applying SMOTE to the training set...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("Resampled label counts:", np.unique(y_res, return_counts=True))

    print("Training RandomForest (SMOTE)...")
    rf = make_rf().fit(X_res, y_res)

    print("Training GBDT (SMOTE)...")
    gb = make_gbdt().fit(X_res, y_res)

    # Evaluate probabilities on test
    print("Evaluating supervised models...")
    y_proba_rf = get_proba(rf, X_test)
    y_proba_gb = get_proba(gb, X_test)

    # ROC/PR plots
    fig, ax = plt.subplots(figsize=(5,4))
    RocCurveDisplay.from_predictions(y_test, y_proba_rf, name="RandomForest", ax=ax)
    RocCurveDisplay.from_predictions(y_test, y_proba_gb, name="GBDT", ax=ax)
    ax.set_title("ROC — RF vs GBDT")
    fig.tight_layout(); fig.savefig(outdir / "roc_both.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5,4))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba_rf, name="RandomForest", ax=ax)
    PrecisionRecallDisplay.from_predictions(y_test, y_proba_gb, name="GBDT", ax=ax)
    ax.set_title("Precision–Recall — RF vs GBDT")
    fig.tight_layout(); fig.savefig(outdir / "pr_both.png", dpi=150); plt.close(fig)

    # Precision–Recall vs Threshold (supervised)
    for name, probs in [("RandomForest", y_proba_rf), ("GBDT", y_proba_gb)]:
        p_vals, r_vals, thr_vals = precision_recall_curve(y_test, probs)
        thr_plot = np.concatenate([thr_vals, [thr_vals[-1] if thr_vals.size else 0.5]]) if thr_vals.size else np.array([0.5])
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(thr_plot, p_vals, label="Precision")
        ax.plot(thr_plot, r_vals, label="Recall")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score"); ax.set_title(f"Precision–Recall vs Threshold — {name}")
        ax.legend(); fig.tight_layout()
        fig.savefig(outdir / f"pr_vs_threshold_{name}.png", dpi=150)
        plt.close(fig)

    # Supervised threshold sweeps + top threshold tables
    best_thr_rf, df_thr_rf = sup_thr_sweep(y_test, y_proba_rf, outdir / "threshold_sweep_rf.csv")
    best_thr_gb, df_thr_gb = sup_thr_sweep(y_test, y_proba_gb, outdir / "threshold_sweep_gbdt.csv")
    top_sup_df = pd.DataFrame([
        {'model':'RandomForest', **_top3_thresholds(df_thr_rf)},
        {'model':'GBDT', **_top3_thresholds(df_thr_gb)},
    ]).set_index('model')
    top_sup_df.to_csv(outdir / 'top_thresholds_supervised.csv')

    # Evaluate at default 0.5 and tuned
    def eval_at(thr, proba):
        y_pred = (proba >= thr).astype(int)
        return compute_metrics(y_test, y_pred, proba), y_pred

    m_rf_05, _ = eval_at(0.50, y_proba_rf)
    m_rf_bt, _ = eval_at(best_thr_rf, y_proba_rf)
    m_gb_05, _ = eval_at(0.50, y_proba_gb)
    m_gb_bt, _ = eval_at(best_thr_gb, y_proba_gb)

    results_sup = pd.DataFrame([
        {"model": "RandomForest@0.50", **m_rf_05},
        {"model": f"RandomForest@{best_thr_rf:.2f}", **m_rf_bt},
        {"model": "GBDT@0.50", **m_gb_05},
        {"model": f"GBDT@{best_thr_gb:.2f}", **m_gb_bt},
    ]).set_index("model").sort_values("f1", ascending=False)
    results_sup.to_csv(outdir / "results_balanced_thresholds.csv")

    # === Export Supervised Alerts ===
    def _pick_thr(df_thr, policy, pmin):
        if policy == "best_f1":
            i = df_thr['f1'].values.argmax()
            return float(df_thr.iloc[i]['threshold']), "best_f1"
        df_ok = df_thr[df_thr['precision'] >= pmin]
        if len(df_ok):
            j = df_ok['recall'].values.argmax()
            return float(df_ok.iloc[j]['threshold']), f"min_precision>={pmin}"
        i = df_thr['f1'].values.argmax()
        return float(df_thr.iloc[i]['threshold']), "fallback_best_f1"

    thr_rf_alert, policy_rf = _pick_thr(df_thr_rf, args.supervised_alert_policy, args.min_precision)
    thr_gb_alert, policy_gb = _pick_thr(df_thr_gb, args.supervised_alert_policy, args.min_precision)

    sup_rf = pd.DataFrame(index=X_test.index)
    sup_rf['predicted_prob'] = y_proba_rf
    sup_rf['predicted_label'] = (y_proba_rf >= thr_rf_alert).astype(int)
    sup_rf['true_label'] = y_test.values
    if "Turbine_ID" in df.columns:
        sup_rf['Turbine_ID'] = df.loc[X_test.index, "Turbine_ID"].values
    rf_alerts = sup_rf[sup_rf['predicted_label'] == 1].copy()
    rf_alerts.to_csv(outdir / "supervised_alerts_RF.csv", index=False)

    sup_gb = pd.DataFrame(index=X_test.index)
    sup_gb['predicted_prob'] = y_proba_gb
    sup_gb['predicted_label'] = (y_proba_gb >= thr_gb_alert).astype(int)
    sup_gb['true_label'] = y_test.values
    if "Turbine_ID" in df.columns:
        sup_gb['Turbine_ID'] = df.loc[X_test.index, "Turbine_ID"].values
    gb_alerts = sup_gb[sup_gb['predicted_label'] == 1].copy()
    gb_alerts.to_csv(outdir / "supervised_alerts_GBDT.csv", index=False)

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
    union['triggered_by'] = union.apply(lambda r: ",".join([m for m,b in [('RF', r['rf_trigger']), ('GBDT', r['gbdt_trigger'])] if b]), axis=1)
    union_alerts = union[(union['rf_trigger']==1) | (union['gbdt_trigger']==1)].copy()
    union_alerts['rf_threshold_used'] = thr_rf_alert
    union_alerts['gbdt_threshold_used'] = thr_gb_alert
    union_alerts['rf_policy'] = policy_rf
    union_alerts['gbdt_policy'] = policy_gb
    union_alerts.to_csv(outdir / "supervised_alerts_union.csv", index=False)

    print(f"Supervised alerts exported: RF={len(rf_alerts)}, GBDT={len(gb_alerts)}, Union={len(union_alerts)}")

    # Feature importances
    _ = plot_importance(rf, feature_names, "Feature Importance — RF (SMOTE)", outdir / "feature_importance_rf_smote.png")
    _ = plot_importance(gb, feature_names, "Feature Importance — GBDT (SMOTE)", outdir / "feature_importance_gbdt_smote.png")

    # ---------------- Unsupervised Anomaly Detection ----------------
    print("Training anomaly detectors on *normal* training data only...")
    X_train_norm = X_train[y_train == 0]

    iso = IsolationForest(
        n_estimators=300,
        contamination=args.contamination,
        random_state=42
    ).fit(X_train_norm)

    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=max(min(args.contamination, 0.49), 0.01)).fit(X_train_norm)

    # Get anomaly scores on test set (higher = more anomalous)
    iso_scores = -iso.score_samples(X_test)
    oc_scores  = -oc.decision_function(X_test)

    # Evaluate against labels
    anom_results = []
    anom_top_rows = []
    for name, scores in [("IsolationForest", iso_scores), ("OneClassSVM", oc_scores)]:
        roc = roc_auc_score(y_test, scores)
        pr = average_precision_score(y_test, scores)
        thr_csv = outdir / f"anomaly_threshold_sweep_{name}.csv"
        best_thr, df_thr = threshold_sweep_scores(y_test, scores, thr_csv)
        top = _top3_thresholds(df_thr)
        anom_top_rows.append({'model': name, **top})

        y_pred = (scores >= best_thr).astype(int)
        m = compute_metrics(y_test, y_pred, scores)
        m["roc_auc"] = roc
        m["pr_auc"] = pr
        m["best_thr"] = best_thr
        m["model"] = name
        anom_results.append(m)

        # ROC/PR
        fig, ax = plt.subplots(figsize=(5,4))
        RocCurveDisplay.from_predictions(y_test, scores, name=name, ax=ax)
        ax.set_title(f"Anomaly ROC — {name}")
        fig.tight_layout(); fig.savefig(outdir / f"anomaly_roc_{name}.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5,4))
        PrecisionRecallDisplay.from_predictions(y_test, scores, name=name, ax=ax)
        ax.set_title(f"Anomaly PR — {name}")
        fig.tight_layout(); fig.savefig(outdir / f"anomaly_pr_{name}.png", dpi=150); plt.close(fig)

        # PR vs threshold for anomaly scores
        p_vals, r_vals, thr_vals = precision_recall_curve(y_test, scores)
        thr_plot = np.concatenate([thr_vals, [thr_vals[-1] if thr_vals.size else scores.mean()]]) if thr_vals.size else np.array([scores.mean()])
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(thr_plot, p_vals, label="Precision")
        ax.plot(thr_plot, r_vals, label="Recall")
        ax.set_xlabel("Score threshold"); ax.set_ylabel("Score"); ax.set_title(f"PR vs Threshold — {name}")
        ax.legend(); fig.tight_layout()
        fig.savefig(outdir / f"anomaly_pr_vs_threshold_{name}.png", dpi=150); plt.close(fig)

        # Score histogram
        fig, ax = plt.subplots(figsize=(5,4))
        ax.hist(scores, bins=40)
        ax.set_title(f"Anomaly Score Histogram — {name}")
        ax.set_xlabel("Score (higher = more anomalous)"); ax.set_ylabel("Count")
        fig.tight_layout(); fig.savefig(outdir / f"anomaly_hist_{name}.png", dpi=150); plt.close(fig)

        # Top anomalies table
        top_idx = np.argsort(scores)[::-1][:50]
        cols_to_save = ["Turbine_ID"] if "Turbine_ID" in df.columns else []
        top_df = pd.DataFrame({
            "row_index": X_test.index[top_idx],
            "score": scores[top_idx],
            "true_label": y_test.iloc[top_idx].values
        })
        for c in cols_to_save:
            top_df[c] = df.loc[X_test.index[top_idx], c].values
        top_df.to_csv(outdir / f"top_anomalies_{name}.csv", index=False)

    anom_df = pd.DataFrame(anom_results).set_index("model").sort_values("f1", ascending=False)
    anom_df.to_csv(outdir / "anomaly_results.csv")

    anom_top_df = pd.DataFrame(anom_top_rows).set_index('model')
    anom_top_df.to_csv(outdir / 'top_thresholds_anomaly.csv')

    # Combined anomaly scores
    combined_scores = pd.DataFrame({
        "row_index": X_test.index,
        "true_label": y_test.values,
        "iso_score": iso_scores,
        "ocsvm_score": oc_scores
    }).sort_values("iso_score", ascending=False)
    if "Turbine_ID" in df.columns:
        combined_scores["Turbine_ID"] = df.loc[X_test.index, "Turbine_ID"]
    combined_scores.to_csv(outdir / "anomaly_scores_test.csv", index=False)

    # Minimal HTML summary
    try:
        html = outdir / "summary_anomaly.html"
        with open(html, "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>Anomaly Summary</title></head><body>")
            f.write("<h1>Unsupervised Anomaly Detection — Summary</h1>")
            f.write("<h2>Anomaly Results</h2>")
            f.write(anom_df.round(4).to_html())
            f.write("<h2>Top Threshold Picks (Anomaly)</h2>")
            f.write(anom_top_df.round(4).to_html())
            f.write("</body></html>")
        print(f"- HTML summary: {html.name}")
    except Exception as e:
        warnings.warn(f"Could not write anomaly HTML summary: {e}")

    # Console summaries
    print("\\n=== SUPERVISED (SMOTE) RESULTS (higher is better) ===")
    print(results_sup.round(4))
    print("\\nTop thresholds (Supervised):\\n", top_sup_df.round(4))
    print("\\n=== ANOMALY RESULTS (higher is better; ROC/PR use scores) ===")
    print(anom_df.round(4))
    print("\\nTop thresholds (Anomaly):\\n", anom_top_df.round(4))
    print("\\nArtifacts saved to:", outdir.resolve())


if __name__ == "__main__":
    main()
