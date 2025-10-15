#!/usr/bin/env python3
"""
Final One-Pager Builder — Decision-Oriented Captions + Top Anomalies + Threshold Tables

Generates a concise, executive-ready HTML (and optional PDF) one-pager:
- Copies & embeds key plots and CSVs from supervised and anomaly outputs
- Adds short, action-focused captions under every figure and table
- Embeds "Top anomalies" tables (first 20 rows)
- Embeds "Top threshold picks" tables for supervised and anomaly models

Usage:
  python make_final_onepager.py \
    --supervised_dir "/path/to/outputs_balanced_or_nb" \
    --anomaly_dir "/path/to/outputs_balanced_anomaly_or_nb" \
    --out "/path/to/Final_Project_Report"
"""

import argparse
from pathlib import Path
import pandas as pd

# Optional PDF backend
PDF_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False


def read_csv_safe(path: Path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None


def copy_if_exists(path: Path, outdir: Path):
    if path and path.exists():
        target = outdir / path.name
        if str(path.resolve()) != str(target.resolve()):
            target.write_bytes(path.read_bytes())
        return target
    return None


# --- DECISION-ORIENTED CAPTIONS ---
CAPTIONS = {
    # Supervised
    "roc_both.png": (
        "Pick the model with the higher ROC curve; it separates failure vs normal better.",
        "Use this model for alerts when you need robust ranking even as thresholds change."
    ),
    "pr_both.png": (
        "Prioritize the model with the higher PR curve; it finds more true maintenance cases with fewer false alarms.",
        "Use this to set realistic expectations in imbalanced data: higher area ⇒ better early-warning utility."
    ),
    "f1_vs_threshold.png": (
        "Choose the threshold at the F1 peak when you want a balanced trade‑off of misses vs false alarms.",
        "Shift right for fewer false alarms (higher precision) or left for fewer misses (higher recall)."
    ),
    "pr_vs_threshold_RandomForest.png": (
        "Pick the RF threshold where recall meets your minimum precision requirement.",
        "Use this to set the alert cutoff based on your tolerated false alarm rate."
    ),
    "pr_vs_threshold_GBDT.png": (
        "Pick the GBDT threshold where recall meets your minimum precision requirement.",
        "Use this to operationalize the boosted model with business‑aligned targets."
    ),
    "feature_importance_rf_smote.png": (
        "Focus monitoring and preventive actions on the top RF sensors; they drive maintenance predictions.",
        "Use these to guide sensor QA, alarms, and spare‑parts readiness."
    ),
    "feature_importance_gbdt_smote.png": (
        "Sensors with highest GBDT importance should be prioritized for health checks and data quality.",
        "If RF and GBDT agree, treat those sensors as critical for reliability."
    ),

    # Anomaly
    "anomaly_roc_both.png": (
        "Select the anomaly model with the higher ROC curve for more reliable ranking of unusual behavior.",
        "Use this when screening many turbines and triaging limited inspection capacity."
    ),
    "anomaly_pr_both.png": (
        "Choose the anomaly model with the stronger PR curve for early-warning alerts with fewer false positives.",
        "Deploy this for proactive maintenance where positive cases are rare."
    ),
    "anomaly_roc_IsolationForest.png": (
        "If IsolationForest’s ROC is higher, trust it to prioritize which turbines to inspect first.",
        "Adopt it for ranking when labels are scarce or noisy."
    ),
    "anomaly_pr_IsolationForest.png": (
        "Strong PR for IsolationForest means you can flag fewer false alarms at useful recall.",
        "Use this thresholding view to align alerts with team bandwidth."
    ),
    "anomaly_roc_OneClassSVM.png": (
        "If One‑Class SVM shows higher ROC, it better separates normal vs anomalous behavior.",
        "Prefer it for ranking when the normal boundary is well‑defined."
    ),
    "anomaly_pr_OneClassSVM.png": (
        "A stronger PR curve for One‑Class SVM indicates more precise anomaly alerts at a given recall.",
        "Set thresholds here to control the volume of anomaly tickets."
    ),
    "anomaly_hist_iso.png": (
        "Right‑tail IsolationForest scores are the prime candidates for inspection.",
        "Set the cutoff to catch enough anomalies without overwhelming the team."
    ),
    "anomaly_hist_ocsvm.png": (
        "Right‑tail One‑Class SVM scores indicate the most unusual turbines.",
        "Tune the threshold to balance early warnings with operational workload."
    ),
    "anomaly_pr_vs_threshold_IsolationForest.png": (
        "Pick the IsolationForest score cutoff that meets your minimum precision or recall target.",
        "Lower cutoffs catch more issues (recall↑) but increase false alarms; raise for fewer tickets."
    ),
    "anomaly_pr_vs_threshold_OneClassSVM.png": (
        "Set the One‑Class SVM cutoff where recall is acceptable and precision meets your SLA.",
        "Use this for policy‑driven alerting (e.g., precision ≥ 0.5)."
    ),
}

TABLE_CAPTIONS = {
    "supervised": (
        "Compare models and thresholds here; pick the configuration with the best F1 for a balanced policy.",
        "If false alarms are costly, choose a higher threshold that preserves acceptable recall."
    ),
    "anomaly": (
        "Use ROC‑AUC/PR‑AUC to choose the anomaly model that best prioritizes inspections.",
        "Higher PR‑AUC indicates more true issues per alert — ideal for limited maintenance capacity."
    ),
    "top_anomalies": (
        "Start with these top anomalies — they are the most unusual and likely to require attention.",
        "Create tickets or schedule checks for the highest‑scored turbines."
    ),
    "top_thresholds_supervised": (
        "Recommended supervised cutoffs: best F1 and highest recall under precision ≥ 0.3 / ≥ 0.5.",
        "Adopt one of these thresholds to align alerts with your tolerance for false positives."
    ),
    "top_thresholds_anomaly": (
        "Recommended anomaly cutoffs: best F1 and precision‑constrained picks for each model.",
        "Select the cutoff that fits your team’s capacity and risk appetite."
    ),
}


def img_tag(path: Path, width_px=520):
    if not path or not path.exists():
        return ""
    return f"<img src='{path.name}' width='{width_px}'>"


def caption_html(line1: str, line2: str):
    return f"<div class='caption'><div>{line1}</div><div>{line2}</div></div>"


def build_html(supervised_dir: Path, anomaly_dir: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect supervised artifacts
    sup_csv = None
    for name in ["results_balanced_thresholds.csv", "results_balanced_thresholds_nb.csv"]:
        p = supervised_dir / name
        if p.exists():
            sup_csv = p; break

    sup_pngs = []
    for name in ["roc_both.png", "pr_both.png", "f1_vs_threshold.png",
                 "pr_vs_threshold_RandomForest.png", "pr_vs_threshold_GBDT.png",
                 "feature_importance_rf_smote.png", "feature_importance_gbdt_smote.png"]:
        p = supervised_dir / name
        if p.exists(): sup_pngs.append(p)
    for fn in sorted(supervised_dir.glob("confusion_matrix_*thr*.png"))[:4]:
        sup_pngs.append(fn)

    # Collect anomaly artifacts
    an_csv = None
    for name in ["anomaly_results.csv", "anomaly_results_nb.csv"]:
        p = anomaly_dir / name
        if p.exists():
            an_csv = p; break
    an_pngs = []
    for name in ["anomaly_roc_both.png", "anomaly_pr_both.png",
                 "anomaly_roc_IsolationForest.png", "anomaly_pr_IsolationForest.png",
                 "anomaly_roc_OneClassSVM.png", "anomaly_pr_OneClassSVM.png",
                 "anomaly_hist_iso.png", "anomaly_hist_ocsvm.png",
                 "anomaly_pr_vs_threshold_IsolationForest.png", "anomaly_pr_vs_threshold_OneClassSVM.png"]:
        p = anomaly_dir / name
        if p.exists(): an_pngs.append(p)

    # Optional top anomalies tables
    top_iso = anomaly_dir / "top_anomalies_IsolationForest.csv"
    top_oc  = anomaly_dir / "top_anomalies_OneClassSVM.csv"
    top_iso_df = read_csv_safe(top_iso)
    top_oc_df  = read_csv_safe(top_oc)

    # Copy assets into outdir
    copied_sup_pngs = [copy_if_exists(p, outdir) for p in sup_pngs]
    copied_an_pngs  = [copy_if_exists(p, outdir) for p in an_pngs]
    sup_df = read_csv_safe(sup_csv) if sup_csv else None
    an_df  = read_csv_safe(an_csv)  if an_csv  else None
    if sup_csv: copy_if_exists(sup_csv, outdir)
    if an_csv:  copy_if_exists(an_csv, outdir)

    # Also copy top threshold CSVs if present
    copy_if_exists(supervised_dir / "top_thresholds_supervised.csv", outdir)
    copy_if_exists(anomaly_dir / "top_thresholds_anomaly.csv", outdir)

    # Build HTML
    html_path = outdir / "final_onepager.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>Final One-Pager</title>")
        f.write("""<style>
        body{font-family:Arial,Helvetica,sans-serif;margin:22px;}
        h1,h2,h3{margin:8px 0;}
        table{border-collapse:collapse;margin-bottom:4px;}
        th,td{border:1px solid #ccc;padding:6px 8px;}
        .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
        .muted{color:#666;font-size:12px}
        .caption{color:#222;font-size:12px;margin:-6px 0 12px 0;}
        </style>""")
        f.write("</head><body>")
        f.write("<h1>Wind Turbine Maintenance — Executive One-Pager</h1>")
        f.write("<p class='muted'>Decision‑oriented summary of supervised classification and unsupervised anomaly detection.</p>")

        # Supervised section
        f.write("<h2>Supervised (Balanced with SMOTE)</h2>")
        if sup_df is not None:
            f.write(sup_df.round(4).to_html(index=True))
            f.write(caption_html(*TABLE_CAPTIONS["supervised"]))
        else:
            f.write("<p class='muted'>No supervised results CSV found.</p>")

        # Top thresholds table (Supervised)
        top_sup = read_csv_safe(supervised_dir / "top_thresholds_supervised.csv")
        if top_sup is not None:
            f.write("<h3>Top threshold picks (Supervised)</h3>")
            f.write(top_sup.round(4).to_html(index=True))
            f.write(caption_html(*TABLE_CAPTIONS["top_thresholds_supervised"]))

        if copied_sup_pngs:
            f.write("<div class='grid'>")
            for p in copied_sup_pngs[:8]:
                if p:
                    f.write(img_tag(p))
                    base = p.name
                    cap = CAPTIONS.get(base, ("Model figure", "Inspect trends vs baseline to pick an action."))
                    f.write(caption_html(*cap))
            f.write("</div>")

        # Anomaly section
        f.write("<h2>Unsupervised Anomaly Detection</h2>")
        if an_df is not None:
            f.write(an_df.round(4).to_html(index=True))
            f.write(caption_html(*TABLE_CAPTIONS["anomaly"]))
        else:
            f.write("<p class='muted'>No anomaly results CSV found.</p>")

        # Top thresholds table (Anomaly)
        top_anom = read_csv_safe(anomaly_dir / "top_thresholds_anomaly.csv")
        if top_anom is not None:
            f.write("<h3>Top threshold picks (Anomaly)</h3>")
            f.write(top_anom.round(4).to_html(index=True))
            f.write(caption_html(*TABLE_CAPTIONS["top_thresholds_anomaly"]))

        if copied_an_pngs:
            f.write("<div class='grid'>")
            for p in copied_an_pngs[:8]:
                if p:
                    f.write(img_tag(p))
                    base = p.name
                    cap = CAPTIONS.get(base, ("Anomaly figure", "Use pattern and area to set an actionable cutoff."))
                    f.write(caption_html(*cap))
            f.write("</div>")

        # Top anomalies tables
        if top_iso_df is not None or top_oc_df is not None:
            f.write("<h2>Top Anomalies (First 20)</h2>")
            if top_iso_df is not None:
                f.write("<h3>IsolationForest</h3>")
                f.write(top_iso_df.head(20).to_html(index=False))
                f.write(caption_html(*TABLE_CAPTIONS["top_anomalies"]))
            if top_oc_df is not None:
                f.write("<h3>One‑Class SVM</h3>")
                f.write(top_oc_df.head(20).to_html(index=False))
                f.write(caption_html(*TABLE_CAPTIONS["top_anomalies"]))

        f.write("<hr><p class='muted'>Generated by make_final_onepager.py</p>")
        f.write("</body></html>")

    return html_path


def build_pdf_from_html_assets(outdir: Path, html_path: Path):
    if not PDF_AVAILABLE:
        return None
    pdf_path = outdir / "final_onepager.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Wind Turbine Maintenance — Executive One-Pager", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Supervised (Balanced with SMOTE)", styles['Heading2']))

    # Supervised table (if copied)
    sup_csv = None
    for name in ["results_balanced_thresholds.csv", "results_balanced_thresholds_nb.csv"]:
        p = outdir / name
        if p.exists():
            sup_csv = p; break
    if sup_csv:
        df = pd.read_csv(sup_csv).round(3)
        data = [df.columns.tolist()] + df.values.tolist()
        t = Table(data, hAlign='LEFT')
        t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.whitesmoke)]))
        story.append(t)
        story.append(Paragraph("Pick the best F1 for balance; raise threshold if false alarms are costly.", styles['Normal']))
        story.append(Spacer(1, 12))

    # Add a couple supervised images
    for name in ["roc_both.png", "pr_both.png"]:
        p = outdir / name
        if p.exists():
            story.append(Image(str(p), width=400, height=280))
            cap = " ".join(CAPTIONS[name])
            story.append(Paragraph(cap, styles['Normal']))
            story.append(Spacer(1, 8))

    story.append(Paragraph("Unsupervised Anomaly Detection", styles['Heading2']))

    an_csv = None
    for name in ["anomaly_results.csv", "anomaly_results_nb.csv"]:
        p = outdir / name
        if p.exists():
            an_csv = p; break
    if an_csv:
        df = pd.read_csv(an_csv).round(3)
        data = [df.columns.tolist()] + df.values.tolist()
        t = Table(data, hAlign='LEFT')
        t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.whitesmoke)]))
        story.append(t)
        story.append(Paragraph("Choose the anomaly model with higher PR‑AUC; set a cutoff that matches SLA.", styles['Normal']))
        story.append(Spacer(1, 12))

    for name in ["anomaly_roc_both.png", "anomaly_pr_both.png"]:
        p = outdir / name
        if p.exists():
            story.append(Image(str(p), width=400, height=280))
            cap = " ".join(CAPTIONS[name])
            story.append(Paragraph(cap, styles['Normal']))
            story.append(Spacer(1, 8))

    doc.build(story)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Create an executive one-page HTML/PDF with action-focused captions.")
    parser.add_argument("--supervised_dir", type=str, required=True, help="Directory with supervised outputs (CSV + images)")
    parser.add_argument("--anomaly_dir", type=str, required=True, help="Directory with anomaly outputs (CSV + images)")
    parser.add_argument("--out", type=str, required=True, help="Directory where the one-pager will be created")
    args = parser.parse_args()

    supervised_dir = Path(args.supervised_dir)
    anomaly_dir = Path(args.anomaly_dir)
    outdir = Path(args.out)

    if not supervised_dir.exists():
        print(f"[WARN] Supervised dir not found: {supervised_dir}")
    if not anomaly_dir.exists():
        print(f"[WARN] Anomaly dir not found: {anomaly_dir}")

    html_path = build_html(supervised_dir, anomaly_dir, outdir)
    print(f"HTML one-pager: {html_path.resolve()}")

    pdf_path = build_pdf_from_html_assets(outdir, html_path)
    if pdf_path:
        print(f"PDF one-pager:  {pdf_path.resolve()}")
    else:
        print("PDF export skipped (reportlab not installed). To enable: pip install reportlab")


if __name__ == "__main__":
    main()
