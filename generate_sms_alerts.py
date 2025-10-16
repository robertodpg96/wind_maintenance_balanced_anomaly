#!/usr/bin/env python3
"""
generate_sms_alerts.py
---------------------------------------
Reads the supervised alerts file (union of RF + GBDT) and generates an
SMS-ready CSV file with Turbine_ID + alert message.

Usage:
    python generate_sms_alerts.py \
      --alerts "/path/to/outputs_balanced_anomaly/supervised_alerts_union.csv" \
      --out "/path/to/outputs_balanced_anomaly/sms_alert_recipients.csv" \
      --site_name "Wind Farm Alpha"

The generated CSV can be sent directly to an SMS gateway or notification API.
"""

import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create SMS alert recipients from supervised alerts.")
    parser.add_argument("--alerts", required=True, help="Path to supervised_alerts_union.csv")
    parser.add_argument("--out", default="sms_alert_recipients.csv", help="Output CSV file")
    parser.add_argument("--site_name", default="Wind Farm", help="Optional site/farm name for message context")
    args = parser.parse_args()

    alerts_path = Path(args.alerts)
    if not alerts_path.exists():
        raise FileNotFoundError(f"Alerts file not found: {alerts_path}")

    df = pd.read_csv(alerts_path)
    if "Turbine_ID" not in df.columns:
        raise ValueError("The supervised_alerts_union.csv file must include a Turbine_ID column.")

    # Current timestamp for message context
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate message per turbine
    messages = []
    for _, row in df.iterrows():
        models = row.get("triggered_by", "model").replace(",", " & ")
        msg = (
            f"⚠️ Maintenance Alert: Turbine {row['Turbine_ID']} at {args.site_name} "
            f"requires inspection. Triggered by {models} model(s). "
            f"Generated on {timestamp}."
        )
        messages.append(msg)

    sms_df = pd.DataFrame({
        "Turbine_ID": df["Turbine_ID"],
        "Message": messages
    })

    sms_df.to_csv(args.out, index=False)
    print(f"✅ SMS alert recipients saved to: {Path(args.out).resolve()}")
    print(f"Total alerts prepared: {len(sms_df)}")

if __name__ == "__main__":
    main()
