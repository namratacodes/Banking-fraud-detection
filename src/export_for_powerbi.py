"""
Phase 7c: Export data to flat CSVs for Power BI.

Power BI can read CSVs natively with zero setup (no ODBC driver needed),
so this script pulls everything from the SQLite db + JSON reports into a
dedicated powerbi_exports/ folder, ready to drag-and-drop into Power BI's
"Get Data" -> "Text/CSV" dialog.
"""

import json
import sqlite3
import pandas as pd
from pathlib import Path

OUT_DIR = Path("../data/powerbi_exports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect("../data/processed/fraud_analytics.db")

# 1. Full transaction table (what most visuals will be built from)
df = pd.read_sql("SELECT * FROM transactions", conn)
df.to_csv(OUT_DIR / "transactions.csv", index=False)
print(f"Exported transactions.csv ({len(df)} rows)")

# 2. Pre-aggregated summary tables matching fraud_analysis.sql queries,
# so Power BI doesn't need to do the grouping itself
queries = {
    "fraud_by_hour": """
        SELECT hour_of_day, COUNT(*) AS total_transactions, SUM(Class) AS fraud_count,
               ROUND(100.0*SUM(Class)/COUNT(*),4) AS fraud_rate_pct
        FROM transactions GROUP BY hour_of_day ORDER BY hour_of_day
    """,
    "fraud_by_amount_band": """
        SELECT amount_band, COUNT(*) AS total_transactions, SUM(Class) AS fraud_count,
               ROUND(100.0*SUM(Class)/COUNT(*),4) AS fraud_rate_pct, ROUND(AVG(Amount),2) AS avg_amount
        FROM transactions GROUP BY amount_band
    """,
    "fraud_by_category": """
        SELECT merchant_category, COUNT(*) AS total_transactions, SUM(Class) AS fraud_count,
               ROUND(100.0*SUM(Class)/COUNT(*),4) AS fraud_rate_pct
        FROM transactions GROUP BY merchant_category ORDER BY fraud_rate_pct DESC
    """,
    "fraud_by_state": """
        SELECT state, COUNT(*) AS total_transactions, SUM(Class) AS fraud_count,
               ROUND(100.0*SUM(Class)/COUNT(*),4) AS fraud_rate_pct
        FROM transactions GROUP BY state ORDER BY fraud_rate_pct DESC
    """,
    "fraud_by_month": """
        SELECT transaction_month, COUNT(*) AS total_transactions, SUM(Class) AS fraud_count,
               ROUND(100.0*SUM(Class)/COUNT(*),4) AS fraud_rate_pct,
               ROUND(SUM(CASE WHEN Class=1 THEN Amount ELSE 0 END),2) AS fraud_amount_at_risk
        FROM transactions GROUP BY transaction_month ORDER BY transaction_month
    """,
    "day_vs_night": """
        SELECT CASE WHEN is_night=1 THEN 'Night' ELSE 'Day' END AS period,
               COUNT(*) AS total_transactions, SUM(Class) AS fraud_count,
               ROUND(100.0*SUM(Class)/COUNT(*),4) AS fraud_rate_pct
        FROM transactions GROUP BY is_night
    """,
}
for name, q in queries.items():
    pd.read_sql(q, conn).to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(f"Exported {name}.csv")

conn.close()

# 3. Model performance reports (already CSVs, just copy)
import shutil
for fname in ["model_comparison.csv", "shap_feature_importance.csv",
              "threshold_sweep.csv", "imbalance_strategy_comparison.csv"]:
    src = Path("../reports") / fname
    if src.exists():
        shutil.copy(src, OUT_DIR / fname)
        print(f"Copied {fname}")

# 4. Flatten the SHAP example explanations JSON into a CSV for the
# Transaction Explorer page
shap_json_path = Path("../reports/shap_example_explanations.json")
if shap_json_path.exists():
    with open(shap_json_path) as f:
        examples = json.load(f)
    rows = []
    for ex in examples:
        rows.append({
            "transaction_id": ex["transaction_id"],
            "amount": ex["amount"],
            "fraud_probability": ex["fraud_probability"],
            "flagged": ex["flagged"],
            "top_reasons": " | ".join(ex["top_reasons"]),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "shap_example_explanations.csv", index=False)
    print("Exported shap_example_explanations.csv")

print(f"\nAll files ready in {OUT_DIR.resolve()}")
print("Open Power BI -> Get Data -> Text/CSV -> select files from this folder.")