"""
Phase 7a: Load transaction data into SQLite for SQL analytics.

IMPORTANT - HONEST DATA NOTE:
The real Kaggle creditcard.csv contains only Time (48 hours of real data),
V1-V28, Amount, and Class. It does NOT include merchant category, geography,
or a real calendar date range (months/years) - these fields don't exist in
the anonymized source data.

To still demonstrate the SQL analytics patterns a real fraud ops team would
use (merchant risk, geographic patterns, monthly trend), this script adds
THREE CLEARLY-LABELED SIMULATED columns:
  - merchant_category : randomly assigned from a realistic category list
  - state              : randomly assigned from a list of Indian states
  - transaction_date   : the real 48-hour Time column stretched across a
                          synthetic 6-month date range, preserving each
                          transaction's relative order

These are NOT real patterns from the data - they are included so the SQL
queries and dashboard can demonstrate the analysis pattern. This is stated
explicitly here and should be stated in the README/dashboard too, so nobody
mistakes simulated geography/category for a genuine finding.
"""

import sqlite3
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

MERCHANT_CATEGORIES = [
    "Grocery", "Electronics", "Fuel/Petrol", "Online Shopping", "Dining",
    "Travel", "Utility Bills", "ATM Withdrawal", "Jewelry", "Pharmacy",
]
# Simulated category risk weights - NOT derived from the data, just used to
# make the simulated assignment look plausible (e.g. jewelry/electronics
# tend to be higher fraud-risk categories in real-world fraud literature)
CATEGORY_FRAUD_WEIGHT = {
    "Grocery": 0.5, "Electronics": 2.5, "Fuel/Petrol": 0.5, "Online Shopping": 2.0,
    "Dining": 0.7, "Travel": 1.8, "Utility Bills": 0.3, "ATM Withdrawal": 1.5,
    "Jewelry": 3.0, "Pharmacy": 0.4,
}

STATES = [
    "Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Madhya Pradesh",
    "Gujarat", "West Bengal", "Telangana", "Uttar Pradesh", "Rajasthan",
]


def add_simulated_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # Merchant category: fraud rows are weighted toward higher-risk categories
    # (simulated effect, not a real pattern) so the SQL query has something
    # meaningful to surface, rather than pure uniform noise.
    weights = np.array([CATEGORY_FRAUD_WEIGHT[c] for c in MERCHANT_CATEGORIES])
    fraud_probs = weights / weights.sum()
    uniform_probs = np.ones(len(MERCHANT_CATEGORIES)) / len(MERCHANT_CATEGORIES)

    categories = []
    for is_fraud in df["Class"]:
        p = fraud_probs if is_fraud == 1 else uniform_probs
        categories.append(np.random.choice(MERCHANT_CATEGORIES, p=p))
    df["merchant_category"] = categories

    # Geography: uniformly random, purely illustrative
    df["state"] = np.random.choice(STATES, size=n)

    # Synthetic date range: stretch the real 48-hour Time column across 6
    # months while preserving relative transaction order.
    start_date = pd.Timestamp("2025-01-01")
    end_date = pd.Timestamp("2025-06-30")
    time_fraction = (df["Time"] - df["Time"].min()) / (df["Time"].max() - df["Time"].min())
    total_seconds = (end_date - start_date).total_seconds()
    df["transaction_date"] = start_date + pd.to_timedelta(time_fraction * total_seconds, unit="s")
    df["transaction_month"] = df["transaction_date"].dt.to_period("M").astype(str)

    return df


def build_amount_band(amount: float) -> str:
    if amount < 10:
        return "0-10"
    elif amount < 50:
        return "10-50"
    elif amount < 100:
        return "50-100"
    elif amount < 500:
        return "100-500"
    elif amount < 1000:
        return "500-1000"
    else:
        return "1000+"


def main():
    df = pd.read_csv("../data/processed/creditcard_features.csv")
    df = add_simulated_dimensions(df)
    df["amount_band"] = df["Amount"].apply(build_amount_band)

    # Keep the table focused - drop raw PCA columns from the SQL table since
    # SQL analytics here is about business dimensions, not model features
    sql_columns = [
        "Time", "Amount", "Class", "hour_of_day", "day_of_week", "is_night",
        "merchant_category", "state", "transaction_date", "transaction_month",
        "amount_band",
    ]
    df_sql = df[sql_columns].copy()
    df_sql["transaction_date"] = df_sql["transaction_date"].astype(str)

    conn = sqlite3.connect("../data/processed/fraud_analytics.db")
    df_sql.to_sql("transactions", conn, if_exists="replace", index_label="transaction_id")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_class ON transactions(Class)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_month ON transactions(transaction_month)")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    fraud_count = conn.execute("SELECT COUNT(*) FROM transactions WHERE Class=1").fetchone()[0]
    print(f"Loaded {count} transactions ({fraud_count} fraud) into "
          f"../data/processed/fraud_analytics.db, table 'transactions'")
    print("NOTE: merchant_category, state, and transaction_date/month are "
          "SIMULATED - the raw dataset has no such columns. See docstring.")
    conn.close()


if __name__ == "__main__":
    main()