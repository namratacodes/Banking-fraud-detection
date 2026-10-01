"""
Feature engineering for banking fraud detection.

Input: raw Kaggle creditcard.csv schema
  - Time   : seconds elapsed since the first transaction in the dataset
  - V1-V28 : PCA-transformed features (anonymized)
  - Amount : transaction amount
  - Class  : 1 = fraud, 0 = legitimate

Note: The public Kaggle dataset has no user/card ID column, so "per-user"
features are approximated using a PCA-fingerprint hash as a proxy for a
cardholder. If/when a real user_id column is available, swap out
`_infer_pseudo_user` for the real key.
"""

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw seconds-elapsed `Time` into hour-of-day / day-of-week features."""
    df = df.copy()
    seconds_in_day = 24 * 60 * 60
    df["hour_of_day"] = (df["Time"] % seconds_in_day) // 3600
    df["day_of_week"] = (df["Time"] // seconds_in_day) % 7
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if (h < 6 or h >= 22) else 0)
    return df


def _infer_pseudo_user(df: pd.DataFrame, n_buckets: int = 200) -> pd.Series:
    """
    The Kaggle dataset has no cardholder ID. To still demonstrate per-user
    velocity/deviation features (a core ask in real fraud systems), we bucket
    transactions by a stable hash of their PCA fingerprint (V1-V5 rounded) as
    a pseudo-user proxy. This is clearly labeled as a proxy, not ground truth.
    """
    fingerprint = df[["V1", "V2", "V3", "V4", "V5"]].round(1).astype(str).agg("_".join, axis=1)
    return fingerprint.apply(lambda s: hash(s) % n_buckets)


def add_velocity_features(df: pd.DataFrame, window_seconds: int = 3600) -> pd.DataFrame:
    """
    Transaction velocity: how many transactions has this pseudo-user made
    in the trailing `window_seconds` (default 1 hour)?
    """
    df = df.copy()
    df["pseudo_user"] = _infer_pseudo_user(df)
    df = df.sort_values(["pseudo_user", "Time"]).reset_index(drop=True)

    velocities = np.zeros(len(df), dtype=int)
    for user, group in df.groupby("pseudo_user"):
        times = group["Time"].values
        idx = group.index.values
        for i, t in enumerate(times):
            window_start = t - window_seconds
            count = np.sum((times[:i] >= window_start))
            velocities[idx[i]] = count

    df["txn_velocity_1h"] = velocities
    return df


def add_amount_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    How much does this transaction's amount deviate from this pseudo-user's
    historical average and rolling average?
    """
    df = df.copy()
    if "pseudo_user" not in df.columns:
        df["pseudo_user"] = _infer_pseudo_user(df)

    df = df.sort_values(["pseudo_user", "Time"]).reset_index(drop=True)

    user_mean = df.groupby("pseudo_user")["Amount"].transform("mean")
    user_std = df.groupby("pseudo_user")["Amount"].transform("std").fillna(0)

    df["amount_dev_from_user_mean"] = (df["Amount"] - user_mean) / user_mean.replace(0, np.nan)
    df["amount_dev_from_user_mean"] = df["amount_dev_from_user_mean"].fillna(0)

    df["amount_zscore_user"] = np.where(
        user_std > 0, (df["Amount"] - user_mean) / user_std, 0
    )

    df["rolling_avg_amount_5"] = (
        df.groupby("pseudo_user")["Amount"]
        .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
    )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full Phase 2 feature engineering pipeline."""
    df = add_time_features(df)
    df = add_velocity_features(df)
    df = add_amount_deviation_features(df)
    return df


if __name__ == "__main__":
    raw_path = "../data/raw/creditcard.csv"
    out_path = "../data/processed/creditcard_features.csv"
    df = pd.read_csv(raw_path)
    df_feat = build_features(df)
    df_feat.to_csv(out_path, index=False)
    print(f"Wrote {len(df_feat)} rows with {df_feat.shape[1]} columns to {out_path}")