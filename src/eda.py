"""Phase 1: EDA for banking fraud detection dataset."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/raw/creditcard.csv")
plt.rcParams["figure.dpi"] = 110

# 1. Class imbalance
fraud_count = df["Class"].sum()
legit_count = len(df) - fraud_count
fraud_rate = 100 * df["Class"].mean()
print(f"Legit: {legit_count} | Fraud: {fraud_count} | Fraud rate: {fraud_rate:.4f}%")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.countplot(x="Class", data=df, ax=ax[0], palette=["#4C72B0", "#C44E52"])
ax[0].set_title("Class Distribution (0=Legit, 1=Fraud)")
ax[0].set_yscale("log")
ax[1].pie([legit_count, fraud_count], labels=["Legit", "Fraud"], autopct="%1.3f%%",
          colors=["#4C72B0", "#C44E52"])
ax[1].set_title("Fraud Share")
plt.tight_layout()
plt.savefig("../reports/class_imbalance.png")
plt.close()

# 2. Time analysis
df["hour"] = (df["Time"] % 86400) // 3600
fraud_by_hour = df[df["Class"] == 1].groupby("hour").size()
legit_by_hour = df[df["Class"] == 0].groupby("hour").size()
legit_rate_by_hour = (fraud_by_hour / (fraud_by_hour + legit_by_hour) * 100).fillna(0)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(fraud_by_hour.index, fraud_by_hour.values, color="#C44E52")
ax[0].set_title("Fraud Count by Hour of Day")
ax[0].set_xlabel("Hour")
ax[1].bar(legit_rate_by_hour.index, legit_rate_by_hour.values, color="#DD8452")
ax[1].set_title("Fraud Rate (%) by Hour of Day")
ax[1].set_xlabel("Hour")
plt.tight_layout()
plt.savefig("../reports/time_analysis.png")
plt.close()

# 3. Amount analysis
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(x="Class", y="Amount", data=df[df["Amount"] < 2000], ax=ax[0])
ax[0].set_title("Amount Distribution by Class (capped at 2000)")
sns.histplot(df[df["Class"] == 1]["Amount"], bins=40, ax=ax[1], color="#C44E52")
ax[1].set_title("Fraud Transaction Amount Distribution")
ax[1].set_xlim(0, 2000)
plt.tight_layout()
plt.savefig("../reports/amount_analysis.png")
plt.close()

print("Fraud amount stats:")
print(df[df["Class"] == 1]["Amount"].describe())
print("\nLegit amount stats:")
print(df[df["Class"] == 0]["Amount"].describe())

# 4. Correlation with target (V1-V28 + Amount)
corr = df.drop(columns=["hour"]).corr()["Class"].drop("Class").sort_values()
fig, ax = plt.subplots(figsize=(8, 9))
corr.plot(kind="barh", ax=ax, color=["#C44E52" if v < 0 else "#4C72B0" for v in corr.values])
ax.set_title("Feature Correlation with Fraud (Class)")
plt.tight_layout()
plt.savefig("../reports/feature_correlation.png")
plt.close()

print("\nTop 5 positively correlated features:")
print(corr.sort_values(ascending=False).head(5))
print("\nTop 5 negatively correlated features:")
print(corr.sort_values().head(5))

# 5. Naive accuracy trap illustration
print(f"\nNaive 'predict all legit' accuracy: {100*legit_count/len(df):.3f}%")
print("This is why accuracy is a useless metric for this problem.")

summary = {
    "total_transactions": int(len(df)),
    "fraud_count": int(fraud_count),
    "fraud_rate_pct": round(fraud_rate, 4),
    "naive_accuracy_pct": round(100 * legit_count / len(df), 3),
    "fraud_amount_mean": round(df[df["Class"] == 1]["Amount"].mean(), 2),
    "legit_amount_mean": round(df[df["Class"] == 0]["Amount"].mean(), 2),
    "top_positive_corr_feature": corr.sort_values(ascending=False).index[0],
    "top_negative_corr_feature": corr.sort_values().index[0],
}
import json
with open("../reports/eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved plots to reports/ and summary to reports/eda_summary.json")