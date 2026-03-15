"""
eda.py  — Exploratory Data Analysis
------
Adapted from your original eda.py to work with bank_fraud_dataset.csv
and the FEATURES defined in features.py.

Outputs saved to results/figures/
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — safe in any environment
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH   = "data/bank_fraud_dataset.csv"
FIGURE_PATH = "results/figures"
os.makedirs(FIGURE_PATH, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df["fraud_bool"].value_counts())
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

# ── 1. Class distribution ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["fraud_bool"].value_counts()
ax.bar(["Legit", "Fraud"], counts.values,
       color=["#22c55e", "#ef4444"], edgecolor="none", width=0.5)
ax.set_title("Class Distribution")
ax.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, f"{v:,}\n({v/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/class_distribution.png", dpi=150)
plt.close()
print("\n[1/6] class_distribution.png saved")

# ── 2. Fraud probability by payment type ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
pt_rate = df.groupby("payment_type")["fraud_bool"].mean().sort_values(ascending=False)
ax.barh(pt_rate.index, pt_rate.values * 100, color="#ef4444", alpha=0.8)
ax.set_xlabel("Fraud rate (%)")
ax.set_title("Fraud rate by payment type")
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/fraud_by_payment_type.png", dpi=150)
plt.close()
print("[2/6] fraud_by_payment_type.png saved")

# ── 3. Velocity distributions ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, col in zip(axes, ["velocity_6h", "velocity_24h", "velocity_4w"]):
    for label, color in [(0, "#22c55e"), (1, "#ef4444")]:
        subset = df[df["fraud_bool"] == label][col]
        ax.hist(subset, bins=40, alpha=0.6, color=color,
                label="Legit" if label == 0 else "Fraud", density=True)
    ax.set_title(col)
    ax.legend(fontsize=8)
plt.suptitle("Velocity distributions — Fraud vs Legit", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/velocity_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("[3/6] velocity_distributions.png saved")

# ── 4. Credit risk score distribution ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for label, color, name in [(0, "#22c55e", "Legit"), (1, "#ef4444", "Fraud")]:
    ax.hist(df[df["fraud_bool"] == label]["credit_risk_score"],
            bins=50, alpha=0.6, color=color, label=name, density=True)
ax.set_title("Credit risk score distribution")
ax.set_xlabel("Credit risk score")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/credit_risk_score.png", dpi=150)
plt.close()
print("[4/6] credit_risk_score.png saved")

# ── 5. Hour-of-day fraud heatmap ─────────────────────────────────────────────
hourly = (df.groupby("transaction_hour")["fraud_bool"]
            .mean()
            .reset_index()
            .rename(columns={"fraud_bool": "fraud_rate"}))

fig, ax = plt.subplots(figsize=(10, 3))
bars = ax.bar(hourly["transaction_hour"], hourly["fraud_rate"] * 100,
              color=plt.cm.Reds(hourly["fraud_rate"] / hourly["fraud_rate"].max()))
ax.set_xlabel("Hour of day (0–23)")
ax.set_ylabel("Fraud rate (%)")
ax.set_title("Fraud rate by hour of day")
ax.axvspan(-0.5, 5.5, alpha=0.08, color="blue", label="Night (0–5)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/fraud_by_hour.png", dpi=150)
plt.close()
print("[5/6] fraud_by_hour.png saved")

# ── 6. Correlation heatmap (numeric features only) ────────────────────────────
from features import FEATURES
numeric_feats = [f for f in FEATURES
                 if f not in ("payment_type", "device_os")
                 and f in df.columns]
corr = df[numeric_feats + ["fraud_bool"]].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = corr.isnull()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=.3, ax=ax, mask=mask,
            annot_kws={"size": 7})
ax.set_title("Feature correlation (including fraud_bool)")
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/correlation_heatmap.png", dpi=150)
plt.close()
print("[6/6] correlation_heatmap.png saved")

print(f"\n✅ All figures saved to {FIGURE_PATH}/")