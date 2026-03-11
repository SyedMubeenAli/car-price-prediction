import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# Fix Windows console encoding for emoji/unicode characters
try:
    if hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Configuration ────────────────────────────────────────────────────────────
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

# Try data directory first, then root directory
DATASET_PATH = os.path.join(current_dir, "..", "data", "car_data.csv")
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join(current_dir, "..", "car_data.csv")

PLOTS_DIR = os.path.join(current_dir, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Styling
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

CURRENT_YEAR = 2026


# =============================================================================
# STEP 1 — Load & Explore Dataset
# =============================================================================
print("=" * 70)
print("  STEP 1: Dataset Overview")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print(f"\n📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
print("── First 5 Rows ──")
print(df.head().to_string())
print("\n── Data Types ──")
print(df.dtypes)
print("\n── Statistical Summary (Numerical) ──")
print(df.describe().to_string())
print("\n── Statistical Summary (Categorical) ──")
print(df.describe(include="object").to_string())


# =============================================================================
# STEP 2 — Missing Values Check
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 2: Missing Values")
print("=" * 70)

missing = df.isnull().sum()
print(f"\n{'Column':<20} {'Missing':>8} {'Percent':>10}")
print("-" * 40)
for col in df.columns:
    pct = 100 * missing[col] / len(df)
    marker = " ✅" if missing[col] == 0 else " ⚠️"
    print(f"{col:<20} {missing[col]:>8} {pct:>9.1f}%{marker}")

print(f"\n✅ Total missing values: {missing.sum()}")


# =============================================================================
# STEP 3 — Feature Engineering for Analysis
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 3: Feature Engineering — Car Age")
print("=" * 70)

df["Car_Age"] = CURRENT_YEAR - df["Year"]
print(f"\n  Created 'Car_Age' = {CURRENT_YEAR} - Year")
print(f"  Car_Age range: {df['Car_Age'].min()} – {df['Car_Age'].max()} years")


# =============================================================================
# STEP 4 — Distribution Plots
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 4: Distribution Plots")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Distribution of Key Numerical Features", fontsize=16, fontweight="bold")

# Selling_Price
sns.histplot(df["Selling_Price"], kde=True, bins=30, color="#2ecc71", ax=axes[0, 0])
axes[0, 0].set_title("Selling Price (Target)")
axes[0, 0].set_xlabel("Selling Price (Lakhs)")

# Present_Price
sns.histplot(df["Present_Price"], kde=True, bins=30, color="#3498db", ax=axes[0, 1])
axes[0, 1].set_title("Present Price")
axes[0, 1].set_xlabel("Present Price (Lakhs)")

# Kms_Driven
sns.histplot(df["Kms_Driven"], kde=True, bins=30, color="#e74c3c", ax=axes[1, 0])
axes[1, 0].set_title("Kilometres Driven")
axes[1, 0].set_xlabel("Kms Driven")

# Car_Age
sns.histplot(df["Car_Age"], kde=True, bins=20, color="#9b59b6", ax=axes[1, 1])
axes[1, 1].set_title("Car Age")
axes[1, 1].set_xlabel("Age (Years)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(PLOTS_DIR, "distributions.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  → Saved: plots/distributions.png")

# Insight
print("""
  📌 Insights:
   • Selling_Price is right-skewed — most cars sell below ₹10L, a few outliers above pkr30L.
   • Present_Price has a similar skew — dominated by affordable vehicles.
   • Kms_Driven is concentrated under 80K km, with a few high-mileage outliers.
   • Most vehicles are 5–15 years old (ages 5–15 given current year).
""")


# =============================================================================
# STEP 5 — Correlation Heatmap
# =============================================================================
print("=" * 70)
print("  STEP 5: Correlation Heatmap")
print("=" * 70)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numerical_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.5,
    square=True, ax=ax,
    cbar_kws={"shrink": 0.8}
)
ax.set_title("Correlation Heatmap (Numerical Features)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  → Saved: plots/correlation_heatmap.png")

print(f"\n  Correlation with Selling_Price:")
sell_corr = corr_matrix["Selling_Price"].drop("Selling_Price").sort_values(ascending=False)
for feature, value in sell_corr.items():
    bar = "█" * int(abs(value) * 20)
    sign = "+" if value > 0 else "-"
    print(f"    {feature:<16} {sign}{abs(value):.3f}  {bar}")

print("""
  📌 Insights:
   • Present_Price has the highest positive correlation with Selling_Price.
   • Year (and inversely Car_Age) strongly affects price — newer cars sell for more.
   • Kms_Driven has a weak negative correlation — mileage matters less than expected.
""")


# =============================================================================
# STEP 6 — Categorical Feature Analysis
# =============================================================================
print("=" * 70)
print("  STEP 6: Categorical Feature Analysis")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Selling Price by Categorical Features", fontsize=16, fontweight="bold")

# By Fuel_Type
sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price", palette="Set2", ax=axes[0])
axes[0].set_title("By Fuel Type")
axes[0].set_ylabel("Selling Price (Lakhs)")

# By Seller_Type
sns.boxplot(data=df, x="Seller_Type", y="Selling_Price", palette="Set2", ax=axes[1])
axes[1].set_title("By Seller Type")
axes[1].set_ylabel("")

# By Transmission
sns.boxplot(data=df, x="Transmission", y="Selling_Price", palette="Set2", ax=axes[2])
axes[2].set_title("By Transmission")
axes[2].set_ylabel("")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(PLOTS_DIR, "categorical_boxplots.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  → Saved: plots/categorical_boxplots.png")

# Value counts
for col in ["Fuel_Type", "Seller_Type", "Transmission"]:
    print(f"\n  {col} distribution:")
    for val, count in df[col].value_counts().items():
        print(f"    {val:<12} {count:>4}  ({100 * count / len(df):.1f}%)")

print("""
  📌 Insights:
   • Diesel vehicles tend to have higher selling prices than Petrol.
   • Dealer-sold vehicles have higher prices than Individual sellers (bikes
     are mostly sold by Individuals at lower prices).
   • Automatic transmission vehicles fetch higher prices on average.
""")


# =============================================================================
# STEP 7 — Selling Price vs Present Price (Scatter)
# =============================================================================
print("=" * 70)
print("  STEP 7: Selling Price vs Present Price")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    df["Present_Price"], df["Selling_Price"],
    c=df["Car_Age"], cmap="plasma", alpha=0.7, s=60, edgecolors="white", linewidth=0.5
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Car Age (Years)")
ax.set_xlabel("Present Price (Lakhs)")
ax.set_ylabel("Selling Price (Lakhs)")
ax.set_title("Selling Price vs Present Price (colored by Car Age)", fontsize=14, fontweight="bold")
ax.plot([0, df["Present_Price"].max()], [0, df["Present_Price"].max()],
        "r--", alpha=0.5, label="Ideal (no depreciation)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "price_scatter.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  → Saved: plots/price_scatter.png")

print("""
  📌 Insights:
   • Strong linear relationship — higher present price → higher selling price.
   • All points fall below the red diagonal (cars depreciate).
   • Lighter dots (newer cars) are closer to the diagonal — less depreciation.
""")


# =============================================================================
# STEP 8 — Feature Importance (Random Forest)
# =============================================================================
print("=" * 70)
print("  STEP 8: Feature Importance (Random Forest)")
print("=" * 70)

# Prepare data for feature importance
df_fi = df.copy()
df_fi = df_fi.drop(columns=["Car_Name"])
df_fi["Fuel_Type"] = df_fi["Fuel_Type"].map({"Petrol": 0, "Diesel": 1, "CNG": 2})
df_fi["Seller_Type"] = df_fi["Seller_Type"].map({"Dealer": 0, "Individual": 1})
df_fi["Transmission"] = df_fi["Transmission"].map({"Manual": 0, "Automatic": 1})

X_fi = df_fi.drop(columns=["Selling_Price"])
y_fi = df_fi["Selling_Price"]

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_fi, y_fi)

importances = pd.Series(rf.feature_importances_, index=X_fi.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
importances.plot(kind="barh", color="#2ecc71", edgecolor="white", ax=ax)
ax.set_title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score")
for i, (val, name) in enumerate(zip(importances.values, importances.index)):
    ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.show()
print("  → Saved: plots/feature_importance.png")

print("\n  Feature Importance Ranking:")
for rank, (feat, imp) in enumerate(importances[::-1].items(), 1):
    bar = "█" * int(imp * 40)
    print(f"    {rank}. {feat:<16} {imp:.4f}  {bar}")

print("""
  📌 Insights:
   • Present_Price is the #1 most important feature — makes sense as the 
     showroom price anchors resale value.
   • Car_Age / Year is second — depreciation is a major factor.
   • Kms_Driven and categoricals contribute less but are still informative.
""")

print("\n" + "=" * 70)
print("  ✅  EDA COMPLETE — All plots saved to notebooks/plots/")
print("=" * 70)
