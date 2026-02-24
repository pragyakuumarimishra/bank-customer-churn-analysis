"""
Bank Customer Churn Analysis
----------------------------
Problem Statement: The bank is experiencing a high rate of customer attrition (churn). 
Objective: Analyze customer demographics and account information to identify key factors 
driving churn and provide actionable business recommendations to improve retention.

Note: This script generates a representative synthetic dataset based on the structure of 
publicly available bank churn datasets (like the Kaggle Churn Modelling dataset) to 
ensure the code is fully runnable out-of-the-box.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA PREPARATION (Simulating Public Data)
# ==========================================
print("--- Starting Bank Customer Churn Analysis ---\n")

np.random.seed(42)
n_samples = 2000

data = {
    "CustomerID": np.arange(1, n_samples + 1),
    "Age": np.random.randint(18, 75, n_samples),
    "Balance": np.random.uniform(0, 200000, n_samples),
    "NumOfProducts": np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.4, 0.08, 0.02]),
    "IsActiveMember": np.random.choice([0, 1], n_samples, p=[0.48, 0.52]),
}

df = pd.DataFrame(data)

# Base churn probability + simple churn logic
churn_prob = np.full(n_samples, 0.10)
churn_prob += np.where((df["Age"] > 45) & (df["Age"] < 60), 0.20, 0)
churn_prob += np.where(df["Balance"] == 0, 0.15, 0)
churn_prob += np.where(df["NumOfProducts"] >= 3, 0.40, 0)
churn_prob += np.where(df["IsActiveMember"] == 0, 0.15, -0.05)

churn_prob = np.clip(churn_prob, 0, 1)
df["Exited"] = np.random.binomial(1, churn_prob)

# Introduce dirty data (for cleaning step)
df.loc[10:20, "Age"] = np.nan
df = pd.concat([df, df.iloc[0:5]])  # duplicates

# ==========================================
# 2. DATA CLEANING
# ==========================================
print("--- Data Cleaning Phase ---")
print(f"Initial shape: {df.shape}")

missing_values = df.isnull().sum().sum()
print(f"Missing values found: {missing_values}")
df["Age"] = df["Age"].fillna(df["Age"].median())

duplicates = df.duplicated().sum()
print(f"Duplicates found: {duplicates}")
df = df.drop_duplicates()

df["Age"] = df["Age"].astype(int)
print(f"Cleaned shape: {df.shape}\n")

# ==========================================
# 3. EDA & VISUALS (AUTO-SAVE)
# ==========================================
print("--- Exploratory Data Analysis ---")

plt.style.use("bmh")

# Create screenshots folder
os.makedirs("screenshots", exist_ok=True)

# A) Overall churn (pie) -> screenshots/churn_overall.png
churn_counts = df["Exited"].value_counts()

fig1 = plt.figure(figsize=(7, 6))
plt.pie(
    churn_counts,
    labels=["Retained (0)", "Churned (1)"],
    autopct="%1.1f%%",
    colors=["#4CAF50", "#F44336"],
    startangle=90,
    explode=(0, 0.1),
)
plt.title("Overall Customer Churn")
plt.tight_layout()
fig1.savefig("screenshots/churn_overall.png", dpi=200)
print("Saved: screenshots/churn_overall.png")

print("Insight 1: Overall Churn")
print(f"The overall churn rate is {churn_counts.get(1, 0) / len(df) * 100:.1f}%.")
print("-" * 40)

# B) Churn by age -> screenshots/churn_by_age.png
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[17, 30, 40, 50, 60, 100],
    labels=["18-30", "31-40", "41-50", "51-60", "60+"],
)
age_churn = df.groupby("AgeGroup", observed=False)["Exited"].mean() * 100

fig2 = plt.figure(figsize=(8, 6))
age_churn.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Churn Rate by Age Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
fig2.savefig("screenshots/churn_by_age.png", dpi=200)
print("Saved: screenshots/churn_by_age.png")

print("Insight 2: Churn by Age")
print("Customers in the 41-60 age brackets show a significantly higher churn rate.")
print("-" * 40)

# C) Churn by number of products -> screenshots/churn_by_products.png
prod_churn = df.groupby("NumOfProducts")["Exited"].mean() * 100

fig3 = plt.figure(figsize=(8, 6))
prod_churn.plot(kind="bar", color="orange", edgecolor="black")
plt.title("Churn by Number of Products")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
fig3.savefig("screenshots/churn_by_products.png", dpi=200)
print("Saved: screenshots/churn_by_products.png")

print("Insight 3: Churn by Product Count")
print("Customers holding 3 or 4 products have an alarmingly high churn rate.")
print("-" * 40)

# Show all figures (optional)
plt.show()

# ==========================================
# 4. CONCLUSIONS & RECOMMENDATIONS
# ==========================================
print("\n=== FINAL CONCLUSIONS & RECOMMENDATIONS ===")
print("1. Target Middle-Aged Customers: Introduce loyalty programs for the 40-60 age demographic.")
print("2. Product Strategy: Investigate why customers with 3+ products are leaving; simplify bundles.")
print("3. Engagement Campaigns: Re-engage inactive members with email/SMS + small incentives.")
print("4. Balance Incentives: Offer better rates/support for high-balance tiers.")
