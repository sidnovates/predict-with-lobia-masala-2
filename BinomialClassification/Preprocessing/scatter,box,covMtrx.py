import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --------------------------
# Load dataset
# --------------------------
df = pd.read_csv(r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\train_updated.csv")

# --------------------------
# Define features
# --------------------------
numeric_features = [
    "ApplicantYears",
    "AnnualEarnings",
    "RequestedSum",
    "TrustMetric",
    "WorkDuration",
    "ActiveAccounts",
    "OfferRate",
    "RepayPeriod",
    "DebtFactor",
]

categorical_features = [
    "QualificationLevel",
    "WorkCategory",
    "RelationshipStatus",
    "FamilyObligation",
    "OwnsProperty",
    "FundUseCase",
    "JointApplicant",
]

TARGET = "RiskFlag"

# --------------------------
# Create folder for saving plots
# --------------------------
os.makedirs("eda_plots", exist_ok=True)

# --------------------------
# 1. Box Plots for Outliers
# --------------------------
for col in numeric_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color="skyblue")
    plt.title(f"Box Plot - {col}")
    plt.tight_layout()
    plt.savefig(f"eda_plots/boxplot_{col}.png")
    plt.close()

print("✔ Box plots saved.")

# --------------------------
# 2. Scatter Plots vs RiskFlag
# --------------------------
for col in numeric_features:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=df[col], y=df[TARGET], alpha=0.4)
    plt.title(f"Scatter Plot - {col} vs {TARGET}")
    plt.tight_layout()
    plt.savefig(f"eda_plots/scatter_{col}_vs_RiskFlag.png")
    plt.close()

print("✔ Scatter plots saved.")

# --------------------------
# 3. Correlation Matrix (only numeric features + RiskFlag)
# --------------------------
corr_df = df[numeric_features + [TARGET]].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("eda_plots/correlation_matrix.png")
plt.close()

print("✔ Correlation matrix saved.")

# --------------------------
# Save correlation matrix CSV if needed
# --------------------------
corr_df.to_csv("eda_plots/correlation_matrix.csv")

print("All EDA plots and correlation file saved in 'eda_plots/' folder.")
