# ============================
#   EDA + OUTLIER ANALYSIS
#   MULTICLASS SPEND CATEGORY
# ============================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# Load Training Data
# ---------------------------------------------------------------------
train_path = "/content/drive/MyDrive/ML_Project/MultinomialClassification/Dataset/train.csv"   # <-- CHANGE PATH
train_df = pd.read_csv(train_path)

print("Train shape:", train_df.shape)
print(train_df.head())

# ---------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------
TARGET = "spend_category"

# ---------------------------------------------------------------------
# Numeric columns (from dataset description)
# ---------------------------------------------------------------------
numeric_cols = [
    "num_females",
    "num_males",
    "mainland_stay_nights",
    "island_stay_nights",
    # # after cleaning / mapping:
    # "total_trip_days_numeric",
    # "days_booked_before_trip_numeric"
]

print("\nNumeric Columns:", numeric_cols)

# ==========================================================
# 1. Summary Statistics
# ==========================================================
print("\n=== SUMMARY STATISTICS ===")
print(train_df[numeric_cols].describe())

# ==========================================================
# 2. Boxplots → Outlier Detection
# ==========================================================
plt.figure(figsize=(16, 8))
train_df[numeric_cols].boxplot()
plt.title("Boxplots of Numeric Features (Outlier Detection)")
plt.xticks(rotation=45)
plt.show()

# Individual boxplots
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=train_df[col])
    plt.title(f"Boxplot: {col}")
    plt.show()

# ==========================================================
# 3. Scatter Plots vs Target
# ==========================================================
for col in numeric_cols:
    plt.figure(figsize=(7, 4))
    sns.scatterplot(
        x=train_df[col],
        y=train_df[TARGET],
        hue=train_df[TARGET],
        palette="Set1"
    )
    plt.title(f"{col} vs {TARGET}")
    plt.show()

# ==========================================================
# 4. Pairplot (numeric + target)
# ==========================================================
sns.pairplot(train_df[numeric_cols + [TARGET]], hue=TARGET)
plt.show()

# ==========================================================
# 5. CORRELATION MATRIX (Pearson)
# ==========================================================
corr_matrix = train_df[numeric_cols].corr()

print("\n=== CORRELATION MATRIX ===")
print(corr_matrix)

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ==========================================================
# 6. COVARIANCE MATRIX (Requested)
# ==========================================================
cov_matrix = train_df[numeric_cols].cov()

print("\n=== COVARIANCE MATRIX ===")
print(cov_matrix)

plt.figure(figsize=(10, 6))
sns.heatmap(cov_matrix, annot=True, cmap="viridis", fmt=".2f")
plt.title("Covariance Matrix Heatmap")
plt.show()

# ==========================================================
# 7. Outlier Detection Using IQR
# ==========================================================
def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return len(outliers), lower, upper

print("\n=== OUTLIERS USING IQR ===")
for col in numeric_cols:
    cnt, low, high = detect_outliers_iqr(train_df, col)
    print(f"{col}: {cnt} outliers found (Range: {low:.2f} to {high:.2f})")

print("\nEDA Completed Successfully!")
