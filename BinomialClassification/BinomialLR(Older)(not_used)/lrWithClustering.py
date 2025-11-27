"""
===============================================================================
LOGISTIC REGRESSION + CLUSTERING FEATURE ENGINEERING
-------------------------------------------------------------------------------
This file trains THREE improved Logistic Regression models using:

    1. K-Means Clustering
    2. Hierarchical Clustering (Agglomerative)
    3. Gaussian Mixture Models (soft clusters)

Each clustering method creates NEW FEATURES extracted from numerical variables.
These features are added to the dataset to help Logistic Regression discover
non-linear patterns in the financial risk data.

OUTPUT:
Each model saves its files into separate folders:
    lr_kmeans/
    lr_hierarchical/
    lr_gmm/

Each folder contains:
    - validation_report.txt
    - test_report.txt
    - accuracy_summary.txt
    - submission.csv

===============================================================================
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Clustering models
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# ================================================================
# PATHS
# ================================================================
train_path = r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\train_updated.csv"
test_path  = r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\test_updated.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

TARGET = "RiskFlag"
ID_COL = "ProfileID"

# ================================================================
# NUMERIC FEATURES FOR CLUSTERING + NORMALIZATION
# (Only meaningful numeric columns used for cluster formation)
# ================================================================
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

# Categorical features (one-hot encoded before LR)
categorical_features = [
    "QualificationLevel",
    "WorkCategory",
    "RelationshipStatus",
    "FamilyObligation",
    "OwnsProperty",
    "FundUseCase",
    "JointApplicant",
]


# ================================================================
# SPLIT DATA (80 / 10 / 10)
# ================================================================
# First split: 80% train, 20% temp
train_set, temp_set = train_test_split(
    train_df, test_size=0.20, random_state=42, stratify=train_df[TARGET]
)

# Second split: temp → 10% val + 10% test
val_set, test_internal = train_test_split(
    temp_set, test_size=0.50, random_state=42, stratify=temp_set[TARGET]
)


print("Train samples:", len(train_set))
print("Validation samples:", len(val_set))
print("Internal Test samples:", len(test_internal))


# ================================================================
# STANDARDIZE NUMERIC FEATURES FOR CLUSTERING ONLY
# ================================================================
scaler = StandardScaler()

# X_full_numeric = scaler.fit_transform(train_set[numeric_features])
# Fit scaler ONLY on the training split (to avoid leakage)
scaler = StandardScaler()
X_train_numeric = scaler.fit_transform(train_set[numeric_features])

# ================================================================
# HELPER FUNCTION TO GENERATE AND SAVE ALL RESULTS
# ================================================================
def run_and_save(model_name, X_train_enh, X_val_enh, X_test_internal_enh, X_final_test_enh):
    """
    Runs Logistic Regression on enhanced datasets
    Saves validation/test reports and submission CSV into separate folders.
    """

    OUTPUT_DIR = model_name
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------
    # Preprocessor for LR
    # -----------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # -----------------------
    # Logistic Regression MODEL
    # -----------------------
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
        ]
    )

    # Train
    model.fit(X_train_enh, y_train) 

    # -----------------------
    # VALIDATION REPORT
    # -----------------------
    val_pred = model.predict(X_val_enh)
    val_acc = accuracy_score(y_val, val_pred)

    with open(os.path.join(OUTPUT_DIR, "validation_report.txt"), "w") as f:
        f.write(classification_report(y_val, val_pred))
        f.write(f"\nValidation Accuracy = {val_acc}\n")

    # -----------------------
    # INTERNAL TEST REPORT
    # -----------------------
    test_pred = model.predict(X_test_internal_enh)
    test_acc = accuracy_score(y_test_internal, test_pred)

    with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w") as f:
        f.write(classification_report(y_test_internal, test_pred))
        f.write(f"\nTest Accuracy = {test_acc}\n")

    # -----------------------
    # ACCURACY SUMMARY
    # -----------------------
    with open(os.path.join(OUTPUT_DIR, "accuracy_summary.txt"), "w") as f:
        f.write(f"Validation Accuracy: {val_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")

    # -----------------------
    # FINAL SUBMISSION
    # -----------------------
    final_pred = model.predict(X_final_test_enh)

    submission = pd.DataFrame({
        "ProfileID": test_df[ID_COL],
        "RiskFlag": final_pred.astype(int)
    })
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)

    print(f"{model_name} Completed!")


# ================================================================
# PREPARE TARGETS
# ================================================================
X_train = train_set.drop([TARGET, ID_COL], axis=1)
y_train = train_set[TARGET]

X_val = val_set.drop([TARGET, ID_COL], axis=1)
y_val = val_set[TARGET]

X_test_internal = test_internal.drop([TARGET, ID_COL], axis=1)
y_test_internal = test_internal[TARGET]

X_final_test = test_df.drop(ID_COL, axis=1)


# ================================================================
# =============== 1️⃣ K-MEANS FEATURE ENGINEERING =================
# ================================================================
print("\nRunning K-Means clustering...")

kmeans = KMeans(n_clusters=5, random_state=42)

# kmeans_labels_full = kmeans.fit_predict(X_full_numeric)
# # Assign cluster labels back properly
# train_set["kmeans_label"] = kmeans_labels_full

kmeans.fit(X_train_numeric)   # train only on 80%

# Map cluster labels to splits
def map_cluster(df):
    return kmeans.predict(scaler.transform(df[numeric_features]))

train_set["kmeans_label"] = map_cluster(train_set)
val_set["kmeans_label"] = map_cluster(val_set)
test_internal["kmeans_label"] = map_cluster(test_internal)
test_df["kmeans_label"] = map_cluster(test_df)

# Train model
run_and_save(
    "lr_kmeans",
    train_set.drop([TARGET, ID_COL], axis=1),
    val_set.drop([TARGET, ID_COL], axis=1),
    test_internal.drop([TARGET, ID_COL], axis=1),
    test_df.drop(ID_COL, axis=1)
)




# ================================================================
# =============== 3️⃣ GMM (Gaussian Mixture Model) ================
# ================================================================
print("\nRunning Gaussian Mixture Model...")

gmm = GaussianMixture(n_components=3, random_state=42)
# gmm.fit(X_full_numeric)
gmm.fit(X_train_numeric)

# Soft probabilities
train_set_gmm = gmm.predict_proba(scaler.transform(train_set[numeric_features]))
val_set_gmm = gmm.predict_proba(scaler.transform(val_set[numeric_features]))
test_internal_gmm = gmm.predict_proba(scaler.transform(test_internal[numeric_features]))
final_test_gmm = gmm.predict_proba(scaler.transform(test_df[numeric_features]))

# Add probability columns
for i in range(3):
    train_set[f"gmm_prob_{i}"] = train_set_gmm[:, i]
    val_set[f"gmm_prob_{i}"] = val_set_gmm[:, i]
    test_internal[f"gmm_prob_{i}"] = test_internal_gmm[:, i]
    test_df[f"gmm_prob_{i}"] = final_test_gmm[:, i]

run_and_save(
    "lr_gmm",
    train_set.drop([TARGET, ID_COL], axis=1),
    val_set.drop([TARGET, ID_COL], axis=1),
    test_internal.drop([TARGET, ID_COL], axis=1),
    test_df.drop(ID_COL, axis=1)
)

print("\nAll Logistic Regression + Clustering models completed successfully!")
