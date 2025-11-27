## No preprocessing done

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
# -----------------------
# 1. Load datasets
# -----------------------
train_df = pd.read_csv(r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\train_updated.csv")
test_df = pd.read_csv(r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\test_updated.csv")
output_path= r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\BinomialLR\submission_logistic.csv"
# -----------------------
# 2. Separate features and target
# -----------------------
# Target + ID
TARGET = "RiskFlag"
ID_COL = "ProfileID"

# -----------------------------
# DEFINE NUMERIC & CATEGORICAL FEATURES
# -----------------------------
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

# -----------------------------
# SPLIT TRAINING DATA (80-10-10)
# -----------------------------
train_full, test_internal = train_test_split(
    train_df, test_size=0.10, random_state=42, stratify=train_df[TARGET]
)

train_set, val_set = train_test_split(
    train_full, test_size=0.1111, random_state=42, stratify=train_full[TARGET]
)
# 0.1111 * 0.90 = ~0.10 (so final split is 80-10-10)

print("Train size:", len(train_set))
print("Validation size:", len(val_set))
print("Internal Test size:", len(test_internal))

# -----------------------------
# PREPARE X, y
# -----------------------------
X_train = train_set.drop([TARGET, ID_COL], axis=1)
y_train = train_set[TARGET]

X_val = val_set.drop([TARGET, ID_COL], axis=1)
y_val = val_set[TARGET]

X_internal_test = test_internal.drop([TARGET, ID_COL], axis=1)
y_internal_test = test_internal[TARGET]

# -----------------------------
# PREPROCESSING PIPELINE
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# -----------------------------
# LOGISTIC REGRESSION MODEL
# -----------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=500))
    ]
)

# -----------------------------
# TRAIN THE MODEL
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# VALIDATION PERFORMANCE
# -----------------------------
val_pred = model.predict(X_val)
print("\nVALIDATION RESULTS:")
print("Accuracy:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))

# -----------------------------
# INTERNAL TEST PERFORMANCE
# -----------------------------
test_pred = model.predict(X_internal_test)
print("\nINTERNAL TEST RESULTS:")
print("Accuracy:", accuracy_score(y_internal_test, test_pred))
print(classification_report(y_internal_test, test_pred))

# -----------------------------
# PREPARE TESTING DATA FOR SUBMISSION
# -----------------------------
X_test_final = test_df.drop(ID_COL, axis=1)

test_predictions = model.predict(X_test_final)

submission = pd.DataFrame({
    "ProfileID": test_df[ID_COL],
    "RiskFlag": test_predictions.astype(int)
})


submission.to_csv(output_path, index=False)

print("Submission file saved as: submission_logistic.csv")
