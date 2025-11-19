## SVM with Linear with No Balanced Dataset Considered

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# ===============================
# PATHS
# ===============================
train_path = r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\train_updated.csv"
test_path  = r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\test_updated.csv"

OUTPUT_DIR = "svm_linear_"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

TARGET = "RiskFlag"
ID_COL = "ProfileID"

numeric_features = [
    "ApplicantYears","AnnualEarnings","RequestedSum","TrustMetric","WorkDuration",
    "ActiveAccounts","OfferRate","RepayPeriod","DebtFactor"
]

categorical_features = [
    "QualificationLevel","WorkCategory","RelationshipStatus","FamilyObligation",
    "OwnsProperty","FundUseCase","JointApplicant"
]

# ===============================
# SPLIT 80 / 10 / 10
# ===============================
train_full, test_internal = train_test_split(
    train_df, test_size=0.10, random_state=42, stratify=train_df[TARGET]
)

train_set, val_set = train_test_split(
    train_full, test_size=0.1111, random_state=42, stratify=train_full[TARGET]
)

X_train = train_set.drop([TARGET, ID_COL], axis=1)
y_train = train_set[TARGET]

X_val = val_set.drop([TARGET, ID_COL], axis=1)
y_val = val_set[TARGET]

X_test_internal = test_internal.drop([TARGET, ID_COL], axis=1)
y_test_internal = test_internal[TARGET]

# ===============================
# PREPROCESSOR
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ===============================
# LINEAR SVM MODEL
# ===============================
# LinearSVC does not support probability; it's fast and handles large datasets
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clf", LinearSVC(max_iter=5000))  # VERY IMPORTANT
    ]
)

# ===============================
# TRAIN
# ===============================
model.fit(X_train, y_train)

# ===============================
# VALIDATION PERFORMANCE
# ===============================
val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)

with open(os.path.join(OUTPUT_DIR, "validation_report.txt"), "w") as f:
    f.write(classification_report(y_val, val_pred))
    f.write(f"\nValidation Accuracy: {val_acc}\n")

# ===============================
# INTERNAL TEST PERFORMANCE
# ===============================
test_pred = model.predict(X_test_internal)
test_acc = accuracy_score(y_test_internal, test_pred)

with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w") as f:
    f.write(classification_report(y_test_internal, test_pred))
    f.write(f"\nTest Accuracy: {test_acc}\n")

# ===============================
# ACCURACY SUMMARY
# ===============================
with open(os.path.join(OUTPUT_DIR, "accuracy_summary.txt"), "w") as f:
    f.write(f"Validation Accuracy: {val_acc}\n")
    f.write(f"Test Accuracy: {test_acc}\n")

# ===============================
# FINAL TEST SET PREDICTIONS (KAGGLE FORMAT)
# ===============================
X_final_test = test_df.drop(ID_COL, axis=1)
final_pred = model.predict(X_final_test)

submission = pd.DataFrame({
    "ProfileID": test_df[ID_COL],
    "RiskFlag": final_pred.astype(int)
})

submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)

print("Linear SVM completed successfully!")
