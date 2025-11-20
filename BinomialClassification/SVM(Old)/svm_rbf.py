import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ===============================
# PATHS
# ===============================
train_path = r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\train_updated.csv"
test_path  = r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\test_updated.csv"

OUTPUT_DIR = "svm_rbf"
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
# SPLIT 20 / 40 / 40
# ===============================
# 20% train, 80% remaining
train_set, remaining = train_test_split(
    train_df,
    test_size=0.80,
    random_state=42,
    stratify=train_df[TARGET]
)

# Split remaining 80% into: 40% val, 40% test
# val = 0.40 total → 0.40 / 0.80 = 0.5 of remaining
val_set, test_internal = train_test_split(
    remaining,
    test_size=0.50,
    random_state=42,
    stratify=remaining[TARGET]
)

X_train = train_set.drop([TARGET, ID_COL], axis=1)
y_train = train_set[TARGET]

X_val = val_set.drop([TARGET, ID_COL], axis=1)
y_val = val_set[TARGET]

X_test_internal = test_internal.drop([TARGET, ID_COL], axis=1)
y_test_internal = test_internal[TARGET]

print(f"Train set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Internal Test set size: {X_test_internal.shape[0]} samples")
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
# RBF KERNEL SVM MODEL
# ===============================
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clf", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            cache_size=500
        ))
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
# =================--------------
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
# FINAL TEST SET PREDICTIONS (KAGGLE SUBMISSION)
# =================--------------
X_final_test = test_df.drop(ID_COL, axis=1)
final_pred = model.predict(X_final_test)

submission = pd.DataFrame({
    "ProfileID": test_df[ID_COL],
    "RiskFlag": final_pred.astype(int)
})

submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)

print("RBF SVM completed successfully!")
