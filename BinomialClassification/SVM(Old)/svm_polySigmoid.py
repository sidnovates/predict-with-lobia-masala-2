##Dataset taken 20% as taking time to train SVM on full dataset

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# ===============================
# PATHS
# ===============================
train_path = r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\train_updated.csv"
test_path  = r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\test_updated.csv"

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


# ============================================================
# NEW SPLIT: 20% TRAIN, 40% VALIDATION, 40% INTERNAL TEST
# ============================================================

# First split: TRAIN 20% + TEMP 80%
train_set, temp_set = train_test_split(
    train_df,
    test_size=0.80,                 # leaving 20% for training
    random_state=42,
    stratify=train_df[TARGET]
)

# Now split TEMP into 40% VAL and 40% TEST
# Since temp_set = 80%, we need to split that into half

val_set, test_internal = train_test_split(
    temp_set,
    test_size=0.50,                 # 0.50 * 80% = 40% of total
    random_state=42,
    stratify=temp_set[TARGET]
)

# Shapes
print("Train size:", len(train_set))
print("Validation size:", len(val_set))
print("Internal Test size:", len(test_internal))

# Prepare X and y
X_train = train_set.drop([TARGET, ID_COL], axis=1)
y_train = train_set[TARGET]

X_val = val_set.drop([TARGET, ID_COL], axis=1)
y_val = val_set[TARGET]

X_test_internal = test_internal.drop([TARGET, ID_COL], axis=1)
y_test_internal = test_internal[TARGET]


# ===============================
# COMMON PREPROCESSOR
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# ============================================================
# =============== HELPER FUNCTION FOR ALL MODELS ===============
# ============================================================
def run_and_save(model_name, model):
    OUTPUT_DIR = model_name
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # ===============================
    # Validation
    # ===============================
    val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    with open(os.path.join(OUTPUT_DIR, "validation_report.txt"), "w") as f:
        f.write(classification_report(y_val, val_pred))
        f.write(f"\nValidation Accuracy: {val_acc}\n")

    # ===============================
    # Internal Test
    # ===============================
    test_pred = pipeline.predict(X_test_internal)
    test_acc = accuracy_score(y_test_internal, test_pred)

    with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w") as f:
        f.write(classification_report(y_test_internal, test_pred))
        f.write(f"\nTest Accuracy: {test_acc}\n")

    # ===============================
    # Accuracy Summary
    # ===============================
    with open(os.path.join(OUTPUT_DIR, "accuracy_summary.txt"), "w") as f:
        f.write(f"Validation Accuracy: {val_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")

    # ===============================
    # Kaggle Submission File
    # ===============================
    X_final_test = test_df.drop(ID_COL, axis=1)
    final_pred = pipeline.predict(X_final_test)

    submission = pd.DataFrame({
        "ProfileID": test_df[ID_COL],
        "RiskFlag": final_pred.astype(int)
    })

    submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)

    print(f"{model_name} Completed!")


# ============================================================
# ================== MODEL 1: POLYNOMIAL SVM ==================
# ============================================================
poly_model = SVC(
    kernel="poly",
    degree=3,
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    cache_size=500
)
run_and_save("svm_poly", poly_model)


# ============================================================
# ================== MODEL 2: SIGMOID SVM =====================
# ============================================================
sigmoid_model = SVC(
    kernel="sigmoid",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    cache_size=500
)
run_and_save("svm_sigmoid", sigmoid_model)


