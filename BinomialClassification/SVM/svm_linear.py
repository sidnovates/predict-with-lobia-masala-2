import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score


# ==========================================================
# PATHS
# ==========================================================
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

# ==========================================================
# PREPROCESSOR (COMMON)
# ==========================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# ==========================================================
# FUNCTION TO RUN AND SAVE MODEL
# ==========================================================
def run_svm_linear(model_dir, X_train, y_train, X_val, y_val, 
                   X_test_internal, y_test_internal, class_weight, final_test_data):

    os.makedirs(model_dir, exist_ok=True)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LinearSVC(C=1.0, class_weight=class_weight, max_iter=5000))
        ]
    )

    model.fit(X_train, y_train)

    import joblib
    # ===== SAVE TRAINED MODEL =====
    model_save_path = os.path.join(model_dir, "trained_model.joblib")
    joblib.dump(model, model_save_path)
    print(f"Model saved at: {model_save_path}")

    # ===== VALIDATION =====
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    with open(os.path.join(model_dir, "classification_validation.txt"), "w") as f:
        f.write(classification_report(y_val, val_pred))
        f.write(f"\nValidation Accuracy: {val_acc}\n")

    # ===== INTERNAL TEST =====
    test_pred = model.predict(X_test_internal)
    test_acc = accuracy_score(y_test_internal, test_pred)

    with open(os.path.join(model_dir, "classification_test.txt"), "w") as f:
        f.write(classification_report(y_test_internal, test_pred))
        f.write(f"\nTest Accuracy: {test_acc}\n")

    # ===== ACCURACY SUMMARY =====
    with open(os.path.join(model_dir, "accuracy_summary.txt"), "w") as f:
        f.write(f"Validation Accuracy: {val_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")

    # ===== FINAL SUBMISSION FILE =====
    final_pred = model.predict(final_test_data)

    submission = pd.DataFrame({
        "ProfileID": test_df[ID_COL],
        "RiskFlag": final_pred.astype(int)
    })

    csv_name = model_dir.split("\\")[-1] + "_SVM_Linear.csv"
    submission.to_csv(os.path.join(model_dir, csv_name), index=False)


# ==========================================================
# MAIN EXECUTION
# ==========================================================
def run_all_linear_svm():

    BASE_DIR = "SVM_Linear"
    os.makedirs(BASE_DIR, exist_ok=True)

    # ==========================================================
    # 80% DATASET
    # ==========================================================
    train_80, test_10_internal = train_test_split(
        train_df, test_size=0.10, random_state=42, stratify=train_df[TARGET]
    )

    train_80_main, val_10 = train_test_split(
        train_80, test_size=0.1111, random_state=42, stratify=train_80[TARGET]
    )

    X_train_80 = train_80_main.drop([TARGET, ID_COL], axis=1)
    y_train_80 = train_80_main[TARGET]

    X_val_80 = val_10.drop([TARGET, ID_COL], axis=1)
    y_val_80 = val_10[TARGET]

    X_test_int_80 = test_10_internal.drop([TARGET, ID_COL], axis=1)
    y_test_int_80 = test_10_internal[TARGET]

    X_final_test = test_df.drop(ID_COL, axis=1)

    # 80% SKEWED
    print("Running 80% Skewed Linear SVM...")
    run_svm_linear(
        model_dir=os.path.join(BASE_DIR, "80_skewed"),
        X_train=X_train_80, y_train=y_train_80,
        X_val=X_val_80, y_val=y_val_80,
        X_test_internal=X_test_int_80, y_test_internal=y_test_int_80,
        class_weight=None,
        final_test_data=X_final_test
    )
    print("80% Skewed Linear SVM Completed!\n")

    # 80% NON-SKEWED
    print("Running 80% Non-Skewed Linear SVM...")
    run_svm_linear(
        model_dir=os.path.join(BASE_DIR, "80_nonskewed"),
        X_train=X_train_80, y_train=y_train_80,
        X_val=X_val_80, y_val=y_val_80,
        X_test_internal=X_test_int_80, y_test_internal=y_test_int_80,
        class_weight="balanced",
        final_test_data=X_final_test
    )
    print("80% Non-Skewed Linear SVM Completed!\n")

    # ==========================================================
    # 20% DATASET
    # ==========================================================
    train_20, _ = train_test_split(
        train_df, train_size=0.20, random_state=42, stratify=train_df[TARGET]
    )

    train_20_main, val_20 = train_test_split(
        train_20, test_size=0.1111, random_state=42, stratify=train_20[TARGET]
    )

    X_train_20 = train_20_main.drop([TARGET, ID_COL], axis=1)
    y_train_20 = train_20_main[TARGET]

    X_val_20 = val_20.drop([TARGET, ID_COL], axis=1)
    y_val_20 = val_20[TARGET]

    X_test_int_20 = test_10_internal.drop([TARGET, ID_COL], axis=1)
    y_test_int_20 = test_10_internal[TARGET]

    # 20% SKEWED
    print("Running 20% Skewed Linear SVM...")
    run_svm_linear(
        model_dir=os.path.join(BASE_DIR, "20_skewed"),
        X_train=X_train_20, y_train=y_train_20,
        X_val=X_val_20, y_val=y_val_20,
        X_test_internal=X_test_int_20, y_test_internal=y_test_int_20,
        class_weight=None,
        final_test_data=X_final_test
    )
    print("20% Skewed Linear SVM Completed!\n")

    # 20% NON-SKEWED
    print("Running 20% Non-Skewed Linear SVM...")
    run_svm_linear(
        model_dir=os.path.join(BASE_DIR, "20_nonskewed"),
        X_train=X_train_20, y_train=y_train_20,
        X_val=X_val_20, y_val=y_val_20,
        X_test_internal=X_test_int_20, y_test_internal=y_test_int_20,
        class_weight="balanced",
        final_test_data=X_final_test
    )
    print("20% Non-Skewed Linear SVM Completed!\n")


# ==========================================================
# RUN EVERYTHING
# ==========================================================
run_all_linear_svm()
print("All Linear SVM Models Completed Successfully!")
