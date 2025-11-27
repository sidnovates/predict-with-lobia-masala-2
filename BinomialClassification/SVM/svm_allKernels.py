
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


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

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)



def run_svm_kernel(model_dir, X_train, y_train, X_val, y_val, 
                   X_test_internal, y_test_internal, class_weight,
                   final_test_data, svm_kernel):

    os.makedirs(model_dir, exist_ok=True)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", SVC(
                kernel=svm_kernel,
                C=1.0,
                gamma="scale",
                class_weight=class_weight,
                probability=False
            ))
        ]
    )

    print(f"\nTraining SVM ({svm_kernel}) → {model_dir}")
    model.fit(X_train, y_train)

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

    # ===== SUMMARY =====
    with open(os.path.join(model_dir, "accuracy_summary.txt"), "w") as f:
        f.write(f"Validation Accuracy: {val_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")

    # ===== FINAL SUBMISSION =====
    final_pred = model.predict(final_test_data)

    submission = pd.DataFrame({
        "ProfileID": test_df[ID_COL],
        "RiskFlag": final_pred.astype(int)
    })

    csv_name = os.path.basename(model_dir) + f"_SVM_{svm_kernel}.csv"
    submission.to_csv(os.path.join(model_dir, csv_name), index=False)

    print(f"Completed → {model_dir}")


def run_all_svm_kernels():

    BASE_DIR = "SVM_Kernels"
    os.makedirs(BASE_DIR, exist_ok=True)

    # ==========================================================
    # 80% SPLIT
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

    # ==========================================================
    # 20% SPLIT
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

    # ==========================================================
    # RUNNING ALL SVM KERNELS
    # ==========================================================

    kernels = ["rbf", "poly", "sigmoid"]

    modes = [
        ("80_skewed",    X_train_80, y_train_80, X_val_80, y_val_80, X_test_int_80, y_test_int_80, None),
        ("80_nonskewed", X_train_80, y_train_80, X_val_80, y_val_80, X_test_int_80, y_test_int_80, "balanced"),
        ("20_skewed",    X_train_20, y_train_20, X_val_20, y_val_20, X_test_int_20, y_test_int_20, None),
        ("20_nonskewed", X_train_20, y_train_20, X_val_20, y_val_20, X_test_int_20, y_test_int_20, "balanced")
    ]

    for kernel in kernels:
        print(f"\n==============================")
        print(f"     Running SVM → {kernel}")
        print(f"==============================\n")

        for mode_name, Xtr, Ytr, Xv, Yv, Xt_int, Yt_int, cw in modes:

            model_dir = os.path.join(BASE_DIR, f"{mode_name}_{kernel}")

            run_svm_kernel(
                model_dir=model_dir,
                X_train=Xtr, y_train=Ytr,
                X_val=Xv, y_val=Yv,
                X_test_internal=Xt_int, y_test_internal=Yt_int,
                class_weight=cw,
                final_test_data=X_final_test,
                svm_kernel=kernel
            )


# RUN EVERYTHING
run_all_svm_kernels()
print("All SVM Kernel Models Completed Successfully!")
