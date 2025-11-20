import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

SEED = 42

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
# PREPROCESSOR
# ==========================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# ==========================================================
# 1. HYPERPARAMETER TUNING ON 80% NON-SKEWED
# ==========================================================
print("\n==============================")
print(" STEP 1 → Hyperparameter Tuning on 80% NON-SKEWED DATASET")
print("==============================\n")

train_full, test_internal = train_test_split(
    train_df, test_size=0.10, stratify=train_df[TARGET], random_state=SEED
)

train_80_main, val_10 = train_test_split(
    train_full, test_size=0.1111, stratify=train_full[TARGET], random_state=SEED
)

X_train_tune = train_80_main.drop([TARGET, ID_COL], axis=1)
y_train_tune = train_80_main[TARGET]

# Pipeline for tuning
pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=500))
    ]
)

param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs"],
    "clf__class_weight": ["balanced"]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train_tune, y_train_tune)

best_params = grid.best_params_
print("\nBest Hyperparameters Found:")
print(best_params)

print("\nHyperparameter tuning on 80% completed!\n")


# ==========================================================
# 2. FUNCTION TO TRAIN WITH BEST PARAMS
# ==========================================================
def run_lr_tuned(model_dir, X_train, y_train, X_val, y_val,
                 X_test_internal, y_test_internal, final_test):
    
    os.makedirs(model_dir, exist_ok=True)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(
                max_iter=500,
                C=best_params["clf__C"],
                penalty="l2",
                solver="lbfgs",
                class_weight=best_params["clf__class_weight"]
            ))
        ]
    )

    model.fit(X_train, y_train)

    # Validation
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    with open(os.path.join(model_dir, "classification_validation.txt"), "w") as f:
        f.write(classification_report(y_val, val_pred))
        f.write(f"\nValidation Accuracy: {val_acc}\n")

    # Internal Test
    test_pred = model.predict(X_test_internal)
    test_acc = accuracy_score(y_test_internal, test_pred)

    with open(os.path.join(model_dir, "classification_test.txt"), "w") as f:
        f.write(classification_report(y_test_internal, test_pred))
        f.write(f"\nTest Accuracy: {test_acc}\n")

    # Summary
    with open(os.path.join(model_dir, "accuracy_summary.txt"), "w") as f:
        f.write(f"Validation Accuracy: {val_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")

    # Final Kaggle Submission
    final_pred = model.predict(final_test)

    csvname = model_dir.split("\\")[-1] + "_Binomial_LR_Tuned.csv"
    df = pd.DataFrame({
        "ProfileID": test_df[ID_COL],
        "RiskFlag": final_pred.astype(int)
    })
    df.to_csv(os.path.join(model_dir, csvname), index=False)



# ==========================================================
# 3. GENERATE ALL 4 MODES (20/80 × skew/non-skew)
# ==========================================================
BASE_DIR = "Binomial_LR_Tuned"
os.makedirs(BASE_DIR, exist_ok=True)

# Splits for 80%
X_train_80 = train_80_main.drop([TARGET, ID_COL], axis=1)
y_train_80 = train_80_main[TARGET]

X_val_80 = val_10.drop([TARGET, ID_COL], axis=1)
y_val_80 = val_10[TARGET]

# Splits for 20%
train_20_raw = train_test_split(
    train_df, train_size=0.20, stratify=train_df[TARGET], random_state=SEED
)[0]

train_20_main, val_20 = train_test_split(
    train_20_raw, test_size=0.1111, stratify=train_20_raw[TARGET], random_state=SEED
)

X_train_20 = train_20_main.drop([TARGET, ID_COL], axis=1)
y_train_20 = train_20_main[TARGET]

X_val_20 = val_20.drop([TARGET, ID_COL], axis=1)
y_val_20 = val_20[TARGET]

# Internal Test
X_test_internal = test_internal.drop([TARGET, ID_COL], axis=1)
y_test_internal = test_internal[TARGET]

# Final Test Features
X_final_test = test_df.drop(ID_COL, axis=1)


# ==========================================================
# 4. RUN ALL 4 MODES
# ==========================================================
print("\nRunning 80% skewed...")
run_lr_tuned(os.path.join(BASE_DIR, "80_skewed"),
             X_train_80, y_train_80, X_val_80, y_val_80,
             X_test_internal, y_test_internal, X_final_test)
print("80% skewed completed!\n")

print("Running 80% non-skewed...")
run_lr_tuned(os.path.join(BASE_DIR, "80_nonskewed"),
             X_train_80, y_train_80, X_val_80, y_val_80,
             X_test_internal, y_test_internal, X_final_test)
print("80% non-skewed completed!\n")

print("Running 20% skewed...")
run_lr_tuned(os.path.join(BASE_DIR, "20_skewed"),
             X_train_20, y_train_20, X_val_20, y_val_20,
             X_test_internal, y_test_internal, X_final_test)
print("20% skewed completed!\n")

print("Running 20% non-skewed...")
run_lr_tuned(os.path.join(BASE_DIR, "20_nonskewed"),
             X_train_20, y_train_20, X_val_20, y_val_20,
             X_test_internal, y_test_internal, X_final_test)
print("20% non-skewed completed!\n")


print("\nAll Tuned Logistic Regression Models Completed Successfully!\n")
