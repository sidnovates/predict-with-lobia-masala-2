import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real
import warnings
warnings.filterwarnings("ignore")


# ============================================
# PATHS
# ============================================
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
# PREPROCESSOR
# ============================================================
def get_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )


# ============================================================
# FUNCTION TO RUN EACH MODE (20/80 × skewed/non-skewed)
# ============================================================
def run_model(mode_name, X_train, y_train, X_val, y_val, 
              X_test_internal, y_test_internal, X_final_test, class_weight):

    folder = os.path.join("Binomial_LR_Bayesian", mode_name)
    os.makedirs(folder, exist_ok=True)

    print(f"\n============================")
    print(f"Running {mode_name} — Random Search")
    print("============================\n")

    # ------------------------------
    # Base LR model for tuning
    # ------------------------------
    base_clf = Pipeline(steps=[
        ("preprocessor", get_preprocessor()),
        ("clf", LogisticRegression(solver="liblinear", max_iter=500))
    ])

    # ------------------------------
    # 1️⃣ RANDOM SEARCH FIRST
    # ------------------------------
    random_grid = {
        "clf__C": np.logspace(-3, 2, 20),   # wide range
        "clf__penalty": ["l2"]
    }

    random_search = RandomizedSearchCV(
        base_clf,
        param_distributions=random_grid,
        n_iter=25,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    print("\nBest params from RandomSearchCV:")
    print(random_search.best_params_)

    best_C_initial = random_search.best_params_["clf__C"]

    # ------------------------------
    # 2️⃣ BAYESIAN OPTIMIZATION (Refinement)
    # ------------------------------
    print(f"\n============================")
    print(f"Running {mode_name} — Bayesian Optimization")
    print("============================\n")

    bayes_clf = Pipeline(steps=[
        ("preprocessor", get_preprocessor()),
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight=class_weight,
            max_iter=500
        ))
    ])

    search_space = {
        "clf__C": Real(best_C_initial / 3, best_C_initial * 3, prior="log-uniform")
    }

    bayes_search = BayesSearchCV(
        bayes_clf,
        search_spaces=search_space,
        n_iter=20,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    bayes_search.fit(X_train, y_train)

    print("\nBest params from Bayesian Search:")
    print(bayes_search.best_params_)

    best_model = bayes_search.best_estimator_

    # ------------------------------
    # VALIDATION PERFORMANCE
    # ------------------------------
    val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    with open(os.path.join(folder, "classification_validation.txt"), "w") as f:
        f.write(classification_report(y_val, val_pred))
        f.write(f"\nValidation Accuracy: {val_acc}\n")

    # ------------------------------
    # INTERNAL TEST PERFORMANCE
    # ------------------------------
    test_pred = best_model.predict(X_test_internal)
    test_acc = accuracy_score(y_test_internal, test_pred)

    with open(os.path.join(folder, "classification_test.txt"), "w") as f:
        f.write(classification_report(y_test_internal, test_pred))
        f.write(f"\nTest Accuracy: {test_acc}\n")

    # ------------------------------
    # SUMMARY FILE
    # ------------------------------
    with open(os.path.join(folder, "accuracy_summary.txt"), "w") as f:
        f.write(f"Validation Accuracy: {val_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write("\nBest Params (Random Search):\n")
        f.write(str(random_search.best_params_))
        f.write("\n\nBest Params (Bayesian Search):\n")
        f.write(str(bayes_search.best_params_))

    # ------------------------------
    # FINAL KAGGLE SUBMISSION
    # ------------------------------
    final_pred = best_model.predict(X_final_test)

    submission = pd.DataFrame({
        "ProfileID": test_df[ID_COL],
        "RiskFlag": final_pred.astype(int)
    })

    submission.to_csv(os.path.join(folder, f"{mode_name}_Bayesian_LR.csv"), index=False)

    print(f"\n{mode_name} — Completed Successfully!\n")


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_all():

    print("\n============== Running Bayesian Tuned Logistic Regression ==============\n")

    os.makedirs("Binomial_LR_Bayesian", exist_ok=True)

    # ----------- 80% SPLIT -----------
    train_80, test_internal = train_test_split(
        train_df, test_size=0.10, random_state=42, stratify=train_df[TARGET]
    )

    train_80_main, val_10 = train_test_split(
        train_80, test_size=0.1111, random_state=42, stratify=train_80[TARGET]
    )

    # ----------- 20% SPLIT -----------
    train_20, _ = train_test_split(
        train_df, train_size=0.20, random_state=42, stratify=train_df[TARGET]
    )

    train_20_main, val_20 = train_test_split(
        train_20, test_size=0.1111, random_state=42, stratify=train_20[TARGET]
    )

    # COMMON TEST SPLIT
    X_test_internal = test_internal.drop([TARGET, ID_COL], axis=1)
    y_test_internal = test_internal[TARGET].values

    X_final_test = test_df.drop(ID_COL, axis=1)

    # --------------------
    # Now running all 4 modes
    # --------------------

    # 80% skewed
    run_model("80_skewed",
              train_80_main.drop([TARGET, ID_COL], axis=1), train_80_main[TARGET],
              val_10.drop([TARGET, ID_COL], axis=1), val_10[TARGET],
              X_test_internal, y_test_internal, X_final_test,
              class_weight=None)

    # 80% non-skewed
    run_model("80_nonskewed",
              train_80_main.drop([TARGET, ID_COL], axis=1), train_80_main[TARGET],
              val_10.drop([TARGET, ID_COL], axis=1), val_10[TARGET],
              X_test_internal, y_test_internal, X_final_test,
              class_weight="balanced")

    # 20% skewed
    run_model("20_skewed",
              train_20_main.drop([TARGET, ID_COL], axis=1), train_20_main[TARGET],
              val_20.drop([TARGET, ID_COL], axis=1), val_20[TARGET],
              X_test_internal, y_test_internal, X_final_test,
              class_weight=None)

    # 20% non-skewed
    run_model("20_nonskewed",
              train_20_main.drop([TARGET, ID_COL], axis=1), train_20_main[TARGET],
              val_20.drop([TARGET, ID_COL], axis=1), val_20[TARGET],
              X_test_internal, y_test_internal, X_final_test,
              class_weight="balanced")

    print("\nALL BAYESIAN TUNED LOGISTIC REGRESSION MODELS COMPLETED!\n")


# ============================================================
# Run it
# ============================================================
run_all()
