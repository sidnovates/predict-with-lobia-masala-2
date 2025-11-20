import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import BayesianOptimization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ============================================================
# PATHS
# ============================================================
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
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

# ============================================================
# TUNABLE MODEL
# ============================================================
def build_tunable_model(hp, input_dim, model_type):

    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    # -------------------------------
    # SMALL / MEDIUM / DEEP logic
    # -------------------------------
    if model_type == "NN_Small":
        hp_units = hp.Choice("units", [32, 64, 128])
        hp_drop  = hp.Float("dropout", 0.1, 0.5, step=0.1)

        model.add(layers.Dense(hp_units, activation="relu"))
        model.add(layers.Dropout(hp_drop))

    elif model_type == "NN_Medium":
        for i in range(2):
            model.add(
                layers.Dense(
                    hp.Int(f"units{i+1}", 64, 256, step=64),
                    activation=hp.Choice(f"act{i+1}", ["relu", "tanh"])
                )
            )
            model.add(layers.Dropout(hp.Float(f"drop{i+1}", 0.1, 0.5, step=0.1)))

    elif model_type == "NN_Deep":
        for i in range(3):
            model.add(
                layers.Dense(
                    hp.Int(f"units{i+1}", 64, 256, step=64),
                    activation="relu"
                )
            )
            model.add(layers.Dropout(hp.Float(f"drop{i+1}", 0.1, 0.5, step=0.1)))

    # OUTPUT LAYER
    model.add(layers.Dense(1, activation="sigmoid"))

    # OPTIMIZER
    lr = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])
    optimizer = keras.optimizers.Adam(lr)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    hp_batch = hp.Choice("batch_size", [64, 128, 256])

    return model


# ============================================================
# TRAINING + TUNING FUNCTION
# ============================================================
def run_bayesian_tuning(
    run_name, model_type,
    X_train, y_train,
    X_val, y_val,
    X_test_internal, y_test_internal,
    X_final_test,
    use_class_weight
):

    print(f"\n----- Bayesian Tuning for: {run_name} ({model_type}) -----\n")

    pre = get_preprocessor()
    X_train_p = pre.fit_transform(X_train)
    X_val_p = pre.transform(X_val)
    X_test_p = pre.transform(X_test_internal)
    X_final_p = pre.transform(X_final_test)

    input_dim = X_train_p.shape[1]

    # CLASS WEIGHTS
    if use_class_weight:
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        cw = {int(c): float(w) for c, w in zip(classes, weights)}
    else:
        cw = None

    # -------------------------------
    # Main tuner (correct API)
    # -------------------------------
    tuner = BayesianOptimization(
        lambda hp: build_tunable_model(hp, input_dim, model_type),
        objective="val_accuracy",
        max_trials=20,
        directory="bayesian_tuning",
        project_name=f"{run_name}_{model_type}"
    )
    tuner.search(
        X_train_p, y_train,
        validation_data=(X_val_p, y_val),
        epochs=30,
        class_weight=cw,
        verbose=1
    )

    # BEST HYPERPARAMETERS
    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("\nBest Hyperparameters:")
    print(best_hp.values)

    model = tuner.hypermodel.build(best_hp)

    # FINAL TRAINING
    model.fit(
        X_train_p, y_train,
        validation_data=(X_val_p, y_val),
        epochs=30,
        batch_size=best_hp["batch_size"],
        class_weight=cw,
        verbose=1
    )

    # OUTPUT DIRECTORY
    out_dir = os.path.join("NN_Bayesian_Tuned", model_type, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # VALIDATION METRICS
    val_pred = (model.predict(X_val_p) > 0.5).astype(int).flatten()
    val_acc = accuracy_score(y_val, val_pred)

    with open(os.path.join(out_dir, "classification_validation.txt"), "w") as f:
        f.write(classification_report(y_val, val_pred))
        f.write(f"\nValidation Accuracy: {val_acc}\n")

    # INTERNAL TEST
    test_pred = (model.predict(X_test_p) > 0.5).astype(int).flatten()
    test_acc = accuracy_score(y_test_internal, test_pred)

    with open(os.path.join(out_dir, "classification_test.txt"), "w") as f:
        f.write(classification_report(y_test_internal, test_pred))
        f.write(f"\nTest Accuracy: {test_acc}\n")

    # SUMMARY
    with open(os.path.join(out_dir, "accuracy_summary.txt"), "w") as f:
        f.write(f"Validation Accuracy: {val_acc}\nTest Accuracy: {test_acc}\n")

    # FINAL SUBMISSION
    final_pred = (model.predict(X_final_p) > 0.5).astype(int).flatten()
    pd.DataFrame({
        "ProfileID": test_df[ID_COL],
        "RiskFlag": final_pred
    }).to_csv(os.path.join(out_dir, f"{run_name}_{model_type}.csv"), index=False)

    print(f"{run_name} ({model_type}) Completed!\n")


# ============================================================
# SELECTED MODELS FOR TUNING
# ============================================================
def run_selected_bayesian_tuning():

    # MAIN SPLITS
    train_full, test_internal = train_test_split(
        train_df, test_size=0.10,
        stratify=train_df[TARGET], random_state=SEED
    )

    # 80% splits
    train_80, val_80 = train_test_split(
        train_full, test_size=0.1111,
        stratify=train_full[TARGET], random_state=SEED
    )

    # 20% splits
    train_20_raw = train_test_split(
        train_df, train_size=0.20,
        stratify=train_df[TARGET], random_state=SEED
    )[0]

    train_20, val_20 = train_test_split(
        train_20_raw, test_size=0.1111,
        stratify=train_20_raw[TARGET], random_state=SEED
    )

    # Final test input
    X_final_test = test_df.drop(ID_COL, axis=1)

    # SELECTED BEST MODELS
    models_to_tune = [
        ("NN_Small",  "20_skewed",   train_20, val_20, False),
        ("NN_Small",  "80_skewed",   train_80, val_80, False),
        ("NN_Medium", "20_skewed",   train_20, val_20, False),
        ("NN_Medium", "80_skewed",   train_80, val_80, False),
        ("NN_Deep",   "80_nonskewed",train_80, val_80, True),
    ]

    for model_type, run_name, train_split, val_split, non_skewed in models_to_tune:

        X_train = train_split.drop([TARGET, ID_COL], axis=1)
        y_train = train_split[TARGET].values

        X_val = val_split.drop([TARGET, ID_COL], axis=1)
        y_val = val_split[TARGET].values

        X_test_internal = test_internal.drop([TARGET, ID_COL], axis=1)
        y_test_internal = test_internal[TARGET].values

        run_bayesian_tuning(
            run_name, model_type,
            X_train, y_train,
            X_val, y_val,
            X_test_internal, y_test_internal,
            X_final_test,
            use_class_weight=non_skewed
        )


if __name__ == "__main__":
    run_selected_bayesian_tuning()
    print("\nAll selected Bayesian NN tuning completed successfully!\n")
