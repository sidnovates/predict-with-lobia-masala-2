# nn_smote_models.py

import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import random

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# -----------------------------
# Paths
# -----------------------------
train_path = r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\train_updated.csv"
test_path  = r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\test_updated.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

TARGET = "RiskFlag"
ID_COL = "ProfileID"

# -----------------------------
# Feature Lists
# -----------------------------
numeric_features = [
    "ApplicantYears","AnnualEarnings","RequestedSum","TrustMetric","WorkDuration",
    "ActiveAccounts","OfferRate","RepayPeriod","DebtFactor"
]

categorical_features = [
    "QualificationLevel","WorkCategory","RelationshipStatus","FamilyObligation",
    "OwnsProperty","FundUseCase","JointApplicant"
]

# -----------------------------
# Shared ColumnTransformer
# -----------------------------
def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

# -----------------------------
# Build Models
# -----------------------------
def build_nn_small(input_dim):
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])

def build_nn_medium(input_dim):
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

def build_nn_deep(input_dim):
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])

# -----------------------------
# Helper: Run & Save
# -----------------------------
def run_smote_nn(model_name, run_name, X_train_raw, y_train, 
                 X_val_raw, y_val, X_test_internal_raw, y_test_internal,
                 X_final_test_raw, use_class_weight):

    base_dir = os.path.join("NN_SMOTE", model_name, run_name)
    os.makedirs(base_dir, exist_ok=True)

    preprocessor = create_preprocessor()

    # Fit ONLY on training split
    X_train_transformed = preprocessor.fit_transform(X_train_raw)
    X_val_transformed   = preprocessor.transform(X_val_raw)
    X_test_transformed  = preprocessor.transform(X_test_internal_raw)
    X_final_test_transformed = preprocessor.transform(X_final_test_raw)

    # -----------------------------
    # Apply SMOTE
    # -----------------------------
    smote = SMOTE(random_state=SEED)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_transformed, y_train)

    input_dim = X_train_balanced.shape[1]

    # -----------------------------
    # Build Model
    # -----------------------------
    if model_name == "NN_SMOTE_Small":
        model = build_nn_small(input_dim)
    elif model_name == "NN_SMOTE_Medium":
        model = build_nn_medium(input_dim)
    else:
        model = build_nn_deep(input_dim)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Class weight
    if use_class_weight:
        from sklearn.utils.class_weight import compute_class_weight
        cw_values = compute_class_weight("balanced", classes=np.unique(y_train_balanced), y=y_train_balanced)
        cw = {0: cw_values[0], 1: cw_values[1]}
    else:
        cw = None

    es = keras.callbacks.EarlyStopping(
        patience=6, monitor="val_loss", restore_best_weights=True, verbose=0
    )

    # -----------------------------
    # Train
    # -----------------------------
    model.fit(
        X_train_balanced, y_train_balanced,
        validation_data=(X_val_transformed, y_val),
        epochs=50,
        batch_size=256,
        callbacks=[es],
        class_weight=cw,
        verbose=1
    )

    # -----------------------------
    # Validation metrics
    # -----------------------------
    val_pred = (model.predict(X_val_transformed) > 0.5).astype(int).flatten()
    val_acc = accuracy_score(y_val, val_pred)
    val_rep = classification_report(y_val, val_pred)

    with open(os.path.join(base_dir, "classification_validation.txt"), "w") as f:
        f.write(val_rep)
        f.write(f"\nValidation Accuracy: {val_acc}\n")

    # -----------------------------
    # Internal test metrics
    # -----------------------------
    test_pred = (model.predict(X_test_transformed) > 0.5).astype(int).flatten()
    test_acc = accuracy_score(y_test_internal, test_pred)
    test_rep = classification_report(y_test_internal, test_pred)

    with open(os.path.join(base_dir, "classification_test.txt"), "w") as f:
        f.write(test_rep)
        f.write(f"\nTest Accuracy: {test_acc}\n")

    # Summary
    with open(os.path.join(base_dir, "accuracy_summary.txt"), "w") as f:
        f.write(f"Validation Accuracy: {val_acc}\nTest Accuracy: {test_acc}\n")

    # -----------------------------
    # Kaggle submission
    # -----------------------------
    final_pred = (model.predict(X_final_test_transformed) > 0.5).astype(int).flatten()

    submission = pd.DataFrame({
        "ProfileID": X_final_test_raw.index.map(lambda i: test_df.iloc[i][ID_COL]).values,
        "RiskFlag": final_pred
    })

    csv_name = f"{run_name}_{model_name}.csv"
    submission.to_csv(os.path.join(base_dir, csv_name), index=False)

    print(f"{model_name} — {run_name} Completed!\n")


# -----------------------------
# Split master datasets
# -----------------------------
train_full, test_internal = train_test_split(
    train_df, test_size=0.10, random_state=SEED, stratify=train_df[TARGET]
)

train_80, val_80 = train_test_split(
    train_full, test_size=0.1111, random_state=SEED, stratify=train_full[TARGET]
)

train_20, _ = train_test_split(
    train_df, train_size=0.20, random_state=SEED, stratify=train_df[TARGET]
)
train_20, val_20 = train_test_split(
    train_20, test_size=0.1111, random_state=SEED, stratify=train_20[TARGET]
)


# -----------------------------
# Prepare split DataFrames
# -----------------------------
X_train_80 = train_80.drop([TARGET, ID_COL], axis=1)
y_train_80 = train_80[TARGET].values

X_val_80 = val_80.drop([TARGET, ID_COL], axis=1)
y_val_80 = val_80[TARGET].values

X_train_20 = train_20.drop([TARGET, ID_COL], axis=1)
y_train_20 = train_20[TARGET].values

X_val_20 = val_20.drop([TARGET, ID_COL], axis=1)
y_val_20 = val_20[TARGET].values

X_test_internal_raw = test_internal.drop([TARGET, ID_COL], axis=1)
y_test_internal_arr = test_internal[TARGET].values

X_final_test_raw = test_df.drop(ID_COL, axis=1)


# -----------------------------
# Run All Models
# -----------------------------
def run_all_smote_nn():
    models = ["NN_SMOTE_Small", "NN_SMOTE_Medium", "NN_SMOTE_Deep"]

    for model_name in models:
        # 80 skewed
        run_smote_nn(
            model_name, "80_skewed",
            X_train_80, y_train_80,
            X_val_80, y_val_80,
            X_test_internal_raw, y_test_internal_arr,
            X_final_test_raw,
            use_class_weight=False
        )

        # 80 non-skewed
        run_smote_nn(
            model_name, "80_nonskewed",
            X_train_80, y_train_80,
            X_val_80, y_val_80,
            X_test_internal_raw, y_test_internal_arr,
            X_final_test_raw,
            use_class_weight=True
        )

        # 20 skewed
        run_smote_nn(
            model_name, "20_skewed",
            X_train_20, y_train_20,
            X_val_20, y_val_20,
            X_test_internal_raw, y_test_internal_arr,
            X_final_test_raw,
            use_class_weight=False
        )

        # 20 non-skewed
        run_smote_nn(
            model_name, "20_nonskewed",
            X_train_20, y_train_20,
            X_val_20, y_val_20,
            X_test_internal_raw, y_test_internal_arr,
            X_final_test_raw,
            use_class_weight=True
        )

    print("All SMOTE Neural Network models completed successfully!")


# -----------------------------
# Execute
# -----------------------------
if __name__ == "__main__":
    run_all_smote_nn()
