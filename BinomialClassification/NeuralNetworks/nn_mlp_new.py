# nn_all_models.py
import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# =====================================================
# Reproducibility
# =====================================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

# =====================================================
# Paths
# =====================================================
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

# =====================================================
# NEW SPLITTING
# =====================================================

# ------------ 80% SPLIT (80 train / 10 val / 10 test) ------------
train_full_80, test_internal_80 = train_test_split(
    train_df,
    test_size=0.10,
    random_state=SEED,
    stratify=train_df[TARGET]
)

train_80_main, val_80 = train_test_split(
    train_full_80,
    test_size=0.1111,  # 10% of original
    random_state=SEED,
    stratify=train_full_80[TARGET]
)


# ------------ 20% SPLIT (20 train / 40 val / 40 test) ------------
subset_20 = train_test_split(
    train_df,
    train_size=0.20,
    random_state=SEED,
    stratify=train_df[TARGET]
)[0]

# First: train = 20%, temp = 80%
train_20_main, temp_20 = train_test_split(
    subset_20,
    test_size=0.80,
    random_state=SEED,
    stratify=subset_20[TARGET]
)

# Then split temp into val = 40%, test = 40%
# temp_20 = 80% of subset
# val = 40% of subset => val fraction inside temp = 40/80 = 0.5
val_20, test_internal_20 = train_test_split(
    temp_20,
    test_size=0.5,
    random_state=SEED,
    stratify=temp_20[TARGET]
)

# =====================================================
# Prepare Internal Tests
# =====================================================
X_test_internal_80 = test_internal_80.drop([TARGET, ID_COL], axis=1)
y_test_internal_80 = test_internal_80[TARGET].values

X_test_internal_20 = test_internal_20.drop([TARGET, ID_COL], axis=1)
y_test_internal_20 = test_internal_20[TARGET].values

X_final_test = test_df.drop(ID_COL, axis=1)

# =====================================================
# NN Architectures
# =====================================================
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

# =====================================================
# RUNNER: trains + saves model + saves preprocessor
# =====================================================
def run_and_save_nn(model_name, run_name, 
                    X_train_df, y_train, 
                    X_val_df, y_val,
                    X_test_df, y_test,
                    X_final_df,
                    class_weight):

    save_dir = os.path.join(model_name, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Preprocessing
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    X_train = preproc.fit_transform(X_train_df)
    X_val   = preproc.transform(X_val_df)
    X_test  = preproc.transform(X_test_df)
    X_final = preproc.transform(X_final_df)

    input_dim = X_train.shape[1]

    # Select architecture
    if model_name == "NN_Small":
        model = build_nn_small(input_dim)
    elif model_name == "NN_Medium":
        model = build_nn_medium(input_dim)
    else:
        model = build_nn_deep(input_dim)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Class weights
    if class_weight == "balanced":
        from sklearn.utils.class_weight import compute_class_weight
        cw_vals = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        cw = {int(c): float(w) for c, w in zip(np.unique(y_train), cw_vals)}
    else:
        cw = None

    # Training
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1
    )

    # Save model + preprocessor
    model.save(os.path.join(save_dir, "model.keras"))
    joblib.dump(preproc, os.path.join(save_dir, "preprocessor.joblib"))

    # Validation evaluation
    val_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
    val_acc  = accuracy_score(y_val, val_pred)

    with open(os.path.join(save_dir, "classification_validation.txt"), "w") as f:
        f.write(classification_report(y_val, val_pred))
        f.write(f"\nValidation Accuracy: {val_acc:.6f}\n")

    # Test evaluation
    test_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    test_acc  = accuracy_score(y_test, test_pred)

    with open(os.path.join(save_dir, "classification_test.txt"), "w") as f:
        f.write(classification_report(y_test, test_pred))
        f.write(f"\nTest Accuracy: {test_acc:.6f}\n")

    # Accuracy summary
    with open(os.path.join(save_dir, "accuracy_summary.txt"), "w") as f:
        f.write(f"Validation Accuracy: {val_acc:.6f}\n")
        f.write(f"Test Accuracy:       {test_acc:.6f}\n")

    # Submission CSV
    final_pred = (model.predict(X_final) > 0.5).astype(int).flatten()

    submission = pd.DataFrame({
        "ProfileID": test_df[ID_COL],
        "RiskFlag": final_pred
    })
    submission.to_csv(os.path.join(save_dir, f"{run_name}_{model_name}.csv"), index=False)

    print(f"[DONE] {model_name} - {run_name}")


# =====================================================
# MAIN EXECUTION
# =====================================================
def run_all_nn_experiments():

    models = ["NN_Small"]

    # 80% splits
    X_train_80 = train_80_main.drop([TARGET, ID_COL], axis=1)
    y_train_80 = train_80_main[TARGET].values

    X_val_80 = val_80.drop([TARGET, ID_COL], axis=1)
    y_val_80 = val_80[TARGET].values

    # 20% splits
    X_train_20 = train_20_main.drop([TARGET, ID_COL], axis=1)
    y_train_20 = train_20_main[TARGET].values

    X_val_20 = val_20.drop([TARGET, ID_COL], axis=1)
    y_val_20 = val_20[TARGET].values

    for model_name in models:

        # 80% skewed
        run_and_save_nn(
            model_name, "80_skewed",
            X_train_80, y_train_80,
            X_val_80, y_val_80,
            X_test_internal_80, y_test_internal_80,
            X_final_test,
            class_weight=None
        )

        # 80% non-skewed
        run_and_save_nn(
            model_name, "80_nonskewed",
            X_train_80, y_train_80,
            X_val_80, y_val_80,
            X_test_internal_80, y_test_internal_80,
            X_final_test,
            class_weight="balanced"
        )

        # 20% skewed
        run_and_save_nn(
            model_name, "20_skewed",
            X_train_20, y_train_20,
            X_val_20, y_val_20,
            X_test_internal_20, y_test_internal_20,
            X_final_test,
            class_weight=None
        )

        # 20% non-skewed
        run_and_save_nn(
            model_name, "20_nonskewed",
            X_train_20, y_train_20,
            X_val_20, y_val_20,
            X_test_internal_20, y_test_internal_20,
            X_final_test,
            class_weight="balanced"
        )

    print("All NN Experiments Completed Successfully!")

# =====================================================
# Run
# =====================================================
if __name__ == "__main__":
    run_all_nn_experiments()
