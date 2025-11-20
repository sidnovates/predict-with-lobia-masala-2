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

# -----------------------------
# Reproducibility seeds
# -----------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

# -----------------------------
# Paths (change only if needed)
# -----------------------------
train_path = r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\train_updated.csv"
test_path  = r"C:\Siddharth\Desktop\SEM5\ML\Project_2\BinomialClassification\Dataset\test_updated.csv"

# -----------------------------
# Load data
# -----------------------------
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

TARGET = "RiskFlag"
ID_COL = "ProfileID"

# -----------------------------
# Feature lists (as provided)
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
# Create held-out internal test (10%) used by all experiments
# -----------------------------
train_full, test_internal = train_test_split(
    train_df, test_size=0.10, random_state=SEED, stratify=train_df[TARGET]
)

# For 80% experiments: we'll split train_full into train(80%) and val(10%) by splitting train_full with test_size ~0.1111
train_80_main, val_10 = train_test_split(
    train_full, test_size=0.1111, random_state=SEED, stratify=train_full[TARGET]
)
# For 20% experiments: sample 20% of original train_df (stratified)
train_20 = train_test_split(train_df, train_size=0.20, random_state=SEED, stratify=train_df[TARGET])[0]
train_20_main, val_20 = train_test_split(
    train_20, test_size=0.1111, random_state=SEED, stratify=train_20[TARGET]
)

# Internal test from earlier
X_test_internal = test_internal.drop([TARGET, ID_COL], axis=1)
y_test_internal = test_internal[TARGET].values

# Final Kaggle test features
X_final_test = test_df.drop(ID_COL, axis=1)

# -----------------------------
# Preprocessor (shared)
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ],
    remainder='drop'
)

# Fit preprocessor on the training data of the largest split to keep consistency.
# We'll fit inside each run to avoid data leakage across variants (we will fit per-run on the run's X_train).

# -----------------------------
# Helper: build models
# -----------------------------
def build_nn_small(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

def build_nn_medium(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

def build_nn_deep(input_dim):
    model = keras.Sequential([
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
    return model

# -----------------------------
# Generic runner for one model & one dataset configuration
# -----------------------------
def run_and_save_nn(model_name, run_name, X_train_df, y_train_ser, X_val_df, y_val_ser,
                   X_test_internal_df, y_test_internal_arr, X_final_test_df, class_weight):
    """
    model_name: 'NN_Small' / 'NN_Medium' / 'NN_Deep'
    run_name: '20_skewed' etc (used as folder name)
    X_train_df, X_val_df, X_test_internal_df, X_final_test_df: dataframes (raw, untransformed)
    class_weight: None or 'balanced' (string) - we will convert to dict for keras if needed
    """
    base_folder = os.path.join(model_name, run_name)
    os.makedirs(base_folder, exist_ok=True)

    # Fit preprocessor on X_train only to avoid leakage
    local_preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
    )
    X_train_p = local_preproc.fit_transform(X_train_df)
    X_val_p   = local_preproc.transform(X_val_df)
    X_test_p  = local_preproc.transform(X_test_internal_df)
    X_final_p = local_preproc.transform(X_final_test_df)

    input_dim = X_train_p.shape[1]

    # Choose constructor based on model_name
    if model_name == "NN_Small":
        model = build_nn_small(input_dim)
    elif model_name == "NN_Medium":
        model = build_nn_medium(input_dim)
    elif model_name == "NN_Deep":
        model = build_nn_deep(input_dim)
    else:
        raise ValueError("Unknown model_name")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # prepare class_weight for keras: if 'balanced', compute dict
    if class_weight == "balanced":
        # compute class weights from training labels
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train_ser)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_ser)
        cw = {int(c): float(w) for c, w in zip(classes, weights)}
    else:
        cw = None

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0)
    ]

    # Fit
    history = model.fit(
        X_train_p, y_train_ser,
        validation_data=(X_val_p, y_val_ser),
        epochs=50,
        batch_size=256,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1
    )

    # Predictions and reports
    val_probs = model.predict(X_val_p, batch_size=256)
    val_pred = (val_probs > 0.5).astype(int).flatten()
    val_acc = accuracy_score(y_val_ser, val_pred)
    val_report = classification_report(y_val_ser, val_pred)

    with open(os.path.join(base_folder, "classification_validation.txt"), "w") as f:
        f.write(val_report)
        f.write("\nValidation Accuracy: {:.6f}\n".format(val_acc))

    test_probs = model.predict(X_test_p, batch_size=256)
    test_pred = (test_probs > 0.5).astype(int).flatten()
    test_acc = accuracy_score(y_test_internal_arr, test_pred)
    test_report = classification_report(y_test_internal_arr, test_pred)

    with open(os.path.join(base_folder, "classification_test.txt"), "w") as f:
        f.write(test_report)
        f.write("\nTest Accuracy: {:.6f}\n".format(test_acc))

    with open(os.path.join(base_folder, "accuracy_summary.txt"), "w") as f:
        f.write("Validation Accuracy: {:.6f}\n".format(val_acc))
        f.write("Test Accuracy:       {:.6f}\n".format(test_acc))

    # Submission CSV (Kaggle format)
    final_probs = model.predict(X_final_p, batch_size=256)
    final_pred = (final_probs > 0.5).astype(int).flatten()

    csv_name = f"{run_name}_{model_name}.csv"
    submission = pd.DataFrame({
        "ProfileID": test_df[ID_COL],
        "RiskFlag": final_pred.astype(int)
    })
    submission.to_csv(os.path.join(base_folder, csv_name), index=False)

    print(f"{model_name} - {run_name} Completed!")

# -----------------------------
# Run all combinations for each model
# -----------------------------
def run_all_nn_experiments():
    models = ["NN_Small", "NN_Medium", "NN_Deep"]

    # Prepare the different dataset splits (dataframes)
    # 80% experiments use train_80_main and val_10; 20% experiments use train_20_main and val_20
    # test_internal is common for internal test
    # final test is test_df (provided)

    # Prepare splits for 80% variant
    X_train_80 = train_80_main.drop([TARGET, ID_COL], axis=1)
    y_train_80 = train_80_main[TARGET].values

    X_val_80 = val_10.drop([TARGET, ID_COL], axis=1)
    y_val_80 = val_10[TARGET].values

    # Prepare splits for 20% variant
    X_train_20 = train_20_main.drop([TARGET, ID_COL], axis=1)
    y_train_20 = train_20_main[TARGET].values

    X_val_20 = val_20.drop([TARGET, ID_COL], axis=1)
    y_val_20 = val_20[TARGET].values

    X_test_int_df = X_test_internal.copy()
    y_test_int_arr = y_test_internal.copy()

    X_final_test_df = X_final_test.copy()

    for model_name in models:
        # 80% skewed
        run_and_save_nn(
            model_name=model_name,
            run_name="80_skewed",
            X_train_df=X_train_80, y_train_ser=y_train_80,
            X_val_df=X_val_80, y_val_ser=y_val_80,
            X_test_internal_df=X_test_int_df, y_test_internal_arr=y_test_int_arr,
            X_final_test_df=X_final_test_df,
            class_weight=None
        )

        # 80% non-skewed
        run_and_save_nn(
            model_name=model_name,
            run_name="80_nonskewed",
            X_train_df=X_train_80, y_train_ser=y_train_80,
            X_val_df=X_val_80, y_val_ser=y_val_80,
            X_test_internal_df=X_test_int_df, y_test_internal_arr=y_test_int_arr,
            X_final_test_df=X_final_test_df,
            class_weight="balanced"
        )

        # 20% skewed
        run_and_save_nn(
            model_name=model_name,
            run_name="20_skewed",
            X_train_df=X_train_20, y_train_ser=y_train_20,
            X_val_df=X_val_20, y_val_ser=y_val_20,
            X_test_internal_df=X_test_int_df, y_test_internal_arr=y_test_int_arr,
            X_final_test_df=X_final_test_df,
            class_weight=None
        )

        # 20% non-skewed
        run_and_save_nn(
            model_name=model_name,
            run_name="20_nonskewed",
            X_train_df=X_train_20, y_train_ser=y_train_20,
            X_val_df=X_val_20, y_val_ser=y_val_20,
            X_test_internal_df=X_test_int_df, y_test_internal_arr=y_test_int_arr,
            X_final_test_df=X_final_test_df,
            class_weight="balanced"
        )

    print("All NN models (Small/Medium/Deep) for 20% & 80% with skewed/non-skewed variants completed successfully!")

# -----------------------------
# Execute
# -----------------------------
if __name__ == "__main__":
    run_all_nn_experiments()
