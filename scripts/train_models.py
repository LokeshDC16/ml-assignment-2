from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Paths
# ============================================================

DATA_PATH = Path("data/phishing_websites.csv")
TEST_DATA_PATH = Path("test_data.csv")
MODEL_DIR = Path("model")
OUTPUT_PATH = Path("outputs/model_comparison.csv")


# ============================================================
# Configuration
# ============================================================

TARGET_COLUMN = "result"
N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# Load and prepare data
# ============================================================

def load_data():
    """Load the original dataset and reproduce the Stage 3 split."""

    df = pd.read_csv(DATA_PATH)

    # Remove exact duplicate rows.
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Create a group for each identical feature pattern.
    groups = pd.factorize(
        X.astype(str).agg("|".join, axis=1)
    )[0]

    # Reproduce the exact split used in prepare_data.py.
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)

    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)

    # Verify against the previously generated test_data.csv.
    saved_test = pd.read_csv(TEST_DATA_PATH)

    saved_X_test = saved_test.drop(columns=[TARGET_COLUMN])
    saved_y_test = saved_test[TARGET_COLUMN]

    if not X_test.equals(saved_X_test):
        raise ValueError(
            "Generated test features do not match test_data.csv."
        )

    if not y_test.equals(saved_y_test):
        raise ValueError(
            "Generated test labels do not match test_data.csv."
        )

    return X_train, X_test, y_train, y_test


# ============================================================
# Model definitions
# ============================================================

def create_models():
    """Create the five models required by the assignment."""

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=5
                    ),
                ),
            ]
        ),

        "Naive Bayes": GaussianNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train one model and calculate all six required metrics."""

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Probability of the positive class (+1).
    class_labels = list(model.classes_)
    positive_class_index = class_labels.index(1)

    y_probability = model.predict_proba(X_test)[
        :, positive_class_index
    ]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_probability),
        "Precision": precision_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    return model, metrics


# ============================================================
# Main
# ============================================================

def main():

    print("\n========================================")
    print("      ML MODEL TRAINING PIPELINE")
    print("========================================")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = load_data()

    print("\n========== DATA ==========")
    print(f"Training instances: {len(X_train)}")
    print(f"Testing instances:  {len(X_test)}")
    print(f"Features:           {X_train.shape[1]}")

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    models = create_models()

    results = []

    # --------------------------------------------------------
    # Train and evaluate
    # --------------------------------------------------------

    print("\n========================================")
    print("          MODEL TRAINING")
    print("========================================")

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        trained_model, metrics = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        # Create safe filename.
        filename = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        model_path = MODEL_DIR / f"{filename}.joblib"

        joblib.dump(trained_model, model_path)

        print(f"Saved model: {model_path}")

        print(
            f"Accuracy : {metrics['Accuracy']:.4f}"
        )
        print(
            f"AUC      : {metrics['AUC']:.4f}"
        )
        print(
            f"Precision: {metrics['Precision']:.4f}"
        )
        print(
            f"Recall   : {metrics['Recall']:.4f}"
        )
        print(
            f"F1       : {metrics['F1']:.4f}"
        )
        print(
            f"MCC      : {metrics['MCC']:.4f}"
        )

        result_row = {
            "ML Model": model_name,
            **metrics,
        }

        results.append(result_row)

    # --------------------------------------------------------
    # Create comparison table
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    metric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]

    results_df[metric_columns] = results_df[
        metric_columns
    ].round(4)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Display final comparison
    # --------------------------------------------------------

    print("\n========================================")
    print("       MODEL COMPARISON")
    print("========================================")

    print(results_df.to_string(index=False))

    print("\n========================================")
    print("Training completed successfully.")
    print(f"Comparison saved to: {OUTPUT_PATH}")
    print("========================================")


if __name__ == "__main__":
    main()