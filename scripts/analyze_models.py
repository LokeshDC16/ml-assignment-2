from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


TEST_DATA_PATH = Path("test_data.csv")
MODEL_DIR = Path("model")
OUTPUT_DIR = Path("outputs")


TARGET_COLUMN = "result"


MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


def main():

    test_data = pd.read_csv(TEST_DATA_PATH)

    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]

    print("\n========================================")
    print("          MODEL ANALYSIS")
    print("========================================")

    for model_name, filename in MODELS.items():

        model_path = MODEL_DIR / filename

        print("\n========================================")
        print(model_name)
        print("========================================")

        model = joblib.load(model_path)

        y_pred = model.predict(X_test)

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        matrix = confusion_matrix(
            y_test,
            y_pred,
            labels=[-1, 1],
        )

        print("\nConfusion Matrix")
        print("Rows = Actual")
        print("Columns = Predicted")
        print("Labels = [-1, 1]")
        print(matrix)

        # ----------------------------------------------------
        # Classification Report
        # ----------------------------------------------------

        report = classification_report(
            y_test,
            y_pred,
            labels=[-1, 1],
            target_names=[
                "Phishing (-1)",
                "Legitimate (1)",
            ],
            zero_division=0,
        )

        print("\nClassification Report")
        print(report)

    print("\n========================================")
    print("Analysis completed successfully.")
    print("========================================")


if __name__ == "__main__":
    main()