from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


DATA_PATH = Path("data/phishing_websites.csv")
TEST_DATA_PATH = Path("test_data.csv")

TARGET_COLUMN = "result"
N_SPLITS = 5
RANDOM_STATE = 42


def main():
    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    print("\n========== ORIGINAL DATASET ==========")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # ---------------------------------------------------------
    # 2. Remove exact duplicate rows
    # ---------------------------------------------------------
    duplicate_count = df.duplicated().sum()

    print("\n========== DUPLICATE HANDLING ==========")
    print(f"Exact duplicate rows found: {duplicate_count}")

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Rows after removing duplicates: {df.shape[0]}")

    # ---------------------------------------------------------
    # 3. Separate features and target
    # ---------------------------------------------------------
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print("\n========== FEATURES AND TARGET ==========")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target column: {TARGET_COLUMN}")

    # ---------------------------------------------------------
    # 4. Create groups based on identical feature patterns
    # ---------------------------------------------------------
    groups = pd.factorize(
        X.astype(str).agg("|".join, axis=1)
    )[0]

    print("\n========== FEATURE GROUPING ==========")
    print(f"Unique feature groups: {len(set(groups))}")

    # ---------------------------------------------------------
    # 5. Stratified group-aware train-test split
    # ---------------------------------------------------------
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)

    y_train = y.iloc[train_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)

    print("\n========== TRAIN-TEST SPLIT ==========")
    print(f"Training instances: {len(X_train)}")
    print(f"Testing instances:  {len(X_test)}")

    print("\nTraining target distribution:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True) * 100)

    print("\nTesting target distribution:")
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True) * 100)

    # ---------------------------------------------------------
    # 6. Verify that feature groups do not overlap
    # ---------------------------------------------------------
    train_groups = set(groups[train_indices])
    test_groups = set(groups[test_indices])

    overlapping_groups = train_groups.intersection(test_groups)

    print("\n========== TRAIN-TEST FEATURE GROUP OVERLAP ==========")
    print(
        "Identical feature patterns present in both train and test:",
        len(overlapping_groups),
    )

    # ---------------------------------------------------------
    # 7. Create test_data.csv
    # ---------------------------------------------------------
    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test

    test_data.to_csv(TEST_DATA_PATH, index=False)

    print("\n========== TEST DATA ==========")
    print(f"Saved to: {TEST_DATA_PATH}")
    print(f"Shape: {test_data.shape}")

    # ---------------------------------------------------------
    # 8. Final verification
    # ---------------------------------------------------------
    print("\n========== FINAL VERIFICATION ==========")
    print(
        f"Missing values in training features: "
        f"{X_train.isnull().sum().sum()}"
    )

    print(
        f"Missing values in testing features:  "
        f"{X_test.isnull().sum().sum()}"
    )

    print("\nData preparation completed successfully.")


if __name__ == "__main__":
    main()