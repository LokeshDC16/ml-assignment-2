import pandas as pd


DATA_PATH = "data/phishing_websites.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("\n========== BASIC INFORMATION ==========")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Target: {df.columns[-1]}")

    print("\n========== COLUMN NAMES ==========")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:2}. {column}")

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    missing_values = df.isnull().sum()
    print(missing_values)

    print("\n========== DUPLICATE ROWS ==========")
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")

    unique_df = df.drop_duplicates()

    print("\n========== AFTER REMOVING DUPLICATES ==========")
    print(f"Rows before: {len(df)}")
    print(f"Rows after:  {len(unique_df)}")
    print(f"Rows removed: {len(df) - len(unique_df)}")

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["result"].value_counts())
    print("\nTarget percentages:")
    print(df["result"].value_counts(normalize=True) * 100)

    print("\n========== TARGET DISTRIBUTION AFTER DEDUPLICATION ==========")
    print(unique_df["result"].value_counts())
    print("\nTarget percentages after deduplication:")
    print(unique_df["result"].value_counts(normalize=True) * 100)

    print("\n========== UNIQUE VALUES PER FEATURE ==========")

    for column in df.columns:
        unique_values = sorted(df[column].unique())
        print(f"{column}: {unique_values}")

    print("\n========== UNIQUE VALUE COUNT PER FEATURE ==========")

    for column in df.columns:
        print(f"{column}: {df[column].nunique()}")

    print("\n========== DUPLICATE FEATURE ROWS WITH CONFLICTING TARGETS ==========")

    feature_columns = df.columns[:-1]

    target_counts = (
        df.groupby(list(feature_columns))["result"]
        .nunique()
    )

    conflicting_rows = target_counts[target_counts > 1]

    print(f"Number of feature combinations with conflicting targets: "
          f"{len(conflicting_rows)}")

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())


if __name__ == "__main__":
    main()