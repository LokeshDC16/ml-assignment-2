from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path("data/phishing_websites.csv")
OUTPUT_DIR = Path("outputs/plots")
TARGET_COLUMN = "result"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Use the same duplicate handling as the ML pipeline
    df = df.drop_duplicates().reset_index(drop=True)

    print("\n========== EDA DATASET ==========")
    print(f"Rows: {df.shape[0]}")
    print(f"Features: {df.shape[1] - 1}")

    # ---------------------------------------------------------
    # 1. Class distribution
    # ---------------------------------------------------------
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()

    plt.figure(figsize=(6, 4))

    class_counts.plot(kind="bar")

    plt.title("Target Class Distribution")
    plt.xlabel("Class (-1 = Phishing, 1 = Legitimate)")
    plt.ylabel("Number of Instances")
    plt.xticks(rotation=0)
    plt.tight_layout()

    class_plot_path = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(class_plot_path, dpi=300)
    plt.close()

    print(f"Saved: {class_plot_path}")

    # ---------------------------------------------------------
    # 2. Correlation heatmap
    # ---------------------------------------------------------
    correlation_matrix = df.corr()

    plt.figure(figsize=(18, 14))

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        square=False,
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    heatmap_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    print(f"Saved: {heatmap_path}")

    # ---------------------------------------------------------
    # 3. Correlation with target
    # ---------------------------------------------------------
    target_correlation = (
        correlation_matrix[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(ascending=False)
    )

    print("\n========== FEATURE CORRELATION WITH TARGET ==========")
    print(target_correlation)

    plt.figure(figsize=(10, 8))

    target_correlation.sort_values().plot(kind="barh")

    plt.title("Feature Correlation with Target")
    plt.xlabel("Correlation")
    plt.ylabel("Feature")
    plt.tight_layout()

    target_plot_path = OUTPUT_DIR / "target_correlation.png"
    plt.savefig(target_plot_path, dpi=300)
    plt.close()

    print(f"\nSaved: {target_plot_path}")

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()