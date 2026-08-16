from pathlib import Path

from ucimlrepo import fetch_ucirepo


def main():
    print("Downloading UCI Phishing Websites dataset...")

    dataset = fetch_ucirepo(id=327)

    features = dataset.data.features
    target = dataset.data.targets

    data = features.copy()
    data[target.columns[0]] = target.iloc[:, 0]

    output_path = Path("data/phishing_websites.csv")
    data.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")


if __name__ == "__main__":
    main()