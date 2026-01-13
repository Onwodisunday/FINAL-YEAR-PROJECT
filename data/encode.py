import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_PATH = "Cloud_Dataset.csv"


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Cloud_Dataset.csv not found")
    return pd.read_csv(path)


def clean_and_encode(df):
    # Strip column names
    df.columns = [c.strip() for c in df.columns]

    # Drop timestamp column
    df = df.drop(columns=["timestamp"], errors="ignore")

    # Drop duplicates
    df = df.drop_duplicates()

    # Remove rows with missing target
    df["target"] = df["target"].astype(str).str.strip()
    df = df[df["target"].notna() & (df["target"] != "")]

    # Handle missing values
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Fill numeric columns with median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical columns with mode
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns
    encoders = {}
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    return df, encoders


def main():
    df = load_data(DATA_PATH)

    df_clean, encoders = clean_and_encode(df)

    # Save cleaned & encoded dataset
    df_clean.to_csv("cleaned_encoded_dataset.csv", index=False)

    # Split data
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Save train/test CSVs
    train_df = X_train.copy()
    train_df["target"] = y_train
    train_df.to_csv("train_data.csv", index=False)

    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv("test_data.csv", index=False)

    # Save encoder mappings
    encoder_df = []
    for col, mapping in encoders.items():
        for k, v in mapping.items():
            encoder_df.append([col, k, v])

    pd.DataFrame(
        encoder_df, columns=["column", "original_value", "encoded_value"]
    ).to_csv("encoding_mapping.csv", index=False)

    print("SUCCESS ✅")
    print("Files generated:")
    print(" - cleaned_encoded_dataset.csv")
    print(" - train_data.csv")
    print(" - test_data.csv")
    print(" - encoding_mapping.csv")


if __name__ == "__main__":
    main()
