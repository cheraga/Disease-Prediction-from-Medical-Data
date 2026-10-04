import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COLUMN


def load_data():
    """
    Load the raw medical dataset.
    """

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    return df


def validate_data(df):
    """
    Basic dataset validation.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    return True


if __name__ == "__main__":

    df = load_data()

    validate_data(df)

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())
