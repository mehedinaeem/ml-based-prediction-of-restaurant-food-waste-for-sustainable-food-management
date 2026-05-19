import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for modeling."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna()
    return df
