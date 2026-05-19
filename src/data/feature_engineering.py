import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used for prediction."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["price_ratio"] = df["avg_checkout_price"] / df["avg_base_price"].replace(0, 1)
    return df
