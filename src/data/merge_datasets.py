import pandas as pd


def merge_datasets(restaurant_df: pd.DataFrame, waste_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge restaurant, waste, and weather data into a single DataFrame."""
    merged = restaurant_df.merge(waste_df, on=["restaurant_id", "date"], how="left")
    merged = merged.merge(weather_df, on="date", how="left")
    return merged
