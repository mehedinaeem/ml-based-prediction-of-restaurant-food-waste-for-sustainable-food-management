import pandas as pd


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include='all')


def validate_columns(df: pd.DataFrame, required_columns: list) -> bool:
    return all(col in df.columns for col in required_columns)
