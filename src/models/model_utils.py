import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
