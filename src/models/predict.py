import joblib
import pandas as pd


def load_model(model_path: str):
    return joblib.load(model_path)


def predict(model, df: pd.DataFrame) -> pd.Series:
    features = df.select_dtypes(include=["number"]).columns
    return pd.Series(model.predict(df[features]), index=df.index)
