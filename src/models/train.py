import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def train_model(df: pd.DataFrame, target: str, model_path: str) -> RandomForestRegressor:
    """Train a random forest regression model and save it to disk."""
    features = df.drop(columns=[target]).select_dtypes(include=["number"]).columns
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model
