from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd


def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred),
    }
