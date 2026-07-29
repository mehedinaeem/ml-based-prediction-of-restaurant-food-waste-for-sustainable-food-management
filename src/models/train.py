"""Train and evaluate restaurant food-waste regression models chronologically."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "eda_processed_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATE_COLUMN = "date"
TARGET = "food_waste_kg"
TRAIN_RATIO = 0.70
CV_SPLITS = 5

SELECTED_FEATURES = [
    "year",
    "month",
    "week",
    "day_of_week",
    "restaurant_id",
    "city_code",
    "region_code",
    "center_type",
    "op_area",
    "unique_meals",
    "dominant_category",
    "dominant_cuisine",
    "avg_checkout_price",
    "avg_base_price",
    "emailer_promo_rate",
    "homepage_feature_rate",
    "is_weekend",
    "is_holiday",
    "temperature_c",
]


def load_and_sort_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load data and stably order records by date, then restaurant."""
    df = pd.read_csv(path)
    if not pd.api.types.is_numeric_dtype(df[DATE_COLUMN]):
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="raise")
    df = df.sort_values(
        by=[DATE_COLUMN, "restaurant_id"], kind="mergesort"
    ).reset_index(drop=True)
    return df


def create_chronological_split(
    df: pd.DataFrame, train_ratio: float = TRAIN_RATIO
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split complete date groups into the earliest train and latest test periods."""
    unique_dates = np.sort(df[DATE_COLUMN].unique())
    split_index = int(len(unique_dates) * train_ratio)
    if split_index <= 0 or split_index >= len(unique_dates):
        raise ValueError("The dataset needs enough unique dates for a non-empty split.")
    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]
    train_df = df[df[DATE_COLUMN].isin(train_dates)].copy()
    test_df = df[df[DATE_COLUMN].isin(test_dates)].copy()
    return train_df, test_df, train_dates, test_dates


def _json_safe_date(value: Any) -> int | float | str:
    """Convert numeric and timestamp date keys to JSON-safe values."""
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return str(value)


def validate_split(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_dates: np.ndarray,
    test_dates: np.ndarray,
) -> dict[str, Any]:
    """Assert temporal separation and print a clear split summary."""
    overlapping_dates = len(set(train_dates).intersection(set(test_dates)))
    assert overlapping_dates == 0
    assert train_df[DATE_COLUMN].max() < test_df[DATE_COLUMN].min()
    assert len(train_df) > 0
    assert len(test_df) > 0

    training_ratio = len(train_df) / len(df)
    testing_ratio = len(test_df) / len(df)
    summary = {
        "split_strategy": "chronological",
        "training_ratio": TRAIN_RATIO,
        "testing_ratio": 1.0 - TRAIN_RATIO,
        "total_records": len(df),
        "training_records": len(train_df),
        "testing_records": len(test_df),
        "total_unique_dates": int(df[DATE_COLUMN].nunique()),
        "training_unique_dates": len(train_dates),
        "testing_unique_dates": len(test_dates),
        "training_start_date": _json_safe_date(train_df[DATE_COLUMN].min()),
        "training_end_date": _json_safe_date(train_df[DATE_COLUMN].max()),
        "testing_start_date": _json_safe_date(test_df[DATE_COLUMN].min()),
        "testing_end_date": _json_safe_date(test_df[DATE_COLUMN].max()),
        "split_date": _json_safe_date(test_dates[0]),
        "overlapping_dates": overlapping_dates,
    }

    print("\n=== CHRONOLOGICAL SPLIT SUMMARY ===")
    print(f"Unique dates: {summary['total_unique_dates']}")
    print(f"Split date (first test date): {summary['split_date']}")
    print(
        f"Training date range: {summary['training_start_date']} to "
        f"{summary['training_end_date']}"
    )
    print(
        f"Testing date range: {summary['testing_start_date']} to "
        f"{summary['testing_end_date']}"
    )
    print(f"Training shape: {train_df.shape} ({training_ratio:.2%})")
    print(f"Testing shape: {test_df.shape} ({testing_ratio:.2%})")
    print(f"Overlapping dates: {overlapping_dates}")
    return summary


def create_models() -> dict[str, Any]:
    """Create models with the repository's existing hyperparameters."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42, n_estimators=100, learning_rate=0.05, max_depth=3
        ),
    }


def evaluate_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "R2 Score": float(r2_score(y_true, predictions)),
    }


def run_time_series_cross_validation(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run five expanding-window folds over complete unique-date groups."""
    unique_dates = np.sort(train_df[DATE_COLUMN].unique())
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    fold_results: list[dict[str, Any]] = []

    print("\n=== 5-FOLD EXPANDING-WINDOW CROSS-VALIDATION ===")
    for fold, (train_indices, validation_indices) in enumerate(
        tscv.split(unique_dates), start=1
    ):
        assert train_indices.max() < validation_indices.min()
        fold_train_dates = unique_dates[train_indices]
        fold_validation_dates = unique_dates[validation_indices]
        assert not set(fold_train_dates).intersection(set(fold_validation_dates))
        assert fold_train_dates.max() < fold_validation_dates.min()

        fold_train = train_df[
            train_df[DATE_COLUMN].isin(fold_train_dates)
        ].copy()
        fold_validation = train_df[
            train_df[DATE_COLUMN].isin(fold_validation_dates)
        ].copy()
        x_fold_train = fold_train[SELECTED_FEATURES].copy()
        y_fold_train = fold_train[TARGET].copy()
        x_fold_validation = fold_validation[SELECTED_FEATURES].copy()
        y_fold_validation = fold_validation[TARGET].copy()

        print(
            f"Fold {fold}: train {fold_train_dates[0]}–{fold_train_dates[-1]}, "
            f"validation {fold_validation_dates[0]}–{fold_validation_dates[-1]}"
        )
        for model_name, model in create_models().items():
            if model_name == "Linear Regression":
                fold_scaler = StandardScaler()
                fit_x = fold_scaler.fit_transform(x_fold_train)
                predict_x = fold_scaler.transform(x_fold_validation)
            else:
                fit_x = x_fold_train
                predict_x = x_fold_validation
            model.fit(fit_x, y_fold_train)
            metrics = evaluate_predictions(
                y_fold_validation, model.predict(predict_x)
            )
            fold_results.append(
                {
                    "Model": model_name,
                    "Fold": fold,
                    "Train Start Date": _json_safe_date(fold_train_dates[0]),
                    "Train End Date": _json_safe_date(fold_train_dates[-1]),
                    "Validation Start Date": _json_safe_date(
                        fold_validation_dates[0]
                    ),
                    "Validation End Date": _json_safe_date(
                        fold_validation_dates[-1]
                    ),
                    "Training Records": len(fold_train),
                    "Validation Records": len(fold_validation),
                    **metrics,
                }
            )

    fold_df = pd.DataFrame(fold_results)
    summary_df = (
        fold_df.groupby("Model", sort=False)
        .agg(
            **{
                "Mean MAE": ("MAE", "mean"),
                "Std MAE": ("MAE", "std"),
                "Mean RMSE": ("RMSE", "mean"),
                "Std RMSE": ("RMSE", "std"),
                "Mean R2": ("R2 Score", "mean"),
                "Std R2": ("R2 Score", "std"),
            }
        )
        .reset_index()
    )
    print("\nCross-validation summary:")
    print(summary_df.to_string(index=False))
    return fold_df, summary_df


def train_final_models(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray], StandardScaler]:
    """Fit final models on training dates and evaluate once on held-out dates."""
    x_train = train_df[SELECTED_FEATURES].copy()
    y_train = train_df[TARGET].copy()
    x_test = test_df[SELECTED_FEATURES].copy()
    y_test = test_df[TARGET].copy()
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    models = create_models()
    predictions: dict[str, np.ndarray] = {}
    results: list[dict[str, Any]] = []

    for model_name, model in models.items():
        if model_name == "Linear Regression":
            model.fit(x_train_scaled, y_train)
            model_predictions = model.predict(x_test_scaled)
        else:
            model.fit(x_train, y_train)
            model_predictions = model.predict(x_test)
        predictions[model_name] = model_predictions
        results.append(
            {"Model": model_name, **evaluate_predictions(y_test, model_predictions)}
        )

    results_df = pd.DataFrame(results).sort_values(
        by="R2 Score", ascending=False
    ).reset_index(drop=True)
    print("\n=== FINAL CHRONOLOGICAL HOLD-OUT RESULTS ===")
    print(results_df.to_string(index=False))
    return results_df, models, predictions, scaler


def generate_figures(
    results_df: pd.DataFrame,
    random_forest: RandomForestRegressor,
    y_test: pd.Series,
    rf_predictions: np.ndarray,
) -> None:
    """Generate publication-readable figures from final hold-out predictions."""
    sns.set_theme(style="whitegrid", context="paper")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    sns.barplot(data=results_df, x="Model", y="R2 Score", ax=ax)
    ax.set_title("Final Chronological Hold-out Performance")
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(
        FIGURES_DIR / "model_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    feature_importance = pd.DataFrame(
        {
            "Feature": SELECTED_FEATURES,
            "Importance": random_forest.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    sns.barplot(
        data=feature_importance.head(10), x="Importance", y="Feature", ax=ax
    )
    ax.set_title("Random Forest Feature Importance")
    fig.savefig(
        FIGURES_DIR / "feature_importance.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(y_test, rf_predictions, alpha=0.45, s=12)
    lower = float(min(y_test.min(), rf_predictions.min()))
    upper = float(max(y_test.max(), rf_predictions.max()))
    ax.plot([lower, upper], [lower, upper], "r--", linewidth=1.2, label="y = x")
    ax.set_xlabel("Actual Food Waste (kg)")
    ax.set_ylabel("Predicted Food Waste (kg)")
    ax.set_title("Actual vs. Predicted (Final Test Set)")
    ax.legend()
    fig.savefig(
        FIGURES_DIR / "actual_vs_predicted.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def save_results(
    split_summary: dict[str, Any],
    results_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    cv_summary_df: pd.DataFrame,
    models: dict[str, Any],
    scaler: StandardScaler,
) -> None:
    """Persist result tables, metadata, fitted models, and feature definitions."""
    results_df.to_csv(REPORTS_DIR / "model_results.csv", index=False)
    fold_df.to_csv(REPORTS_DIR / "time_series_cv_fold_results.csv", index=False)
    cv_summary_df.to_csv(
        REPORTS_DIR / "time_series_cv_summary.csv", index=False
    )
    with (REPORTS_DIR / "chronological_split_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(split_summary, file, indent=2)

    joblib.dump(models["Random Forest"], MODELS_DIR / "random_forest.pkl")
    joblib.dump(models["Gradient Boosting"], MODELS_DIR / "gradient_boosting.pkl")
    joblib.dump(models["Linear Regression"], MODELS_DIR / "linear_regression.pkl")
    joblib.dump(models["Decision Tree"], MODELS_DIR / "decision_tree.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(SELECTED_FEATURES, MODELS_DIR / "selected_features.pkl")
    joblib.dump(split_summary, MODELS_DIR / "split_metadata.pkl")


def main() -> None:
    """Execute the chronological training and evaluation pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATA_PATH}")
    df = load_and_sort_data()
    print(f"Dataset shape: {df.shape}")
    train_df, test_df, train_dates, test_dates = create_chronological_split(df)
    split_summary = validate_split(
        df, train_df, test_df, train_dates, test_dates
    )
    fold_df, cv_summary_df = run_time_series_cross_validation(train_df)
    results_df, models, predictions, scaler = train_final_models(train_df, test_df)
    generate_figures(
        results_df,
        models["Random Forest"],
        test_df[TARGET],
        predictions["Random Forest"],
    )
    save_results(
        split_summary, results_df, fold_df, cv_summary_df, models, scaler
    )
    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()
