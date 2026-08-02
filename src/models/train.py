"""Train and evaluate restaurant food-waste regressors chronologically."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.construct_target import generate_target_dataset


CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SELECTED_FEATURES = [
    "year", "month", "week", "day_of_week", "restaurant_id", "city_code",
    "region_code", "center_type", "op_area", "unique_meals",
    "dominant_category", "dominant_cuisine", "avg_checkout_price",
    "avg_base_price", "emailer_promo_rate", "homepage_feature_rate",
    "is_weekend", "is_holiday", "temperature_c",
]
FORBIDDEN_REALISTIC_FEATURES = {
    "food_prepared_kg", "food_sold_kg", "num_orders", "food_waste_kg_original",
    "food_waste_kg_base", "gaussian_noise_kg", "consumption_ratio", "food_waste_kg",
}
LEAKAGE_FEATURES = ["food_prepared_kg", "food_sold_kg", "num_orders"]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    configured_features = config["training"]["feature_columns"]
    if configured_features != SELECTED_FEATURES:
        raise ValueError("Configured realistic features differ from the required feature list.")
    leaked = sorted(set(configured_features) & FORBIDDEN_REALISTIC_FEATURES)
    if leaked:
        raise ValueError(f"Target/leakage columns entered realistic features: {leaked}")
    if config["training"]["split_strategy"] != "chronological":
        raise ValueError("Only the chronological split strategy is permitted.")
    if config["training"]["cv_method"] != "time_series_split":
        raise ValueError("Only expanding-window TimeSeriesSplit is permitted.")
    return config


def load_and_sort_data(path: Path, date_column: str = "date") -> pd.DataFrame:
    df = pd.read_csv(path)
    if not pd.api.types.is_numeric_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors="raise")
    return df.sort_values([date_column, "restaurant_id"], kind="mergesort").reset_index(drop=True)


def create_chronological_split(
    df: pd.DataFrame, train_ratio: float = 0.70, date_column: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    unique_dates = np.sort(df[date_column].unique())
    split_index = int(len(unique_dates) * train_ratio)
    if split_index <= 0 or split_index >= len(unique_dates):
        raise ValueError("The dataset needs enough unique dates for a non-empty split.")
    train_dates, test_dates = unique_dates[:split_index], unique_dates[split_index:]
    return (
        df[df[date_column].isin(train_dates)].copy(),
        df[df[date_column].isin(test_dates)].copy(),
        train_dates,
        test_dates,
    )


def _json_safe_date(value: Any) -> int | float | str:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return str(value)


def validate_split(
    df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame,
    train_dates: np.ndarray, test_dates: np.ndarray, train_ratio: float = 0.70,
    date_column: str = "date",
) -> dict[str, Any]:
    overlap = len(set(train_dates).intersection(test_dates))
    assert overlap == 0
    assert train_df[date_column].max() < test_df[date_column].min()
    summary = {
        "split_strategy": "chronological",
        "training_ratio": float(train_ratio),
        "testing_ratio": float(1.0 - train_ratio),
        "total_records": len(df),
        "training_records": len(train_df),
        "testing_records": len(test_df),
        "total_unique_dates": int(df[date_column].nunique()),
        "training_unique_dates": len(train_dates),
        "testing_unique_dates": len(test_dates),
        "training_start_date": _json_safe_date(train_dates[0]),
        "training_end_date": _json_safe_date(train_dates[-1]),
        "testing_start_date": _json_safe_date(test_dates[0]),
        "testing_end_date": _json_safe_date(test_dates[-1]),
        "split_date": _json_safe_date(test_dates[0]),
        "overlapping_dates": overlap,
    }
    print("\n=== CHRONOLOGICAL SPLIT SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


def create_models() -> dict[str, Any]:
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
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "R2 Score": float(r2_score(y_true, predictions)),
    }


def run_time_series_cross_validation(
    train_df: pd.DataFrame, features: list[str] = SELECTED_FEATURES,
    target: str = "food_waste_kg", date_column: str = "date", cv_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.sort(train_df[date_column].unique())
    fold_results: list[dict[str, Any]] = []
    print(f"\n=== {cv_splits}-FOLD EXPANDING-WINDOW CROSS-VALIDATION ===")
    for fold, (train_indices, validation_indices) in enumerate(
        TimeSeriesSplit(n_splits=cv_splits).split(unique_dates), start=1
    ):
        fold_train_dates = unique_dates[train_indices]
        fold_validation_dates = unique_dates[validation_indices]
        assert fold_train_dates.max() < fold_validation_dates.min()
        assert not set(fold_train_dates).intersection(fold_validation_dates)
        fold_train = train_df[train_df[date_column].isin(fold_train_dates)]
        fold_validation = train_df[train_df[date_column].isin(fold_validation_dates)]
        x_train, y_train = fold_train[features], fold_train[target]
        x_validation, y_validation = fold_validation[features], fold_validation[target]
        for model_name, model in create_models().items():
            if model_name == "Linear Regression":
                scaler = StandardScaler()
                fit_x = scaler.fit_transform(x_train)
                predict_x = scaler.transform(x_validation)
            else:
                fit_x, predict_x = x_train, x_validation
            model.fit(fit_x, y_train)
            metrics = evaluate_predictions(y_validation, model.predict(predict_x))
            fold_results.append({
                "Model": model_name, "Fold": fold,
                "Train Start Date": _json_safe_date(fold_train_dates[0]),
                "Train End Date": _json_safe_date(fold_train_dates[-1]),
                "Validation Start Date": _json_safe_date(fold_validation_dates[0]),
                "Validation End Date": _json_safe_date(fold_validation_dates[-1]),
                "Training Records": len(fold_train),
                "Validation Records": len(fold_validation), **metrics,
            })
    fold_df = pd.DataFrame(fold_results)
    summary_df = fold_df.groupby("Model", sort=False).agg(**{
        "Mean MAE": ("MAE", "mean"), "Std MAE": ("MAE", "std"),
        "Mean RMSE": ("RMSE", "mean"), "Std RMSE": ("RMSE", "std"),
        "Mean R2": ("R2 Score", "mean"), "Std R2": ("R2 Score", "std"),
    }).reset_index()
    print(summary_df.to_string(index=False))
    return fold_df, summary_df


def train_final_models(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    features: list[str] = SELECTED_FEATURES, target: str = "food_waste_kg",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray], StandardScaler]:
    x_train, y_train = train_df[features], train_df[target]
    x_test, y_test = test_df[features], test_df[target]
    scaler = StandardScaler()
    x_train_scaled, x_test_scaled = scaler.fit_transform(x_train), scaler.transform(x_test)
    models, predictions, rows = create_models(), {}, []
    for name, model in models.items():
        if name == "Linear Regression":
            model.fit(x_train_scaled, y_train)
            predictions[name] = model.predict(x_test_scaled)
        else:
            model.fit(x_train, y_train)
            predictions[name] = model.predict(x_test)
        rows.append({"Model": name, **evaluate_predictions(y_test, predictions[name])})
    results = pd.DataFrame(rows).sort_values("R2 Score", ascending=False).reset_index(drop=True)
    print("\n=== FINAL CHRONOLOGICAL HOLD-OUT RESULTS ===")
    print(results.to_string(index=False))
    return results, models, predictions, scaler


def run_ablation_study(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target: str = "food_waste_kg"
) -> pd.DataFrame:
    configurations = [
        ("Configuration 1", SELECTED_FEATURES + LEAKAGE_FEATURES, LEAKAGE_FEATURES, []),
        ("Configuration 2", SELECTED_FEATURES + ["num_orders"], ["num_orders"], ["food_prepared_kg", "food_sold_kg"]),
        ("Configuration 3", SELECTED_FEATURES, [], LEAKAGE_FEATURES),
    ]
    rows = []
    for name, features, included, excluded in configurations:
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(train_df[features], train_df[target])
        metrics = evaluate_predictions(test_df[target], model.predict(test_df[features]))
        rows.append({
            "Configuration": name,
            "Included Leakage Features": ", ".join(included) if included else "None",
            "Excluded Leakage Features": ", ".join(excluded) if excluded else "None",
            "Training Records": len(train_df), "Testing Records": len(test_df), **metrics,
        })
    result = pd.DataFrame(rows)
    print("\n=== LEAKAGE ABLATION RESULTS ===")
    print(result.to_string(index=False))
    return result


def generate_figures(
    results: pd.DataFrame, random_forest: RandomForestRegressor,
    y_test: pd.Series, predictions: np.ndarray, gaussian_df: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    sns.barplot(data=results, x="Model", y="R2 Score", ax=ax)
    ax.set_title("Final Chronological Hold-out Performance")
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    importance = pd.DataFrame({
        "Feature": SELECTED_FEATURES, "Importance": random_forest.feature_importances_
    }).sort_values("Importance", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    sns.barplot(data=importance, x="Importance", y="Feature", ax=ax)
    ax.set_title("Random Forest Feature Importance")
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(y_test, predictions, alpha=0.45, s=12)
    lower, upper = float(min(y_test.min(), predictions.min())), float(max(y_test.max(), predictions.max()))
    ax.plot([lower, upper], [lower, upper], "r--", linewidth=1.2, label="y = x")
    ax.set(xlabel="Actual Food Waste (kg)", ylabel="Predicted Food Waste (kg)", title="Actual vs. Predicted (Final Test Set)")
    ax.legend()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    correlation_columns = SELECTED_FEATURES + ["food_waste_kg"]
    correlation = gaussian_df[correlation_columns].corr()
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    sns.heatmap(correlation, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlations with Gaussian Proxy Target")
    ax.tick_params(axis="x", labelrotation=70, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_results(
    split_summary: dict[str, Any], results: pd.DataFrame, folds: pd.DataFrame,
    cv_summary: pd.DataFrame, ablation: pd.DataFrame, models: dict[str, Any],
    scaler: StandardScaler,
) -> None:
    # Parallel tree prediction can vary at the final IEEE-754 bit during summation.
    # Canonical persistence at 12 decimal places makes result artifacts byte-stable
    # while retaining substantially more precision than any reported table needs.
    results.round(12).to_csv(REPORTS_DIR / "model_results.csv", index=False)
    folds.round(12).to_csv(REPORTS_DIR / "time_series_cv_fold_results.csv", index=False)
    cv_summary.round(12).to_csv(REPORTS_DIR / "time_series_cv_summary.csv", index=False)
    ablation.round(12).to_csv(REPORTS_DIR / "ablation_results.csv", index=False)
    with (REPORTS_DIR / "chronological_split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(split_summary, file, indent=2)
    for name, filename in [
        ("Random Forest", "random_forest.pkl"), ("Gradient Boosting", "gradient_boosting.pkl"),
        ("Linear Regression", "linear_regression.pkl"), ("Decision Tree", "decision_tree.pkl"),
    ]:
        joblib.dump(models[name], MODELS_DIR / filename)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(SELECTED_FEATURES, MODELS_DIR / "selected_features.pkl")
    joblib.dump(split_summary, MODELS_DIR / "split_metadata.pkl")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    training = config["training"]
    if not config["target_construction"]["enabled"]:
        raise ValueError("Gaussian target construction must be enabled.")
    generate_target_dataset(CONFIG_PATH)
    data_path = PROJECT_ROOT / config["target_construction"]["generated_dataset"]
    df = load_and_sort_data(data_path, training["date_column"])
    train_df, test_df, train_dates, test_dates = create_chronological_split(
        df, float(training["train_ratio"]), training["date_column"]
    )
    split = validate_split(df, train_df, test_df, train_dates, test_dates,
                           float(training["train_ratio"]), training["date_column"])
    folds, cv_summary = run_time_series_cross_validation(
        train_df, SELECTED_FEATURES, training["target"], training["date_column"], int(training["cv_splits"])
    )
    results, models, predictions, scaler = train_final_models(
        train_df, test_df, SELECTED_FEATURES, training["target"]
    )
    ablation = run_ablation_study(train_df, test_df, training["target"])
    generate_figures(results, models["Random Forest"], test_df[training["target"]],
                     predictions["Random Forest"], df)
    save_results(split, results, folds, cv_summary, ablation, models, scaler)
    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()
