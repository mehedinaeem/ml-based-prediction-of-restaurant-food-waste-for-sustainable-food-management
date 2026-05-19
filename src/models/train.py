import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import os

import warnings
warnings.filterwarnings("ignore")


# =====================================================
# CREATE REQUIRED DIRECTORIES
# =====================================================

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


# =====================================================
# LOAD DATASET
# =====================================================

print("\nLoading Dataset...\n")

df = pd.read_csv(
    "data/processed/eda_processed_dataset.csv"
)

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# =====================================================
# FEATURE SELECTION
# =====================================================

print("\nUsing Realistic Feature Selection...\n")

selected_features = [
    'year',
    'month',
    'week',
    'day_of_week',
    'restaurant_id',
    'city_code',
    'region_code',
    'center_type',
    'op_area',
    'unique_meals',
    'dominant_category',
    'dominant_cuisine',
    'avg_checkout_price',
    'avg_base_price',
    'emailer_promo_rate',
    'homepage_feature_rate',
    'is_weekend',
    'is_holiday',
    'temperature_c'
]

print("Selected Features:\n")

for feature in selected_features:
    print("-", feature)

X = df[selected_features]

y = df['food_waste_kg']

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# =====================================================
# TRAIN TEST SPLIT (80/20)
# =====================================================

print("\nSplitting Dataset Using 80/20 Rule...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# =====================================================
# FEATURE SCALING
# =====================================================

print("\nScaling Features...\n")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =====================================================
# STORE RESULTS
# =====================================================

results = []


# =====================================================
# LINEAR REGRESSION
# =====================================================

print("\nTraining Linear Regression...\n")

linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_train
)

linear_predictions = linear_model.predict(
    X_test_scaled
)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

results.append([
    "Linear Regression",
    linear_mae,
    linear_rmse,
    linear_r2
])

print("Linear Regression Completed")


# =====================================================
# DECISION TREE
# =====================================================

print("\nTraining Decision Tree...\n")

tree_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=8
)

tree_model.fit(
    X_train,
    y_train
)

tree_predictions = tree_model.predict(
    X_test
)

tree_mae = mean_absolute_error(
    y_test,
    tree_predictions
)

tree_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tree_predictions
    )
)

tree_r2 = r2_score(
    y_test,
    tree_predictions
)

results.append([
    "Decision Tree",
    tree_mae,
    tree_rmse,
    tree_r2
])

print("Decision Tree Completed")


# =====================================================
# RANDOM FOREST
# =====================================================

print("\nTraining Random Forest...\n")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(
    X_test
)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

results.append([
    "Random Forest",
    rf_mae,
    rf_rmse,
    rf_r2
])

print("Random Forest Completed")


# =====================================================
# GRADIENT BOOSTING
# =====================================================

print("\nTraining Gradient Boosting...\n")

gb_model = GradientBoostingRegressor(
    random_state=42,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3
)

gb_model.fit(
    X_train,
    y_train
)

gb_predictions = gb_model.predict(
    X_test
)

gb_mae = mean_absolute_error(
    y_test,
    gb_predictions
)

gb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        gb_predictions
    )
)

gb_r2 = r2_score(
    y_test,
    gb_predictions
)

results.append([
    "Gradient Boosting",
    gb_mae,
    gb_rmse,
    gb_r2
])

print("Gradient Boosting Completed")


# =====================================================
# RESULTS DATAFRAME
# =====================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2 Score"
    ]
)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n==============================")
print("MODEL PERFORMANCE RESULTS")
print("==============================\n")

print(results_df)


# =====================================================
# SAVE RESULTS
# =====================================================

results_df.to_csv(
    "reports/model_results.csv",
    index=False
)

print("\nResults Saved Successfully")


# =====================================================
# SAVE MODELS
# =====================================================

joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)

joblib.dump(
    gb_model,
    "models/gradient_boosting.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nModels Saved Successfully")


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop Important Features:\n")

print(feature_importance.head(10))


# =====================================================
# FEATURE IMPORTANCE PLOT
# =====================================================

plt.figure(figsize=(12,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance.head(10)
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(
    "reports/figures/feature_importance.png"
)

plt.show()


# =====================================================
# ACTUAL VS PREDICTED PLOT
# =====================================================

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.5
)

plt.xlabel("Actual Food Waste")
plt.ylabel("Predicted Food Waste")

plt.title("Actual vs Predicted Food Waste")

plt.tight_layout()

plt.savefig(
    "reports/figures/actual_vs_predicted.png"
)

plt.show()


# =====================================================
# MODEL COMPARISON PLOT
# =====================================================

plt.figure(figsize=(10,6))

sns.barplot(
    x='Model',
    y='R2 Score',
    data=results_df
)

plt.title("Model Performance Comparison")

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig(
    "reports/figures/model_comparison.png"
)

plt.show()


# =====================================================
# BEST MODEL
# =====================================================

best_model = results_df.iloc[0]

print("\n==============================")
print("BEST MODEL")
print("==============================\n")

print(best_model)


# =====================================================
# TRAINING COMPLETED
# =====================================================

print("\nModel Training Pipeline Completed Successfully\n")