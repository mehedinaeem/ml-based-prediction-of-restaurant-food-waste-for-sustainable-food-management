import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_feature_importance(importances, feature_names, output_path=None):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importances")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()


def plot_actual_vs_predicted(actual: pd.Series, predicted: pd.Series, output_path=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], color="red")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()
