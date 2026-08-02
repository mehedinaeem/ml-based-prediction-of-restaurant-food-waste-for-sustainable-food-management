"""Construct the reproducible Gaussian proxy target used by the experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"
REQUIRED_COLUMNS = [
    "date",
    "restaurant_id",
    "food_prepared_kg",
    "food_sold_kg",
    "food_waste_kg",
]
AUDIT_COLUMNS = [
    "food_waste_kg_original",
    "consumption_ratio",
    "food_waste_kg_base",
    "gaussian_noise_kg",
    "food_waste_kg",
]


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars into values accepted by the JSON encoder."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def construct_gaussian_target(
    df: pd.DataFrame,
    alpha: float,
    beta: float,
    sigma_kg: float,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a sorted copy with one reproducible Gaussian target realization."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if sigma_kg < 0:
        raise ValueError("sigma_kg must be non-negative.")

    required = df[REQUIRED_COLUMNS]
    if required.isna().any().any():
        bad_columns = required.columns[required.isna().any()].tolist()
        raise ValueError(f"Missing values found in required columns: {bad_columns}")
    numeric_columns = REQUIRED_COLUMNS[1:]
    non_finite = ~np.isfinite(df[numeric_columns].to_numpy(dtype=float))
    if non_finite.any():
        rows = np.unique(np.where(non_finite)[0]).tolist()
        raise ValueError(f"Infinite values found in required numeric columns at rows: {rows[:20]}")
    if (df["food_prepared_kg"] <= 0).any():
        raise ValueError("food_prepared_kg must be greater than zero for every record.")
    if (df["food_sold_kg"] < 0).any():
        raise ValueError("food_sold_kg must be non-negative for every record.")
    if (df["food_sold_kg"] > df["food_prepared_kg"]).any():
        raise ValueError("food_sold_kg cannot exceed food_prepared_kg.")

    result = df.sort_values(
        ["date", "restaurant_id"], kind="mergesort"
    ).reset_index(drop=True).copy()
    result["food_waste_kg_original"] = result["food_waste_kg"]
    result["consumption_ratio"] = (
        result["food_sold_kg"] / result["food_prepared_kg"]
    )
    result["food_waste_kg_base"] = (
        alpha
        * result["food_prepared_kg"]
        * (1.0 - beta * result["consumption_ratio"])
    )

    deterministic_target = result["food_prepared_kg"] - result["food_sold_kg"]
    if not np.allclose(
        result["food_waste_kg_original"], deterministic_target, atol=1e-8, rtol=1e-8
    ):
        maximum_difference = float(
            np.max(np.abs(result["food_waste_kg_original"] - deterministic_target))
        )
        raise ValueError(
            "Original target is inconsistent with food_prepared_kg - food_sold_kg; "
            f"maximum difference is {maximum_difference:.12g}."
        )
    if not np.allclose(
        result["food_waste_kg_base"],
        result["food_waste_kg_original"],
        atol=1e-8,
        rtol=1e-8,
    ):
        maximum_difference = float(
            np.max(
                np.abs(
                    result["food_waste_kg_base"]
                    - result["food_waste_kg_original"]
                )
            )
        )
        raise ValueError(
            "Noise-free target is inconsistent with the original target; "
            f"maximum difference is {maximum_difference:.12g}. Check alpha and beta."
        )

    rng = np.random.default_rng(random_seed)
    result["gaussian_noise_kg"] = rng.normal(
        loc=0.0, scale=sigma_kg, size=len(result)
    )
    result["food_waste_kg"] = (
        result["food_waste_kg_base"] + result["gaussian_noise_kg"]
    )
    negative_mask = result["food_waste_kg"] < 0
    if negative_mask.any():
        affected = result.loc[
            negative_mask,
            ["date", "restaurant_id", *AUDIT_COLUMNS],
        ]
        raise AssertionError(
            "Gaussian construction produced negative targets; no clipping was applied. "
            f"Affected records:\n{affected.to_string(index=False)}"
        )

    metadata = {
        "formula": "W_i = alpha * P_i * (1 - beta * C_i) + epsilon_i",
        "alpha": float(alpha),
        "beta": float(beta),
        "sigma_kg": float(sigma_kg),
        "random_seed": int(random_seed),
        "total_records": int(len(result)),
        "noise_mean": float(result["gaussian_noise_kg"].mean()),
        "noise_standard_deviation": float(result["gaussian_noise_kg"].std(ddof=0)),
        "noise_minimum": float(result["gaussian_noise_kg"].min()),
        "noise_maximum": float(result["gaussian_noise_kg"].max()),
        "base_target_mean": float(result["food_waste_kg_base"].mean()),
        "base_target_standard_deviation": float(
            result["food_waste_kg_base"].std(ddof=0)
        ),
        "final_target_mean": float(result["food_waste_kg"].mean()),
        "final_target_standard_deviation": float(
            result["food_waste_kg"].std(ddof=0)
        ),
        "final_target_minimum": float(result["food_waste_kg"].min()),
        "final_target_maximum": float(result["food_waste_kg"].max()),
        "negative_target_count": int(negative_mask.sum()),
        "maximum_difference_between_original_and_base_target": float(
            np.max(
                np.abs(
                    result["food_waste_kg_original"]
                    - result["food_waste_kg_base"]
                )
            )
        ),
    }
    return result, {key: _json_safe(value) for key, value in metadata.items()}


def generate_target_dataset(config_path: Path = CONFIG_PATH) -> tuple[pd.DataFrame, dict]:
    """Build and save the configured dataset and its construction metadata."""
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    settings = config["target_construction"]
    source_path = PROJECT_ROOT / settings["source_dataset"]
    generated_path = PROJECT_ROOT / settings["generated_dataset"]
    metadata_path = PROJECT_ROOT / "reports" / "target_construction_metadata.json"

    source_df = pd.read_csv(source_path)
    generated_df, metadata = construct_gaussian_target(
        source_df,
        alpha=float(settings["alpha"]),
        beta=float(settings["beta"]),
        sigma_kg=float(settings["sigma_kg"]),
        random_seed=int(settings["random_seed"]),
    )
    metadata.update(
        {
            "source_dataset": settings["source_dataset"],
            "generated_dataset": settings["generated_dataset"],
            "prepared_column": settings["prepared_column"],
            "sold_column": settings["sold_column"],
            "original_target_column": settings["original_target_column"],
            "final_target_column": settings["generated_target_column"],
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "NumPy version": np.__version__,
            "Pandas version": pd.__version__,
        }
    )
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    generated_df.to_csv(generated_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)

    print("\n=== GAUSSIAN TARGET CONSTRUCTION SUMMARY ===")
    print(f"Formula: {metadata['formula']}")
    print(
        f"alpha={metadata['alpha']}, beta={metadata['beta']}, "
        f"sigma={metadata['sigma_kg']} kg, seed={metadata['random_seed']}"
    )
    print(f"Records: {metadata['total_records']}")
    print(f"Noise mean: {metadata['noise_mean']:.12f} kg")
    print(f"Noise standard deviation (population): {metadata['noise_standard_deviation']:.12f} kg")
    print(f"Maximum original/base difference: {metadata['maximum_difference_between_original_and_base_target']:.12g} kg")
    print(f"Negative target count: {metadata['negative_target_count']}")
    print(f"Saved dataset: {generated_path}")
    print(f"Saved metadata: {metadata_path}")
    return generated_df, metadata


if __name__ == "__main__":
    generate_target_dataset()
