"""Regression tests for target construction and chronological isolation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.construct_target import AUDIT_COLUMNS, construct_gaussian_target
from src.models.train import FORBIDDEN_REALISTIC_FEATURES, SELECTED_FEATURES, create_chronological_split


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "eda_processed_dataset.csv"


def _source() -> pd.DataFrame:
    return pd.read_csv(SOURCE)


def _construct(seed: int = 42) -> pd.DataFrame:
    return construct_gaussian_target(_source(), 1.0, 1.0, 1.0, seed)[0]


def test_same_seed_produces_identical_noise() -> None:
    first, second = _construct(), _construct()
    np.testing.assert_array_equal(first["gaussian_noise_kg"], second["gaussian_noise_kg"])


def test_different_seed_produces_different_noise() -> None:
    assert not np.array_equal(_construct(42)["gaussian_noise_kg"], _construct(43)["gaussian_noise_kg"])


def test_base_original_and_generated_target_identities() -> None:
    generated = _construct()
    np.testing.assert_allclose(generated["food_waste_kg_base"], generated["food_prepared_kg"] - generated["food_sold_kg"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(generated["food_waste_kg_original"], generated["food_waste_kg_base"], atol=1e-8, rtol=1e-8)
    np.testing.assert_array_equal(generated["food_waste_kg"], generated["food_waste_kg_base"] + generated["gaussian_noise_kg"])


def test_no_negative_targets_for_required_parameters() -> None:
    assert (_construct()["food_waste_kg"] >= 0).all()


def test_realistic_features_exclude_target_construction_and_leakage() -> None:
    assert not set(SELECTED_FEATURES).intersection(FORBIDDEN_REALISTIC_FEATURES)


def test_construction_does_not_modify_original_file() -> None:
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    _construct()
    assert before == hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def test_generated_data_contains_all_audit_columns() -> None:
    assert set(AUDIT_COLUMNS).issubset(_construct().columns)


def test_chronological_dates_do_not_overlap() -> None:
    train, test, train_dates, test_dates = create_chronological_split(_construct())
    assert set(train_dates).isdisjoint(test_dates)
    assert train["date"].max() < test["date"].min()
