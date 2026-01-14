# tests/test_drift.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift import psi_numeric, psi_categorical, build_feature_drift_report


def test_psi_numeric_small_when_same_distribution():
    """
    Verify that PSI for numeric features remains small
    when reference and current distributions are nearly identical.
    """
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(loc = 0.0, scale = 1.0, size = 5000))
    cur = pd.Series(rng.normal(loc = 0.0, scale = 1.0, size = 5000))

    psi, info = psi_numeric(ref, cur, n_bins = 10, include_missing_bin = True)
    assert 0.0 <= psi < 0.05  # should be small for very similar distributions
    assert "ref_missing_rate" in info


def test_psi_numeric_large_when_shifted():
    """
    Verify that PSI increases when the numeric distribution
    is significantly shifted between reference and current data.
    """
    rng = np.random.default_rng(123)
    ref = pd.Series(rng.normal(loc = 0.0, scale = 1.0, size = 5000))
    cur = pd.Series(rng.normal(loc = 2.0, scale = 1.0, size = 5000))  # strong shift

    psi, _ = psi_numeric(ref, cur, n_bins = 10, include_missing_bin = True)
    assert psi > 0.10


def test_psi_categorical_detects_distribution_change_and_new_category():
    """
    Verify that categorical PSI detects both
    distribution changes and the appearance of new categories.
    """
    ref = pd.Series(["A"] * 900 + ["B"] * 100)
    cur = pd.Series(["A"] * 500 + ["B"] * 200 + ["C"] * 300)  # new category C

    psi, info = psi_categorical(ref, cur, include_missing_as_category = True)
    assert psi > 0.10
    assert info["ref_unique"] >= 2
    assert info["cur_unique"] >= 3


def test_build_feature_drift_report_smoke():
    """
    Smoke test for the end-to-end feature drift report builder.

    Ensures that:
        - both numeric and categorical features are included,
        - required output columns are present,
        - the function runs without errors on a minimal dataset.
    """
    x_ref = pd.DataFrame(
        {
            "num": [0, 0, 0, 1, 1, 1],
            "cat": ["A", "A", "B", "B", "B", "B"],
        }
    )
    x_cur = pd.DataFrame(
        {
            "num": [10, 10, 10, 11, 11, 11],
            "cat": ["A", "C", "C", "C", "C", "C"],
        }
    )

    df = build_feature_drift_report(
        x_ref = x_ref,
        x_cur = x_cur,
        numeric_features = ["num"],
        categorical_features = ["cat"],
        n_bins_numeric = 3,
    )

    assert set(df["feature"]) == {"num", "cat"}
    assert "psi" in df.columns
    assert "status" in df.columns
