# tests/test_preprocessing.py
"""
Tests for preprocessing utilities.

The test suite verifies stable behavior of helper functions used during
feature preparation and preprocessing:
- correct separation of numeric and categorical features
- proper construction of X/y matrices for train and test datasets
- consistent behavior of the one-hot encoding preprocessor during fit/transform
"""

import pandas as pd
import numpy as np

from src.data.preprocessing import split_features, make_xy, build_preprocessor_ohe
from src.utils.config import ID_COL, TARGET_COL


def test_split_features_separates_numeric_and_categorical():
    """
    The test checks that numeric and categorical features are separated correctly.

    Identifier and target columns are expected to be excluded from both feature groups
    """

    df = pd.DataFrame(
        {
            ID_COL: [1, 2, 3],
            TARGET_COL: [0, 1, 0],
            "age": [20, 30, 40],          # numeric
            "income": [100.0, 200.5, 150.2],  # numeric
            "city": ["A", "B", "A"],      # categorical
        }
    )

    spec = split_features(df)

    assert "age" in spec.numeric
    assert "income" in spec.numeric
    assert "city" in spec.categorical

    # Identifier and target columns must not be treated as features
    assert ID_COL not in spec.numeric and ID_COL not in spec.categorical
    assert TARGET_COL not in spec.numeric and TARGET_COL not in spec.categorical


def test_make_xy_splits_train_and_test_correctly():
    """
    The test verifies correct construction of X_train, y_train and X_test.

    Identifier and target columns are expected to be removed from feature matrices,
    while the target vector is returned separately
    """

    train_df = pd.DataFrame(
        {
            ID_COL: [1, 2],
            TARGET_COL: [0, 1],
            "age": [20, 30],
            "city": ["A", "B"],
        }
    )

    test_df = pd.DataFrame(
        {
            ID_COL: [10, 11],
            "age": [25, 35],
            "city": ["A", "C"],
        }
    )

    x_train, y, x_test = make_xy(train_df, test_df)

    # Target vector is expected to be numeric
    assert list(y.values) == [0.0, 1.0]

    # Training features must not include identifier or target columns
    assert ID_COL not in x_train.columns
    assert TARGET_COL not in x_train.columns

    # Test features must not include identifier column
    assert ID_COL not in x_test.columns

    # Feature columns should remain unchanged
    assert set(x_train.columns) == {"age", "city"}
    assert set(x_test.columns) == {"age", "city"}

    # Output shapes are expected to match input sizes
    assert x_train.shape == (2, 2)
    assert x_test.shape == (2, 2)


def test_build_preprocessor_ohe_fit_transform():
    """
    The test checks that the OHE preprocessor can be fitted and applied consistently.

    Unseen categorical values during transform are expected to be handled without errors,
    and output dimensionality should remain stable
    """

    df_train = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["A", "B", "A"],
        }
    )

    df_test = pd.DataFrame(
        {
            "age": [25, 35],
            "city": ["A", "C"],  # unseen category
        }
    )

    categorical = ["city"]

    preprocessor = build_preprocessor_ohe(categorical)

    x_train = preprocessor.fit_transform(df_train)
    x_test = preprocessor.transform(df_test)

    # Preprocessor outputs are expected to be numpy arrays
    assert isinstance(x_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)

    # Number of features should remain consistent after transform
    assert x_train.shape[1] == x_test.shape[1]

    # Number of rows should match the input data
    assert x_train.shape[0] == len(df_train)
    assert x_test.shape[0] == len(df_test)
