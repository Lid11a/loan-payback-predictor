# src/data/preprocessing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.utils.config import ID_COL, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureSpec:
    """
    Container for numeric and categorical feature names.
    """

    numeric: List[str]
    categorical: List[str]


def split_features(df: pd.DataFrame) -> FeatureSpec:
    """
    The function splits dataset columns into numeric and categorical features.

    Identifier and target columns are excluded from the feature set.
    The resulting feature groups follow the same logic as used during EDA.

    Returns a FeatureSpec object containing lists of numeric and categorical
    feature names.
    """

    x = df.drop(columns=[ID_COL, TARGET_COL], errors="ignore")

    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = x.select_dtypes(include="object").columns.tolist()

    logger.info(
        "Features split. numeric=%s categorical=%s",
        len(numeric),
        len(categorical),
    )

    return FeatureSpec(numeric=numeric, categorical=categorical)


def make_xy(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    The function prepares feature matrices and target vector for model training
    and inference.

    The target and identifier columns are removed from the feature matrices.
    The target vector is returned separately.

    Returns X_train, y_train, and X_test.
    """

    y = train_df[TARGET_COL].astype(float)

    x_train = train_df.drop(columns=[ID_COL, TARGET_COL])
    x_test = test_df.drop(columns=[ID_COL])

    logger.info("Prepared matrices. x_train=%s x_test=%s", x_train.shape, x_test.shape)

    return x_train, y, x_test


def build_preprocessor_ohe(categorical_features: List[str]) -> ColumnTransformer:
    """
    The function builds a preprocessing pipeline for model training and inference.

    Categorical features are encoded using one-hot encoding, while numeric
    features are passed through without modification.

    Returns a fitted ColumnTransformer configuration.
    """

    logger.info("Building OHE preprocessor. categorical_features=%s", len(categorical_features))

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="passthrough",
        n_jobs=-1,
    )
