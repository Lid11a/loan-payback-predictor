# src/models/train_baseline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from src.data.load import load_kaggle_data
from src.data.preprocessing import make_xy, split_features
from src.utils.config import ID_COL, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelBundle:
    """
    A small container used to save a trained model together with basic metadata.

    model contains the fitted scikit-learn Pipeline (preprocessing + classifier).
    threshold stores the decision threshold used for turning probabilities into 0/1 labels.
    meta stores additional information about how the artifact was created.
    """

    model: Any
    threshold: float
    meta: Dict[str, Any]


def build_preprocessor_baseline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    The function builds a baseline preprocessing step for a linear model.

    Numeric features are standardized with StandardScaler.
    Categorical features are one-hot encoded with OneHotEncoder.
    The output is a single ColumnTransformer that produces a purely numeric feature matrix.
    """

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
        n_jobs=-1,
    )


def train_baseline(
    data_dir: str | Path = "data/raw",
    model_dir: str | Path = "models",
    artifact_name: str = "baseline_logreg.joblib",
    seed: int = 42,
) -> Path:
    """
    The function trains a baseline Logistic Regression model and saves it as a joblib artifact.

    The baseline model is a scikit-learn Pipeline consisting of:
    - preprocessing (scaling numeric features and one-hot encoding categorical features),
    - LogisticRegression classifier with class balancing.

    A simple default threshold of 0.5 is used for the baseline bundle.
    The function returns the path to the saved artifact file.
    """

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Baseline train started. data_dir=%s model_dir=%s artifact=%s seed=%s",
        data_dir,
        model_dir,
        artifact_name,
        seed,
    )

    train_df, test_df = load_kaggle_data(data_dir)
    x_train, y, _ = make_xy(train_df, test_df)

    spec = split_features(train_df)

    logger.info(
        "Baseline features. numeric=%s categorical=%s",
        len(spec.numeric),
        len(spec.categorical),
    )

    preprocessor = build_preprocessor_baseline(spec.numeric, spec.categorical)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=seed,
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    penalty="l2",
                    C=1.0,
                ),
            ),
        ]
    )

    logger.info("Fitting baseline LogisticRegression pipeline...")

    pipeline.fit(x_train, y)

    bundle = ModelBundle(
        model=pipeline,
        threshold=0.5,
        meta={
            "type": "baseline_logreg",
            "seed": seed,
            "target_col": TARGET_COL,
            "id_col": ID_COL,
        },
    )

    out_path = model_dir_path / artifact_name
    logger.info("Saving baseline bundle to: %s", out_path)
    joblib.dump(bundle, out_path)
    logger.info("Baseline train finished.")

    return out_path


if __name__ == "__main__":
    p = train_baseline()
    logger.info("Saved baseline: %s", p)

