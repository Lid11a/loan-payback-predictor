# src/models/predict.py
from __future__ import annotations

from pathlib import Path
import re

import joblib
import pandas as pd

from src.data.load import load_kaggle_data
from src.data.preprocessing import make_xy
from src.utils.config import ID_COL, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


def safe_name(s: str) -> str:
    """
    The function converts an arbitrary string into a filesystem-friendly name.

    Non-alphanumeric characters are replaced with underscores, and leading/trailing
    underscores are removed.

    Returns a sanitized string that can be used in file names.
    """

    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def predict_and_save(
    data_dir: str | Path = "data/raw",
    model_path: str | Path = "models/best_model.joblib",
    out_dir: str | Path = "data/predictions",
    tag: str = "model",
) -> Path:
    """
    The function loads a trained model bundle and generates predictions for the Kaggle test set.

    The model bundle is expected to contain:
    - a trained model (LightGBM Booster),
    - a fitted preprocessor (ColumnTransformer),
    - a selected decision threshold.

    Two output files are produced:
    - a submission file with predicted probabilities,
    - a decision file with probabilities and threshold-based binary predictions.

    Returns the path to the generated submission file.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Predict started. data_dir=%s model_path=%s out_dir=%s tag=%s",
        data_dir,
        model_path,
        out_dir,
        tag,
    )

    bundle = joblib.load(model_path)

    threshold = bundle["threshold"]
    logger.info("Model bundle loaded. threshold=%.6f", threshold)

    train_df, test_df = load_kaggle_data(data_dir)
    _, _, x_test = make_xy(train_df, test_df)

    x_test_proc = bundle["preprocessor"].transform(x_test)
    y_proba = bundle["model"].predict(x_test_proc)

    submission = pd.DataFrame(
        {
            ID_COL: test_df[ID_COL].values,
            TARGET_COL: y_proba,
        }
    )

    sub_path = out_path / f"submission_{safe_name(tag)}.csv"
    submission.to_csv(sub_path, index=False)

    logger.info("Saved submission: %s", sub_path)

    decision = submission.rename(columns={TARGET_COL: "proba"}).copy()
    decision["prediction"] = (decision["proba"] >= threshold).astype(int)

    decision_path = out_path / f"decisions_{safe_name(tag)}_thr_{threshold:.4f}.csv"
    decision.to_csv(decision_path, index=False)

    logger.info("Saved decisions: %s", decision_path)
    logger.info("Predict finished.")

    return sub_path


if __name__ == "__main__":
    p = predict_and_save()
    logger.info("Saved submission: %s", p)

