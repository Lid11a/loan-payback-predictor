# src/models/predict.py
from __future__ import annotations

from pathlib import Path
import re
import argparse

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


def resolve_model_path(
    model_path: str | Path,
    run_id: str | None = None,
    experiment_id: str = "1",
) -> Path:
    """
    The function resolves the model bundle path.

    If run_id is provided, the model bundle is loaded from the MLflow artifacts folder:
    mlruns/<experiment_id>/<run_id>/artifacts/model/best_model.joblib

    If run_id is not provided, the function returns model_path (default: models/best_model.joblib).

    Returns the resolved path to the model bundle.
    """

    if run_id:
        return (
            Path("mlruns")
            / str(experiment_id)
            / str(run_id)
            / "artifacts"
            / "model"
            / "best_model.joblib"
        )

    return Path(model_path)


def predict_and_save(
    data_dir: str | Path = "data/raw",
    model_path: str | Path = "models/best_model.joblib",
    out_dir: str | Path = "data/predictions",
    tag: str = "model",
    run_id: str | None = None,
    experiment_id: str = "1",
) -> Path:
    """
    The function loads a trained model bundle and generates predictions for the Kaggle test set.

    The model bundle is expected to contain:
    - a trained model (LightGBM Booster),
    - a fitted preprocessor (ColumnTransformer),
    - a selected decision threshold.

    If run_id is provided, the model bundle is loaded from MLflow artifacts:
    mlruns/<experiment_id>/<run_id>/artifacts/model/best_model.joblib

    Two output files are produced:
    - a submission file with predicted probabilities,
    - a decision file with probabilities and threshold-based binary predictions.

    Returns the path to the generated submission file.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    resolved_model_path = resolve_model_path(
        model_path=model_path,
        run_id=run_id,
        experiment_id=experiment_id,
    )

    logger.info(
        "Predict started. data_dir=%s model_path=%s out_dir=%s tag=%s run_id=%s",
        data_dir,
        resolved_model_path,
        out_dir,
        tag,
        run_id,
    )

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {resolved_model_path}\n"
            f"If you used --run-id, the expected path is:\n"
            f"mlruns/{experiment_id}/{run_id}/artifacts/model/best_model.joblib\n"
            f"If you did not use --run-id, check:\n"
            f"{Path(model_path)}"
        )

    bundle = joblib.load(resolved_model_path)

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

    run_suffix = f"_run_{safe_name(run_id)}" if run_id else ""
    sub_path = out_path / f"submission_{safe_name(tag)}{run_suffix}.csv"
    submission.to_csv(sub_path, index=False)

    logger.info("Saved submission: %s", sub_path)

    decision = submission.rename(columns={TARGET_COL: "proba"}).copy()
    decision["prediction"] = (decision["proba"] >= threshold).astype(int)

    decision_path = out_path / f"decisions_{safe_name(tag)}{run_suffix}_thr_{threshold:.4f}.csv"
    decision.to_csv(decision_path, index=False)

    logger.info("Saved decisions: %s", decision_path)
    logger.info("Predict finished.")

    return sub_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run_id to load the model from mlruns/",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="1",
        help="MLflow experiment id (default: 1)",
    )
    args = parser.parse_args()

    p = predict_and_save(run_id=args.run_id, experiment_id=args.experiment_id)
    logger.info("Saved submission: %s", p)
