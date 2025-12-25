# src/models/promote.py
from __future__ import annotations

from pathlib import Path
import argparse
import shutil

from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_mlflow_model_path(run_id: str, experiment_id: str = "1") -> Path:
    """
    The function builds the path to a model bundle stored by MLflow for a given run_id.

    Expected location:
    mlruns/<experiment_id>/<run_id>/artifacts/model/best_model.joblib

    Returns the resolved Path.
    """
    return (
        Path("mlruns")
        / str(experiment_id)
        / str(run_id)
        / "artifacts"
        / "model"
        / "best_model.joblib"
    )


def promote_run_to_active_model(
    run_id: str,
    experiment_id: str = "1",
    model_dir: str | Path = "models",
    active_name: str = "best_model.joblib",
) -> Path:
    """
    The function makes a selected MLflow run model the "active" model of the project.

    It copies the model bundle from MLflow artifacts into:
    models/<active_name>

    It also writes the selected run_id into:
    models/latest_run.txt

    Returns the path to the active model bundle.
    """
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    src_path = resolve_mlflow_model_path(run_id=run_id, experiment_id=experiment_id)
    if not src_path.exists():
        raise FileNotFoundError(
            f"MLflow model bundle not found for run_id={run_id}\n"
            f"Expected: {src_path}\n"
            f"Tip: check experiment_id (default is 1) and that artifacts were logged."
        )

    dst_path = model_dir_path / active_name

    logger.info(
        "Promoting run to active model. run_id=%s experiment_id=%s",
        run_id,
        experiment_id,
    )
    logger.info("Copying: %s -> %s", src_path, dst_path)

    shutil.copy2(src_path, dst_path)

    (model_dir_path / "latest_run.txt").write_text(run_id, encoding="utf-8")

    logger.info("Active model is now: %s", dst_path)
    logger.info("latest_run.txt updated with run_id=%s", run_id)

    return dst_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote an MLflow run model to the active project model.")
    parser.add_argument("--run-id", type=str, required=True, help="MLflow run_id to promote")
    parser.add_argument("--experiment-id", type=str, default="1", help="MLflow experiment id (default: 1)")
    args = parser.parse_args()

    promote_run_to_active_model(run_id=args.run_id, experiment_id=args.experiment_id)
