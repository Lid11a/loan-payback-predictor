# src/models/train.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src.data.load import load_kaggle_data
from src.data.preprocessing import make_xy, split_features, build_preprocessor_ohe
from src.utils.config import TARGET_COL, ID_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_threshold_by_target_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float = 0.20,
    grid_size: int = 1000,
) -> Tuple[float, Dict[str, Any]]:
    """
    The function selects a decision threshold such that the false positive rate (FPR)
    is as close as possible to `target_fpr`.

    The threshold search is performed over either all unique score values or over a
    quantile-based grid when the number of unique values is large.

    Returns the selected threshold and a dictionary containing FPR and confusion
    matrix components at that threshold.
    """

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    uniq = np.unique(y_score)
    thresholds = (
        uniq
        if uniq.size <= grid_size
        else np.quantile(y_score, np.linspace(0.001, 0.999, grid_size + 1))
    )

    best_thr = None
    best_gap = float("inf")
    best_info = None

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        fpr = fp / (fp + tn + 1e-12)
        gap = abs(fpr - target_fpr)

        if gap < best_gap:
            best_gap = gap
            best_thr = float(thr)
            best_info = {
                "FPR": float(fpr),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }

    return float(best_thr), best_info


def train_best_model(
    data_dir: str | Path = "data/raw",
    model_dir: str | Path = "models",
    artifact_name: str = "best_model.joblib",
    seed: int = 42,
    target_fpr: float = 0.20,
) -> Path:
    """
    The function trains the final LightGBM model and saves a model bundle to disk.

    Training uses one-hot encoding for categorical features and LightGBM cross-validation
    with early stopping to determine the number of boosting iterations. A separate split
    is used to select a decision threshold that targets a specified false positive rate.

    Returns the path to the saved model bundle file.
    """

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Train started. data_dir=%s model_dir=%s artifact=%s seed=%s target_fpr=%.4f",
        data_dir,
        model_dir,
        artifact_name,
        seed,
        target_fpr,
    )

    # Set MLflow experiment (creates it if it does not exist)
    mlflow.set_experiment("loan_payback_training")

    # One MLflow run = one training execution
    with mlflow.start_run() as run:
        # Unique ID of this training run (used to locate model + metrics later)
        run_id = run.info.run_id
        logger.info("MLflow run_id=%s", run_id)

        # Data loading
        train_df, test_df = load_kaggle_data(data_dir)
        x_train, y, _ = make_xy(train_df, test_df)

        # Preprocessing
        spec = split_features(train_df)
        preprocessor = build_preprocessor_ohe(spec.categorical)

        x_proc = preprocessor.fit_transform(x_train)
        dtrain = lgb.Dataset(x_proc, label=y)

        # Model parameters
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.1,
            "max_depth": 6,
            "lambda_l2": 0.5,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "num_leaves": 60,
            "min_data_in_leaf": 45,
            "verbose": -1,
            "seed": seed,
        }

        # Log parameters to MLflow
        mlflow.log_params(lgb_params)
        mlflow.log_param("target_fpr", target_fpr)

        # Cross-validation
        cv_res = lgb.cv(
            lgb_params,
            dtrain,
            num_boost_round=5000,
            nfold=5,
            stratified=True,
            shuffle=True,
            seed=seed,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50),
            ],
        )

        best_iter = len(cv_res["valid auc-mean"])
        mlflow.log_param("best_iter", best_iter)

        logger.info("CV finished. best_iter=%s", best_iter)

        # Final training
        model = lgb.train(lgb_params, dtrain, num_boost_round=best_iter)

        # Threshold selection
        _, x_thr_val, _, y_thr_val = train_test_split(
            x_train,
            y,
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )

        x_thr_val_proc = preprocessor.transform(x_thr_val)
        y_thr_score = model.predict(x_thr_val_proc)

        best_thr, thr_info = find_threshold_by_target_fpr(
            y_true=y_thr_val,
            y_score=y_thr_score,
            target_fpr=target_fpr,
        )

        logger.info(
            "Threshold selected. threshold=%.6f achieved_fpr=%.6f",
            best_thr,
            thr_info["FPR"],
        )

        # Log metrics to MLflow
        mlflow.log_metric("threshold", best_thr)
        mlflow.log_metric("FPR", thr_info["FPR"])
        mlflow.log_metric("TP", thr_info["TP"])
        mlflow.log_metric("FP", thr_info["FP"])
        mlflow.log_metric("TN", thr_info["TN"])
        mlflow.log_metric("FN", thr_info["FN"])

        # Save model bundle
        bundle = {
            "model": model,
            "preprocessor": preprocessor,
            "threshold": best_thr,
            "meta": {
                "target_col": TARGET_COL,
                "id_col": ID_COL,
                "best_iter": best_iter,
                "threshold_info": thr_info,
                "categorical_features": spec.categorical,
                "numeric_features": spec.numeric,
                "lgb_params": lgb_params,
                "run_id": run_id,
            },
        }

        out_path = model_dir_path / artifact_name
        joblib.dump(bundle, out_path)

        (model_dir_path / "latest_run.txt").write_text(run_id, encoding="utf-8")

        # Log model artifact to MLflow
        mlflow.log_artifact(str(out_path), artifact_path="model")

        logger.info("Train finished. Model saved to %s", out_path)

        return out_path


if __name__ == "__main__":
    path = train_best_model()
    logger.info("Saved: %s", path)
