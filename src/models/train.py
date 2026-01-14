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
    The function selects a decision threshold such that the achieved False Positive Rate (FPR)
    is as close as possible to the target value.

    The threshold is selected by scanning either all unique score values
    or a quantile-based grid when the score distribution is large.

    Returns:
        threshold: selected probability threshold
        info: dictionary with achieved FPR and confusion matrix components
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
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel()

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
    threshold_holdout_size: float = 0.20,
) -> Path:
    """
    The function trains the final LightGBM model and produces a production-ready model bundle.

    Training procedure:
    1. Split the training data into:
       - a fitting subset (used to train the model),
       - a holdout subset (used ONLY to select the decision threshold).
    2. Train the model (with CV and early stopping) on the fitting subset.
    3. Select a probability threshold on the holdout subset to match target FPR.
    4. Re-train the final model on ALL available training data
       using the previously selected threshold.

    This approach avoids threshold overfitting while keeping the final model
    trained on the full dataset.
    """

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents = True, exist_ok = True)

    logger.info(
        "Train started. data_dir = %s model_dir = %s artifact = %s seed = %s target_fpr = %.4f",
        data_dir,
        model_dir,
        artifact_name,
        seed,
        target_fpr,
    )

    mlflow.set_experiment("loan_payback_training")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run_id = %s", run_id)

        # Load data and build feature matrices
        train_df, test_df = load_kaggle_data(data_dir)
        x_all, y_all, _ = make_xy(train_df, test_df)

        # Feature specification is stored only for metadata/debugging purposes
        spec = split_features(train_df)

        # Split data:
        # - x_fit / y_fit: used to train the model
        # - x_thr / y_thr: used ONLY to select the decision threshold
        x_fit, x_thr, y_fit, y_thr = train_test_split(
            x_all,
            y_all,
            test_size = threshold_holdout_size,
            random_state = seed,
            stratify = y_all,
        )

        # Preprocessor is fitted only on the fitting subset
        preprocessor_fit = build_preprocessor_ohe(spec.categorical)
        x_fit_proc = preprocessor_fit.fit_transform(x_fit)
        dtrain = lgb.Dataset(x_fit_proc, label = y_fit)

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

        mlflow.log_params(lgb_params)
        mlflow.log_param("target_fpr", target_fpr)

        # Cross-validation to select the number of boosting iterations
        cv_res = lgb.cv(
            lgb_params,
            dtrain,
            num_boost_round = 5000,
            nfold = 5,
            stratified = True,
            shuffle = True,
            seed = seed,
            callbacks = [
                lgb.early_stopping(stopping_rounds = 50),
                lgb.log_evaluation(period = 50),
            ],
        )

        best_iter = len(cv_res["valid auc-mean"])
        mlflow.log_param("best_iter", best_iter)
        logger.info("CV finished. best_iter = %s", best_iter)

        # Train model on the fitting subset
        model_fit = lgb.train(lgb_params, dtrain, num_boost_round = best_iter)

        # Select threshold on the holdout subset (unseen during training)
        x_thr_proc = preprocessor_fit.transform(x_thr)
        y_thr_score = model_fit.predict(x_thr_proc)

        best_thr, thr_info = find_threshold_by_target_fpr(
            y_true = y_thr,
            y_score = y_thr_score,
            target_fpr = target_fpr,
        )

        logger.info(
            "Threshold selected on holdout. threshold = %.6f achieved_fpr = %.6f",
            best_thr,
            thr_info["FPR"],
        )

        mlflow.log_metric("threshold", best_thr)
        mlflow.log_metric("FPR_holdout", thr_info["FPR"])

        # Refit final model on ALL data to maximize predictive quality.
        # The threshold remains fixed and is NOT re-optimized.
        preprocessor_final = build_preprocessor_ohe(spec.categorical)
        x_all_proc = preprocessor_final.fit_transform(x_all)
        dtrain_all = lgb.Dataset(x_all_proc, label = y_all)

        model_final = lgb.train(lgb_params, dtrain_all, num_boost_round = best_iter)

        bundle = {
            "model": model_final,
            "preprocessor": preprocessor_final,
            "threshold": float(best_thr),
            "meta": {
                "target_col": TARGET_COL,
                "id_col": ID_COL,
                "best_iter": best_iter,
                "threshold_info": thr_info,
                "threshold_holdout_size": threshold_holdout_size,
                "categorical_features": spec.categorical,
                "numeric_features": spec.numeric,
                "lgb_params": lgb_params,
                "run_id": run_id,
            },
        }

        out_path = model_dir_path / artifact_name
        joblib.dump(bundle, out_path)

        # Store run_id of the latest successfully trained model
        (model_dir_path / "latest_run.txt").write_text(run_id, encoding = "utf-8")

        mlflow.log_artifact(str(out_path), artifact_path = "model")
        logger.info("Train finished. Model saved to %s", out_path)

        return out_path


if __name__ == "__main__":
    path = train_best_model()
    logger.info("Saved: %s", path)
