# src/models/train_holdout.py
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.data.load import load_kaggle_data
from src.data.preprocessing import make_xy, split_features, build_preprocessor_ohe
from src.utils.config import TARGET_COL, ID_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_holdout_check(
    data_dir: str | Path = "data/raw",
    model_dir: str | Path = "models",
    artifact_name: str = "best_model_holdout_check.joblib",
    seed: int = 42,
) -> Path:
    """Runs a simple holdout training check similar to the notebook workflow.
    The function splits the training data into an 80/20 train/validation holdout.
    The model is trained on the 80% subset using LightGBM CV to select best_iter.
    ROC-AUC is computed on the 20% holdout subset.
    The function saves the trained model and the fitted preprocessor as a joblib artifact.
    """

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Holdout check started. data_dir=%s model_dir=%s artifact=%s seed=%s",
        data_dir,
        model_dir,
        artifact_name,
        seed,
    )

    train_df, test_df = load_kaggle_data(data_dir)
    x, y, _ = make_xy(train_df, test_df)

    x_tr, x_val, y_tr, y_val = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    spec = split_features(train_df)
    preprocessor = build_preprocessor_ohe(spec.categorical)

    x_tr_proc = preprocessor.fit_transform(x_tr)
    x_val_proc = preprocessor.transform(x_val)

    dtrain = lgb.Dataset(x_tr_proc, label=y_tr)

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
    logger.info("CV finished. best_iter=%s", best_iter)

    model = lgb.train(lgb_params, dtrain, num_boost_round=best_iter)

    y_val_proba = model.predict(x_val_proc)
    holdout_auc = roc_auc_score(y_val, y_val_proba)

    logger.info("HOLDOUT CHECK (80/20). best_iter=%s holdout_auc=%.6f", best_iter, holdout_auc)

    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "threshold": 0.5,
        "meta": {
            "mode": "holdout_check_80_20",
            "seed": seed,
            "best_iter": best_iter,
            "holdout_auc": float(holdout_auc),
            "target_col": TARGET_COL,
            "id_col": ID_COL,
            "lgb_params": lgb_params,
        },
    }

    out_path = model_dir_path / artifact_name
    logger.info("Saving holdout-check bundle to: %s", out_path)
    joblib.dump(bundle, out_path)
    logger.info("Holdout check finished.")

    return out_path


if __name__ == "__main__":
    p = train_holdout_check()
    logger.info("Saved: %s", p)

