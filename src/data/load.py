# src/data/load.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.data.download import download_kaggle_competition
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_kaggle_data(
    data_dir: str | Path = "data/raw",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    The function loads Kaggle competition training and test datasets.

    The presence of the required CSV files is ensured by calling
    `download_kaggle_competition`. The function then reads the datasets
    from disk and returns them as pandas DataFrames.

    Returns the training and test datasets as pandas DataFrames.
    """

    data_path = Path(data_dir)
    logger.info("Loading Kaggle data. data_dir = %s", data_path)

    # The presence of the required data files is ensured
    data_path = download_kaggle_competition(data_path)
    logger.info("Raw data directory resolved: %s", data_path)

    train_path = data_path / "train.csv"
    test_path = data_path / "test.csv"

    # A sanity check verifies that the expected files exist on disk
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train.csv and test.csv in {data_path}"
        )

    # The CSV files are read into pandas DataFrames
    logger.info("Reading CSV files: %s and %s", train_path, test_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(
        "Data loaded successfully. train_shape = %s test_shape = %s",
        train_df.shape,
        test_df.shape,
    )

    return train_df, test_df
