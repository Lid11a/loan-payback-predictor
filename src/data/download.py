# src/data/download.py
from __future__ import annotations

from pathlib import Path
import zipfile
import subprocess

from src.utils.config import COMPETITION_SLUG as COMPETITION
from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_kaggle_competition(data_dir: str | Path = "data/raw") -> Path:
    """
    The function ensures that Kaggle competition data is available locally.

    If train.csv and test.csv already exist in `data_dir`, it returns the directory
    path without performing any download. Otherwise, it downloads the competition
    archive via the Kaggle CLI and extracts its contents.

    Returns the path to the directory containing the extracted CSV files.
    """

    # The input path is normalized and the target directory is ensured to exist
    data_path = Path(data_dir)
    data_path.mkdir(parents = True, exist_ok = True)
    logger.info("Download started. competition = %s data_dir = %s", COMPETITION, data_path)

    train_csv = data_path / "train.csv"
    test_csv = data_path / "test.csv"

    # If the expected files are already present, no further action is required
    if train_csv.exists() and test_csv.exists():
        logger.info("Raw data already exists: %s and %s", train_csv, test_csv)
        return data_path

    # The Kaggle CLI is invoked to download the competition archive
    res = subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(data_path)],
        text = True,
        capture_output = True,
    )

    # Kaggle CLI output is logged to assist with diagnostics if needed
    if res.stdout:
        logger.info("Kaggle stdout:\n%s", res.stdout)
    if res.stderr:
        logger.warning("Kaggle stderr:\n%s", res.stderr)

    # An exception is raised if the CLI command did not complete successfully
    res.check_returncode()

    # Downloaded zip archives are located in the target directory
    zips = list(data_path.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(
            "Kaggle download finished, but no .zip found. "
            "This may indicate an issue with Kaggle API access or competition permissions."
        )

    # The largest archive is selected, which typically corresponds to the main dataset
    zip_path = max(zips, key = lambda p: p.stat().st_size)

    logger.info("Extracting archive: %s", zip_path)

    # The contents of the archive are extracted into the target directory
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_path)

    # Optional cleanup: the archive may be removed after extraction
    # zip_path.unlink(missing_ok = True)

    # A final check verifies that the expected CSV files are present
    if not (train_csv.exists() and test_csv.exists()):
        raise FileNotFoundError(
            f"Expected {train_csv.name} and {test_csv.name} in {data_path}. "
            "This may be related to missing permissions for the competition on Kaggle."
        )

    logger.info("Download finished. Files ready in %s", data_path)

    return data_path
