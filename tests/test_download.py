# tests/test_download.py
"""
Tests for `download_kaggle_competition`.

The test suite focuses on behavior that should remain stable without requiring
real Kaggle access:
- if `train.csv` and `test.csv` already exist, no external download is invoked;
- if files are missing, a mocked "Kaggle download" produces a zip archive that
  is extracted into the target directory.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import Mock

from src.data.download import download_kaggle_competition


def _make_zip_with_csvs(zip_path: Path) -> None:
    """
    The helper creates a small zip archive that contains `train.csv` and `test.csv`.

    This archive simulates the dataset artifact that would normally be produced by
    the Kaggle CLI.
    """

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("train.csv", "id,target\n1,0\n")
        zf.writestr("test.csv", "id\n10\n")


def test_download_skips_when_files_exist(tmp_path, monkeypatch):
    """
    The test verifies that the download step is skipped when expected CSV files exist.

    The Kaggle CLI invocation is replaced with a stub that fails the test if called.
    """

    # The raw directory is prepared with the expected CSV files
    data_dir = tmp_path / "raw"
    data_dir.mkdir(parents = True, exist_ok = True)
    (data_dir / "train.csv").write_text("id,target\n1,0\n", encoding = "utf-8")
    (data_dir / "test.csv").write_text("id\n10\n", encoding = "utf-8")

    # The stub represents an unwanted external call in this scenario
    def _fail_run(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("subprocess.run should NOT be called when files exist")

    # The patch is applied on the module where `subprocess.run` is used
    import src.data.download as download_module
    monkeypatch.setattr(download_module.subprocess, "run", _fail_run)

    out = download_kaggle_competition(data_dir)

    # The function is expected to return the same directory without side effects
    assert out == data_dir


def test_download_extracts_zip_when_missing(tmp_path, monkeypatch):
    """
    The test verifies that a zip archive is extracted when CSV files are missing.

    The Kaggle CLI invocation is mocked to create a zip archive in the target directory.
    After the function call, extracted `train.csv` and `test.csv` are expected to exist.
    """

    # The raw directory is prepared without train/test files
    data_dir = tmp_path / "raw"
    data_dir.mkdir(parents = True, exist_ok = True)

    # The mocked download creates a zip file in the same directory
    zip_path = data_dir / "dataset.zip"

    def _fake_run(*args, **kwargs):
        _ = args, kwargs
        _make_zip_with_csvs(zip_path)

        # The object below mimics the minimal interface of `subprocess.run(...)` result
        res = Mock()
        res.stdout = "ok"
        res.stderr = ""
        res.returncode = 0
        res.check_returncode = Mock()  # success path
        return res

    # The patch is applied on the module where `subprocess.run` is used
    import src.data.download as download_module
    monkeypatch.setattr(download_module.subprocess, "run", _fake_run)

    out = download_kaggle_competition(data_dir)

    # The function is expected to return the directory and to extract the CSV files
    assert out == data_dir
    assert (data_dir / "train.csv").exists()
    assert (data_dir / "test.csv").exists()
