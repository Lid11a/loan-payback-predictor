# tests/test_logger_setup.py
from pathlib import Path
import logging

from src.utils.logger import setup_logging


def test_setup_logging_creates_logs_dir(tmp_path, monkeypatch):
    """
    The function verifies that `setup_logging` creates the logs directory
    when logging is initialized.
    """

    monkeypatch.chdir(tmp_path)

    root = logging.getLogger()
    root.handlers.clear()  # important in tests

    setup_logging(level = "INFO")

    assert Path("logs").exists()
