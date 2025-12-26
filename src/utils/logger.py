from __future__ import annotations

import logging
import os
from pathlib import Path


def _add_file_handler(root: logging.Logger, log_path: Path, formatter: logging.Formatter) -> None:
    """
    The helper adds a FileHandler to the root logger.
    """
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)


def setup_logging(level: str | None = None) -> None:
    """
    The function configures project-wide logging.

    Logs are written to:
    - console (stdout)
    - logs/api.log    for loggers under "src.api"
    - logs/models.log for loggers under "src.models"
    - logs/data.log   for loggers under "src.data"
    - logs/utils.log  for loggers under "src.utils"
    """

    root = logging.getLogger()

    # If logging is already configured, this prevents duplicate handlers
    if root.handlers:
        return

    lvl = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    root.setLevel(lvl)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Always log to console
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create dedicated loggers per layer and attach file handlers to them
    mapping = {
        "src.api": log_dir / "api.log",
        "src.models": log_dir / "models.log",
        "src.data": log_dir / "data.log",
        "src.utils": log_dir / "utils.log",
    }

    for logger_name, file_path in mapping.items():
        lg = logging.getLogger(logger_name)
        lg.setLevel(lvl)
        lg.propagate = False  # avoid duplicates (console is already on root)

        _add_file_handler(lg, file_path, formatter)


def get_logger(name: str) -> logging.Logger:
    """
    The function returns a module-level logger with project-wide configuration applied.
    """
    setup_logging()
    return logging.getLogger(name)
