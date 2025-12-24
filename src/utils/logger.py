# src/utils/logger.py
from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    """
    The function configures a consistent logging format for the project.

    The logging level can be controlled via:
    - the `level` argument, or
    - the LOG_LEVEL environment variable (DEBUG / INFO / WARNING / ERROR).

    If logging is already configured (handlers exist), the function does nothing.
    """

    root = logging.getLogger()

    # If logging is already configured, this prevents duplicate handlers
    if root.handlers:
        return

    lvl = (level or os.getenv("LOG_LEVEL") or "INFO").upper()

    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """
    The function returns a module-level logger with project-wide configuration applied.
    """

    setup_logging()
    return logging.getLogger(name)
