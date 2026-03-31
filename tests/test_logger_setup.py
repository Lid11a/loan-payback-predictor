# tests/test_logger_setup.py
from pathlib import Path
import logging

from src.utils.logger import setup_logging


LOGGER_FILE_MAP = {
    "src.api": "api.log",
    "src.models": "models.log",
    "src.data": "data.log",
    "src.utils": "utils.log",
    "src.monitoring": "monitoring.log",
}


def _clear_logger(logger: logging.Logger) -> None:
    """
    Close and remove all handlers from a logger to avoid test cross-contamination.
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def _reset_logging_state() -> None:
    """
    Reset root logger and project loggers before each test.
    """
    _clear_logger(logging.getLogger())

    for logger_name in LOGGER_FILE_MAP:
        _clear_logger(logging.getLogger(logger_name))


def test_setup_logging_creates_logs_dir(tmp_path, monkeypatch):
    """
    The function verifies that `setup_logging` creates the logs directory.
    """
    monkeypatch.chdir(tmp_path)
    _reset_logging_state()

    setup_logging(level = "INFO")

    assert (tmp_path / "logs").is_dir()


def test_setup_logging_adds_console_handler_with_warning_level(tmp_path, monkeypatch):
    """
    The function verifies that `setup_logging` adds exactly one console handler
    to the root logger and sets its level to WARNING.
    """
    monkeypatch.chdir(tmp_path)
    _reset_logging_state()

    setup_logging(level = "INFO")

    root = logging.getLogger()
    stream_handlers = [h for h in root.handlers if type(h) is logging.StreamHandler]

    assert len(stream_handlers) == 1
    assert stream_handlers[0].level == logging.WARNING


def test_setup_logging_creates_file_handlers_for_each_layer(tmp_path, monkeypatch):
    """
    The function verifies that each project layer logger gets its own FileHandler
    pointing to the expected file in the logs directory.
    """
    monkeypatch.chdir(tmp_path)
    _reset_logging_state()

    setup_logging(level = "INFO")

    for logger_name, file_name in LOGGER_FILE_MAP.items():
        logger = logging.getLogger(logger_name)
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]

        assert len(file_handlers) == 1
        assert Path(file_handlers[0].baseFilename).name == file_name
        assert (tmp_path / "logs" / file_name).exists()


def test_setup_logging_does_not_duplicate_file_handlers_on_repeated_calls(tmp_path, monkeypatch):
    """
    The function verifies that repeated calls to `setup_logging` do not add
    duplicate FileHandlers to the same logger.
    """
    monkeypatch.chdir(tmp_path)
    _reset_logging_state()

    setup_logging(level = "INFO")
    setup_logging(level = "INFO")

    for logger_name, file_name in LOGGER_FILE_MAP.items():
        logger = logging.getLogger(logger_name)
        matching_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).name == file_name
        ]

        assert len(matching_handlers) == 1


def test_setup_logging_uses_env_log_level_when_argument_is_not_passed(tmp_path, monkeypatch):
    """
    The function verifies that `setup_logging` uses LOG_LEVEL from environment
    when the `level` argument is not provided.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    _reset_logging_state()

    setup_logging()

    root = logging.getLogger()
    api_logger = logging.getLogger("src.api")

    assert root.level == logging.DEBUG
    assert api_logger.level == logging.DEBUG


def test_api_child_logger_writes_message_to_api_log(tmp_path, monkeypatch):
    """
    The function verifies that a child logger under `src.api`
    writes messages into logs/api.log.
    """
    monkeypatch.chdir(tmp_path)
    _reset_logging_state()

    setup_logging(level = "INFO")

    logger = logging.getLogger("src.api.test_module")
    message = "api logger test message"

    logger.warning(message)

    # Flush handlers to make sure the message is written to disk before assertions.
    for handler in logging.getLogger("src.api").handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    log_text = (tmp_path / "logs" / "api.log").read_text(encoding="utf-8")

    assert message in log_text
