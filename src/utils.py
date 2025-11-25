from __future__ import annotations

import logging
import os
import sys
from typing import Dict

LOG_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_COLOR_MAP = {
    "DEBUG": "\033[36m",   # Cyan
    "INFO": "\033[32m",    # Green
    "WARNING": "\033[33m", # Yellow
    "ERROR": "\033[31m",   # Red
    "CRITICAL": "\033[35m",
}
_RESET = "\033[0m"
_handlers_initialized = False
_loggers: Dict[str, logging.Logger] = {}


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = _COLOR_MAP.get(record.levelname, "")
        if color:
            return f"{color}{message}{_RESET}"
        return message


def _configure_root_logger() -> None:
    global _handlers_initialized
    if _handlers_initialized:
        return
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColorFormatter(LOG_FORMAT, DATE_FORMAT))
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    _handlers_initialized = True


def get_logger(name: str) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]
    _configure_root_logger()
    logger = logging.getLogger(name)
    logger.propagate = True
    _loggers[name] = logger
    return logger


def log_section(logger: logging.Logger, title: str) -> None:
    logger.info("========== %s ==========", title.upper())


if __name__ == "__main__":
    test_logger = get_logger(__name__)
    log_section(test_logger, "logger demo")
    test_logger.debug("Debugging details.")
    test_logger.info("Informational message.")
    test_logger.warning("Warning about potential issue.")
    test_logger.error("Error encountered during processing.")
