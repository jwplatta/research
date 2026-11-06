"""Logging setup for the backtest module."""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "backtest"


def get_logger() -> logging.Logger:
    """Return the shared backtest logger (no handlers attached yet)."""
    return logging.getLogger(LOGGER_NAME)


def configure(out_dir: Path, strategy_name: str) -> logging.Logger:
    """Attach a file handler and a stderr handler to the backtest logger.

    Creates ``out_dir/{strategy_name}_{timestamp}.log``.
    Safe to call multiple times — clears existing handlers first.
    """
    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(logging.DEBUG)

    # Clear any handlers from a previous run in the same process
    log.handlers.clear()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"{strategy_name}_{ts}.log"

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(fmt)

    log.addHandler(file_handler)
    log.addHandler(stderr_handler)

    log.info("Log file: %s", log_path)
    return log
