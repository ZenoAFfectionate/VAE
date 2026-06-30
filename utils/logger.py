"""Unified logging utility.

Creates a logger that writes simultaneously to the console (clean, readable
output) and to a persistent log file inside the experiment directory.
"""

from __future__ import annotations

import logging
import os
import sys

_LOGGER_NAME = "vae"


def setup_logger(log_dir: str, log_name: str = "train.log",
                 name: str = _LOGGER_NAME) -> logging.Logger:
    """Initialise and return a configured logger.

    The logger emits records to both ``stdout`` and ``<log_dir>/<log_name>``.
    Re-initialising clears previous handlers so repeated calls stay clean.

    Args:
        log_dir: Directory in which the log file is written (created if absent).
        log_name: File name of the log file.
        name: Logger name.

    Returns:
        The configured :class:`logging.Logger` instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove stale handlers to avoid duplicated log lines across runs.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, log_name),
                                       encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_config(logger: logging.Logger, title: str, config: dict) -> None:
    """Pretty-print a configuration dictionary with clear divider lines."""
    divider = "=" * 60
    logger.info(divider)
    logger.info(title)
    logger.info(divider)
    for key in sorted(config):
        logger.info(f"{key:>20} : {config[key]}")
    logger.info(divider)
