"""
Project-wide logging configuration.

Functions
---------
setup_logging(level)
    Configure project-wide logging with consistent formatting.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure project-wide logging with consistent formatting.

    Parameters
    ----------
    level : int
        Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``).

    Returns
    -------
    logging.Logger
        The configured ``adoption_accelerator`` logger.
    """
    logger = logging.getLogger("adoption_accelerator")
    logger.setLevel(level)

    # Avoid duplicate handlers on re-run
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
