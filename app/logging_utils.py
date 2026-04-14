"""Shared logging setup for the AI doppelganger."""

from __future__ import annotations

import logging
import sys

from uvicorn.logging import DefaultFormatter

LOG_FORMAT = "%(levelprefix)s [%(asctime)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application logging once for CLI and server entrypoints."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            if handler.formatter is None:
                handler.setFormatter(
                    DefaultFormatter(
                        fmt=LOG_FORMAT,
                        datefmt=DATE_FORMAT,
                        use_colors=True,
                    )
                )
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        DefaultFormatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT,
            use_colors=True,
        )
    )
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
