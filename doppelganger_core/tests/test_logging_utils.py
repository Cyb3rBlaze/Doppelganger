"""Tests for shared logging configuration."""

from __future__ import annotations

import logging

from uvicorn.logging import DefaultFormatter

from app.logging_utils import DATE_FORMAT, LOG_FORMAT, configure_logging


def test_configure_logging_adds_default_formatter_when_missing() -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    try:
        root_logger.handlers.clear()
        configure_logging()
        assert root_logger.handlers
        assert isinstance(root_logger.handlers[0].formatter, DefaultFormatter)
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)


def test_configure_logging_updates_existing_handler_without_formatter() -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    handler = logging.StreamHandler()
    handler.setFormatter(None)
    try:
        root_logger.handlers = [handler]
        configure_logging()
        assert isinstance(handler.formatter, DefaultFormatter)
        assert handler.formatter._fmt == LOG_FORMAT
        assert handler.formatter.datefmt == DATE_FORMAT
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
