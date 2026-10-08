"""Logging configuration with optional JSON output and request correlation.

The application uses the standard library :mod:`logging` module.  A
:class:`contextvars.ContextVar` carries the current request id so every log
record emitted while handling a request can be correlated, even across
``await`` boundaries and threadpool hops.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

_CONFIGURED = False


def set_request_id(request_id: str | None) -> None:
    """Bind the request id used by the log formatter for the current task."""
    _request_id_ctx.set(request_id)


def get_request_id() -> str | None:
    """Return the request id bound to the current task, if any."""
    return _request_id_ctx.get()


class RequestIdFilter(logging.Filter):
    """Inject the current request id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Minimal dependency-free structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable single-line formatter for local development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | [%(request_id)s] %(message)s",
            datefmt="%H:%M:%S",
        )


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure the root logger exactly once (idempotent)."""
    global _CONFIGURED

    root = logging.getLogger()
    root.setLevel(level.upper())

    if _CONFIGURED:
        for handler in root.handlers:
            handler.setLevel(level.upper())
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level.upper())
    handler.setFormatter(JsonFormatter() if json_logs else ConsoleFormatter())
    handler.addFilter(RequestIdFilter())

    root.handlers.clear()
    root.addHandler(handler)

    # Quieten noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
