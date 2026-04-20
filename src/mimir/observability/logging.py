"""Structured JSON logging helpers."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra = {
            k: v
            for k, v in record.__dict__.items()
            if k not in logging.LogRecord.__dict__
            and not k.startswith("_")
            and k not in ("msg", "args", "exc_info", "exc_text", "stack_info", "message")
        }
        if extra:
            payload["extra"] = extra
        return json.dumps(payload, default=str)


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that emits JSON to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_pipeline_event(
    logger: logging.Logger,
    event: str,
    chunk_id: str,
    **kwargs: Any,
) -> None:
    """Emit a structured pipeline event."""
    logger.info(
        event,
        extra={"chunk_id": chunk_id, "event": event, **kwargs},
    )
