"""Structured logging configuration (spec §Phase 7.1).

Call configure_logging() once at service startup (in main.py lifespan).
After configuration, all stdlib logging calls are rendered as JSON by
structlog, with request_id propagated via contextvars.
"""

from __future__ import annotations

import logging
import logging.config
import sys
from contextvars import ContextVar

import structlog

# Per-request context variable — set by RequestIDMiddleware, read by structlog processors.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def _add_request_id(logger: object, method: str, event_dict: dict) -> dict:
    event_dict["request_id"] = request_id_var.get()
    return event_dict


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure structlog + stdlib logging for the retriever service.

    Args:
        level:     Root log level (e.g. "INFO", "DEBUG").
        json_logs: Render as JSON (production default); False renders
                   human-friendly console output for local dev.
    """
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        _add_request_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)

    # Silence noisy third-party libraries.
    for lib in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(lib).setLevel(logging.WARNING)
