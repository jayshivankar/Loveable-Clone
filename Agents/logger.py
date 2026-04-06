"""
Structured JSON logging with structlog.

Usage:
    from Agents.logger import get_logger
    log = get_logger("planner")
    log.info("analysing_prompt", session_id="abc123", prompt_length=42)
"""

from __future__ import annotations

import logging
import sys

import structlog

_log_callbacks = []

def register_log_callback(cb) -> None:
    _log_callbacks.append(cb)

def sse_callback_processor(logger, log_method, event_dict):
    # Call all registered callbacks with a copy so they don't mutate the log
    for cb in _log_callbacks:
        try:
            cb(dict(event_dict))
        except Exception:
            pass
    return event_dict

def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            sse_callback_processor,
            structlog.processors.JSONRenderer(),   # JSON for production
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "codeforge") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
