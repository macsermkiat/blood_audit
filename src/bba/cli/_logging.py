"""Structlog bridge — emit JSON via stdlib logging so SIEM, file
handlers, and pytest's ``caplog`` all see the same scrubbed events.

Configuration is install-once-per-process. Subsequent calls are
no-ops, which lets every entrypoint (subcommand callback, excepthook
installer, test fixture) ask for a logger without coordinating who
configures first.
"""

from __future__ import annotations

from typing import Final

import structlog


_BBA_LOGGER_NAME: Final[str] = "bba.cli"


_CONFIGURED: bool = False


def _configure_once() -> None:
    """Wire structlog → stdlib logging on the first call only.

    Idempotent so tests that re-import or re-install can call freely.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str = _BBA_LOGGER_NAME) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog BoundLogger bound to ``name``.

    Output is JSON via stdlib logging, which means ``caplog`` /
    ``logging.handlers.RotatingFileHandler`` / a SIEM forwarder can all
    consume the same stream without extra adapters.
    """
    _configure_once()
    return structlog.stdlib.get_logger(name)
