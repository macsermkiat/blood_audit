"""Typed exceptions for bba.monitoring.

A small typed hierarchy so operators can ``except MonitoringError`` for
broad recovery while alerting consumers can pin specific subclasses.
"""

from __future__ import annotations


class MonitoringError(Exception):
    """Base for every exception raised inside :mod:`bba.monitoring`."""


class InsufficientHistoryError(MonitoringError):
    """The drift window or sentinel history is too small to evaluate.

    Raised by :mod:`bba.monitoring.drift_sprt` when fewer than ``min_n``
    observations have accumulated, and by :mod:`bba.monitoring.sentinel`
    when no previous-week run exists to compare against.
    """


class SentinelStaleError(MonitoringError):
    """The sentinel manifest on disk does not match the requested seed.

    Raised when a previously-persisted manifest at the same ``size``/``seed``
    pair has shifted membership — typically a sign that the underlying
    population changed between weeks AND the manifest was rebuilt rather
    than reused (the manifest is supposed to be built ONCE).
    """


class GoldenSetMismatchError(MonitoringError):
    """Baseline and current golden-set runs do not cover the same rows.

    The golden-set drift probe compares paired rows by ``audit_id``; a row
    in ``current`` that has no counterpart in ``baseline`` (or vice versa)
    indicates the golden set itself was edited, which breaks the
    quarterly comparison contract.
    """


__all__ = (
    "GoldenSetMismatchError",
    "InsufficientHistoryError",
    "MonitoringError",
    "SentinelStaleError",
)
