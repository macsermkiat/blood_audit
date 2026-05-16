"""Typed exceptions for bba.monitoring.

A small typed hierarchy so operators can ``except MonitoringError`` for
broad recovery while alerting consumers can pin specific subclasses.
"""

from __future__ import annotations


class MonitoringError(Exception):
    """Base for every exception raised inside :mod:`bba.monitoring`."""


class InsufficientHistoryError(MonitoringError):
    """The drift window or sentinel history is too small to evaluate.

    Raised by :mod:`bba.monitoring.sentinel` when no previous-week run
    exists to compare against, or when the intersection of the manifest
    and the two run mappings is empty (nothing to pair κ on).
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
)
