"""Typed errors raised by :mod:`bba.audit_orders`.

Each error names the rule it represents so callers can pattern-match on the
type (not on substring-search of the message) when deciding whether to
quarantine the offending record or fail the whole run.
"""

from __future__ import annotations


class AuditOrdersError(Exception):
    """Base class for all :mod:`bba.audit_orders` errors."""


class UnrecoverableAnchorError(AuditOrdersError):
    """Both ``REQDATE+REQTIME`` and ``BDVSTDATE+BDVSTTIME`` are unusable.

    Raised when a record passes every inclusion/exclusion gate but the anchor
    datetime cannot be derived from either the primary or the fallback pair.
    Per PRD §"Output schema", every persisted ``audit_orders`` row must carry
    an ``order_datetime`` — a missing anchor is an integrity bug, not a
    silently-dropped row.
    """
