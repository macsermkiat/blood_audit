"""Domain exceptions for ``bba.report_generator``.

Raised at the report-generation boundary so callers (the monthly CLI, the
dashboard's "download my own-view" handler) can distinguish a recoverable
input problem from an internal bug.
"""

from __future__ import annotations


class ReportGenerationError(Exception):
    """Base class for every report-generator exception.

    Catch this at the CLI / dashboard boundary to render a user-facing
    error message instead of a bare traceback.
    """


class EmptyInputError(ReportGenerationError):
    """Raised when zero ``MonthlyReportRow`` instances fall inside the
    requested month.

    An empty report is not the same as a successful zero-volume month: the
    monthly run guarantees ``ingest`` has already loaded the month's data,
    so a true zero would indicate ``audit_orders`` excluded every order or
    the month filter is wrong. Surfacing rather than silently emitting an
    empty CSV/PDF gives the operator a chance to investigate.
    """


class FooterStampError(ReportGenerationError):
    """Raised when the reproducibility footer is missing or partial.

    PRD §"Output schema" requires ``policy_version``, ``model_id``,
    ``redactor_version`` on every persisted artifact. The aggregate layer
    enforces presence at the schema boundary; this exception covers the
    rare case where the CSV/PDF writer is invoked with a partially
    constructed footer (e.g., during a half-finished refactor).
    """
