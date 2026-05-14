"""Pandera schemas (version v1) for the 10 HOSxP CSV tables.

The schemas are the authoritative description of expected columns and dtypes.
Their joint sha256 fingerprint feeds into ``run_id``, so a silent schema bump
(forgetting to bump v1 → v2) is reflected as a new ``run_id`` and re-ingests
rather than silently mutating prior outputs.

Schema-drift policy (PRD §1, fix E29): an input CSV with an unknown column
MUST fail loud with :class:`SchemaDriftError` containing the offending columns
and the table name. Silent column drops are forbidden.
"""

from __future__ import annotations

from pandera.pandas import DataFrameSchema

from bba.ingest.models import CSVTable


class SchemaDriftError(Exception):
    """Raised when an input CSV has columns not declared in its pandera schema.

    The error message must include the table name and the unknown column(s)
    so operators can act without re-running with extra logging.
    """


def get_schema(table: CSVTable) -> DataFrameSchema:
    """Return the v1 pandera schema for ``table``."""
    raise NotImplementedError


def all_tables() -> tuple[CSVTable, ...]:
    """Return the canonical tuple of all 10 required CSV tables.

    Used by the pipeline to drive ingestion and by tests to assert schema
    coverage.
    """
    raise NotImplementedError


def schema_fingerprint() -> str:
    """Return the joint sha256 hex digest over all v1 schemas (stable per code release)."""
    raise NotImplementedError
