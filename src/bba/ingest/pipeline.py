"""End-to-end ingest orchestration: CSV → pandera validation → DuckDB+Parquet.

Public entry point: :func:`ingest`. On a re-run with an unchanged input set
and code version the function MUST return an :class:`IngestResult` with
``skipped_idempotent=True`` rather than rewriting the store.
"""

from __future__ import annotations

from bba.ingest.models import IngestConfig, IngestResult


def ingest(config: IngestConfig) -> IngestResult:
    """Ingest the configured CSV directory into the Parquet store.

    Per PRD §1:
      - validate each CSV against its pandera schema (fail loud on drift via
        :class:`~bba.ingest.schemas.SchemaDriftError`);
      - parse every time column with :func:`~bba.ingest.time_parser.parse_hosxp_time`;
      - normalize timestamps to UTC;
      - derive ``run_id`` deterministically and no-op when already complete.
    """
    raise NotImplementedError
