"""DuckDB+Parquet writer for the ingest store.

Exposes a single read-only predicate :func:`is_run_complete` that the pipeline
uses to short-circuit a re-run when the same ``run_id`` is already fully persisted.
"""

from __future__ import annotations

from pathlib import Path


def is_run_complete(output_dir: Path, run_id: str) -> bool:
    """Return True iff a previous ingest with ``run_id`` finished writing all tables.

    A True return means the pipeline MUST no-op (``skipped_idempotent=True`` on the
    resulting :class:`~bba.ingest.models.IngestResult`).
    """
    raise NotImplementedError
