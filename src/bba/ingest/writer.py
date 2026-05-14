"""DuckDB+Parquet writer surface for the ingest store.

The Phase-1 contract exposed to the pipeline is:

* :func:`is_run_complete` — read-only predicate consulted at the start of every
  run; if it returns True the pipeline MUST no-op (``skipped_idempotent=True``).
* :func:`mark_run_complete` — write the completion marker atomically once all
  per-table writes have succeeded. Atomicity prevents a partially written run
  from being treated as complete on the next invocation.

The actual Parquet partition writes are filled in by subsequent tickets (#4–#7)
that consume the validated dataframes; for now the marker bookkeeping is what
the idempotency contract relies on.
"""

from __future__ import annotations

from pathlib import Path


def _marker_path(output_dir: Path, run_id: str) -> Path:
    return output_dir / f"_run_{run_id}.complete"


def is_run_complete(output_dir: Path, run_id: str) -> bool:
    """Return True iff a previous ingest with ``run_id`` finished writing all tables.

    A True return means the pipeline MUST no-op and return an
    :class:`~bba.ingest.models.IngestResult` with ``skipped_idempotent=True``.
    """
    if not output_dir.exists():
        return False
    return _marker_path(output_dir, run_id).is_file()


def mark_run_complete(output_dir: Path, run_id: str) -> None:
    """Atomically write the completion marker for ``run_id``.

    The marker is created via a write-then-rename so a crash mid-write cannot
    leave a half-formed marker that future runs would mistake for completion.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = _marker_path(output_dir, run_id)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text("ok\n", encoding="utf-8")
    tmp.replace(target)
