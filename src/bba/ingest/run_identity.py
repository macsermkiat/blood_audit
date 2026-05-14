"""Run identity — the stable fingerprint of a single ingest invocation.

A ``run_id`` is ``sha256(input_csv_hash + schema_fingerprint + code_version)``,
where ``input_csv_hash`` is the sha256 of canonically-ordered ``"<table>:<file_sha>"``
pairs across the validated input set. Same byte content + same schemas + same
code yields the same ``run_id``, which is the entire basis of idempotent re-runs.

This module concentrates four formerly-shallow pieces (PRD §1 fix E32):

* the hash-of-hashes aggregation across the validated input set;
* the sha256 concat formula;
* the on-disk completion marker convention;
* the write-then-rename atomicity that prevents half-formed markers.

All four answer the single question "is this run already done?", so they live
behind one interface (:class:`RunIdentity`) with two value constructors and
two state-changing methods.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from bba.ingest.models import CSVTable


def _aggregate_input_hash(per_file_hashes: Mapping[CSVTable, str]) -> str:
    """Stable sha256 over ``<table>:<file_sha>`` pairs sorted by table name.

    Sorting ensures the aggregate hash is invariant under directory-iteration
    order changes, which matters for cross-platform reproducibility.
    """
    h = hashlib.sha256()
    for table in sorted(per_file_hashes):
        h.update(table.encode("utf-8"))
        h.update(b":")
        h.update(per_file_hashes[table].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _compute(input_csv_hash: str, schema_fp: str, code_version: str) -> str:
    payload = (input_csv_hash + schema_fp + code_version).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Stable identity of a single ingest run.

    Construct via :meth:`from_inputs` (production path: pass per-file hashes)
    or :meth:`from_components` (test path: pass already-aggregated inputs).
    The ``run_id`` field is the 64-char sha256 hex digest. The on-disk marker
    convention is private to this class — callers go through :meth:`is_complete`
    and :meth:`mark_complete`, never the file path directly.
    """

    run_id: str

    @classmethod
    def from_inputs(
        cls,
        per_file_hashes: Mapping[CSVTable, str],
        schema_fingerprint: str,
        code_version: str,
    ) -> RunIdentity:
        """Build identity from validated per-table file hashes."""
        return cls(
            run_id=_compute(
                _aggregate_input_hash(per_file_hashes),
                schema_fingerprint,
                code_version,
            ),
        )

    @classmethod
    def from_components(
        cls,
        input_csv_hash: str,
        schema_fingerprint: str,
        code_version: str,
    ) -> RunIdentity:
        """Build identity from already-aggregated components.

        Useful in tests that exercise the formula directly without staging
        actual CSVs on disk.
        """
        return cls(run_id=_compute(input_csv_hash, schema_fingerprint, code_version))

    def is_complete(self, output_dir: Path) -> bool:
        """True iff a previous run with this ``run_id`` finished writing all tables.

        A True return means the pipeline MUST no-op (``skipped_idempotent=True``).
        """
        if not output_dir.exists():
            return False
        return self._marker_path(output_dir).is_file()

    def mark_complete(self, output_dir: Path) -> None:
        """Atomically write the completion marker for this ``run_id``.

        write-then-rename so a crash mid-write cannot leave a half-formed marker
        that future runs would mistake for completion.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        target = self._marker_path(output_dir)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text("ok\n", encoding="utf-8")
        tmp.replace(target)

    def _marker_path(self, output_dir: Path) -> Path:
        return output_dir / f"_run_{self.run_id}.complete"
