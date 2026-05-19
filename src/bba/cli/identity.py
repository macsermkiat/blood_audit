"""Run-level idempotency identity for ``bba audit``.

The CLI's ``run_id`` is the 16-char prefix of
:class:`bba.ingest.RunIdentity` computed over every HOSxP-named CSV in
``input_csv.parent`` — the same set of files that
:func:`bba.ingest.ingest` actually reads. Hashing only the operator's
``--input`` file would make ``bba audit`` blind to sibling-table edits:
e.g., a freshly-exported ``Diagnosis.csv`` next to an unchanged
``BDVST.csv`` would yield the same run_id as the prior audit, so the
idempotency check would no-op and reuse stale results.

The CLI keeps the short 16-char prefix (grep-friendly in logs) while
delegating the per-file aggregation to
:class:`bba.ingest.RunIdentity`, so the CLI's idempotency key and the
ingest module's run identity stay aligned by construction.
"""

from __future__ import annotations

from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import NewType, cast

from bba.ingest import RunIdentity, all_tables, content_hash
from bba.ingest.models import CSVTable

from bba.cli.exceptions import CliError


InputCsvHash = NewType("InputCsvHash", str)
"""sha256 hex digest of an input CSV's bytes (see :mod:`bba.ingest.hashing`)."""


SchemaFingerprint = NewType("SchemaFingerprint", str)
"""sha256 hex digest of the active HOSxP schema set — see
:func:`bba.ingest.schema_fingerprint`."""


CodeVersion = NewType("CodeVersion", str)
"""The running package version, single-sourced from ``pyproject.toml``
via :func:`importlib.metadata.version`."""


RunId = NewType("RunId", str)
"""16-char hex prefix of sha256(input_hash + schema_fp + code_version)."""


RUN_ID_LENGTH: int = 16
"""Truncation length of :data:`RunId`. Changing this is a breaking change
to every persisted audit row's identity, so it lives as a module-level
constant a reviewer is unlikely to miss."""


_PACKAGE_NAME: str = "blood-audit"
"""Distribution name on PyPI / in pyproject.toml. The bba submodule is
shipped inside this distribution (see ``[tool.uv.build-backend]``
``module-name = ["bba", "blood_audit"]``)."""


def code_version() -> CodeVersion:
    """Return the running ``blood-audit`` package version.

    Single-sourced via :func:`importlib.metadata.version` so the version
    string can never drift between the wheel metadata and a hand-edited
    constant.
    """
    return CodeVersion(_pkg_version(_PACKAGE_NAME))


def compute_run_id(
    *,
    input_csv: Path,
    schema_fingerprint: SchemaFingerprint,
    code_version_str: CodeVersion,
) -> RunId:
    """Compute the CLI ``run_id`` for the (bundle, schema, code) triple.

    The bundle is every HOSxP-named CSV file in ``input_csv.parent``
    (the same set :func:`bba.ingest.ingest` reads). Each file is hashed
    via :func:`bba.ingest.content_hash` and the per-file hashes feed
    :meth:`bba.ingest.RunIdentity.from_inputs`, whose 64-char digest is
    truncated to :data:`RUN_ID_LENGTH` for the CLI's grep-friendly id.

    Raises :class:`CliError` if no HOSxP-named CSVs are present in the
    parent directory — :func:`bba.ingest.ingest` would raise
    :class:`bba.ingest.IncompleteInputError` later anyway; failing
    early avoids persisting an idempotency marker for an unanchored
    bundle.
    """
    bundle_dir = input_csv.parent
    known: frozenset[str] = frozenset(all_tables())
    per_file_hashes: dict[CSVTable, str] = {}
    for csv_path in sorted(bundle_dir.glob("*.csv")):
        if csv_path.stem in known:
            per_file_hashes[cast(CSVTable, csv_path.stem)] = content_hash(csv_path)
    if not per_file_hashes:
        raise CliError(
            f"compute_run_id: {bundle_dir} contains no HOSxP-named CSV "
            "files; cannot derive a stable run identity for an unanchored "
            f"input bundle ({input_csv})"
        )
    identity = RunIdentity.from_inputs(
        per_file_hashes,
        schema_fingerprint,
        code_version_str,
    )
    return RunId(identity.run_id[:RUN_ID_LENGTH])
