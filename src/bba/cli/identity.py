"""Run-level idempotency identity for ``bba audit``.

The CLI's ``run_id`` is the truncated sha256 over
``input_csv_hash + schema_fingerprint + code_version`` (PRD §20). It is
deliberately *not* the same construction as :class:`bba.ingest.RunIdentity`
— ingest works at the *bundle* level (a directory of HOSxP tables) and
keeps the full 64-char digest; the CLI works at the *single-file* level
that the operator passes on the command line and surfaces a short 16-char
prefix that is grep-friendly in logs without sacrificing collision
resistance for Phase-1-scale corpora.

Both constructions agree on the *recipe* (sha256 over input bytes +
schema fingerprint + package version) so a single change in any
component changes the run_id and forces a fresh audit, satisfying the
"no silent re-use of stale outputs" invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import NewType


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


def code_version() -> CodeVersion:
    """Return the running ``blood-audit`` package version.

    Single-sourced via :func:`importlib.metadata.version` so the version
    string can never drift between the wheel metadata and a hand-edited
    constant.

    GREEN-phase implementation: read ``importlib.metadata.version("blood-audit")``.
    """
    raise NotImplementedError("code_version() — GREEN phase wires importlib.metadata")


def compute_run_id(
    *,
    input_csv: Path,
    schema_fingerprint: SchemaFingerprint,
    code_version_str: CodeVersion,
) -> RunId:
    """Compute the CLI ``run_id`` for the (input, schema, code) triple.

    The recipe is::

        sha256(content_hash(input_csv) + schema_fingerprint + code_version)
        .hexdigest()[:RUN_ID_LENGTH]

    ``content_hash`` is :func:`bba.ingest.hashing.content_hash`, a
    chunked sha256 over the file bytes — *not* over the file's modified
    time or path. Two identical-byte inputs at different paths therefore
    produce the same ``run_id``.

    GREEN-phase implementation will:
    1. ``content_hash(input_csv)`` → 64-char hex
    2. concat with ``schema_fingerprint`` and ``code_version_str``
    3. sha256 + hexdigest()[:RUN_ID_LENGTH]
    """
    raise NotImplementedError(
        "compute_run_id — GREEN phase wires the sha256-over-concat recipe"
    )
