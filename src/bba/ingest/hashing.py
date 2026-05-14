"""Content + schema fingerprint hashing and run_id derivation.

PRD §1, fix E32: ``run_id = sha256(input_csv_hash + schema_fingerprint + code_version)``.
Same inputs ⇒ same run_id ⇒ idempotent re-runs (the writer no-ops a complete run).
The concatenation order is part of the public contract — changing it would
invalidate every previously persisted ``audit_id``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 65_536


def content_hash(path: Path) -> str:
    """Return the sha256 hex digest of the file contents at ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_run_id(input_csv_hash: str, schema_fp: str, code_version: str) -> str:
    """Return ``sha256(input_csv_hash + schema_fp + code_version)`` as a hex digest.

    The concatenation order is fixed by contract; do not reorder.
    """
    payload = (input_csv_hash + schema_fp + code_version).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
