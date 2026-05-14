"""Content + schema fingerprint hashing and run_id derivation.

PRD §1, fix E32: ``run_id = sha256(input_csv_hash + schema_fingerprint + code_version)``.
Same inputs ⇒ same run_id ⇒ idempotent re-runs (the writer no-ops a complete run).
"""

from __future__ import annotations

from pathlib import Path


def content_hash(path: Path) -> str:
    """Return the sha256 hex digest of the file contents at ``path``."""
    raise NotImplementedError


def compute_run_id(input_csv_hash: str, schema_fp: str, code_version: str) -> str:
    """Return ``sha256(input_csv_hash + schema_fp + code_version)`` as a hex digest.

    The concatenation order is part of the public contract — changing it would
    break reproducibility of prior audit rows. Do not reorder.
    """
    raise NotImplementedError
