"""Content-hash primitive for per-file digests.

The run-identity formula and the on-disk completion-marker convention live in
:mod:`bba.ingest.run_identity` — this module is the small utility that streams
a file through sha256 and returns the hex digest.
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
