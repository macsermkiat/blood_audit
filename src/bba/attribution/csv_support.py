"""Shared CSV-loading guard for the attribution loaders.

Both raw inputs (``DCT.csv``, ``BDVST.csv``) are wide hospital exports;
the loaders read only the columns they need via ``csv.DictReader`` and
must fail loud — naming the file and the missing column — when a column
is absent, mirroring the ingest layer's schema-drift philosophy
(:class:`bba.ingest.schemas.SchemaDriftError`): a silent ``None`` from a
renamed column would zero out attribution without any other warning.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def require_columns(
    fieldnames: Sequence[str] | None,
    required: Sequence[str],
    path: Path,
) -> None:
    """Raise :class:`ValueError` naming every required column missing
    from ``fieldnames`` (or all of them when the file has no header)."""
    present = set(fieldnames or ())
    missing = [col for col in required if col not in present]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {missing}; "
            f"expected at least {list(required)} in the header "
            f"(got {sorted(present)})"
        )
