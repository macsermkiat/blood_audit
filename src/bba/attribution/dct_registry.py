"""Loader for the ``DCT.csv`` doctor registry.

``DCT.csv`` is the HOSxP doctor master table: ``Dct`` (the code
``BDVST.DCTREQ`` points at) plus masked name parts and the department
columns ``Deptlct`` / ``Deptname``. Codes and department names are
non-PHI; person names arrive pre-masked (``ส*****``).
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from bba.attribution.csv_support import require_columns
from bba.attribution.models import DoctorRecord


_REQUIRED_COLUMNS: tuple[str, ...] = (
    "Dct",
    "Prefix",
    "Fname",
    "Lname",
    "Deptlct",
    "Deptname",
)


def _display_name(prefix: str, fname: str, lname: str) -> str:
    """Compose the masked display name (``พญ.ส***** ว*****``).

    Empty parts collapse cleanly so a registry row with only a first
    name still renders without stray whitespace.
    """
    head = f"{prefix}{fname}".strip()
    return " ".join(part for part in (head, lname) if part)


def load_dct_registry(path: Path) -> Mapping[str, DoctorRecord]:
    """Read ``DCT.csv`` into a ``Dct`` → :class:`DoctorRecord` mapping.

    Rows with an empty ``Dct`` are skipped (they cannot be referenced by
    any ``DCTREQ``). A duplicated ``Dct`` key fails loud: the registry is
    the attribution ground truth, and last-wins would silently reassign
    every order of that doctor to whichever row happened to come later.
    """
    registry: dict[str, DoctorRecord] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        require_columns(reader.fieldnames, _REQUIRED_COLUMNS, path)
        for row in reader:
            dct = (row["Dct"] or "").strip()
            if not dct:
                continue
            if dct in registry:
                raise ValueError(
                    f"{path} contains duplicate Dct key {dct!r}; the doctor "
                    "registry must be unique per code — fix the export "
                    "before ranking on it"
                )
            registry[dct] = DoctorRecord(
                dct=dct,
                display_name=_display_name(
                    (row["Prefix"] or "").strip(),
                    (row["Fname"] or "").strip(),
                    (row["Lname"] or "").strip(),
                ),
                deptlct=(row["Deptlct"] or "").strip(),
                deptname=(row["Deptname"] or "").strip(),
            )
    return registry
