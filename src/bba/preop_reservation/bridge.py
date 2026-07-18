"""Fail-loud loader for the vendored OPRTACT->ICD9 bridge reference.

Inert library module (ticket #197): nothing in the package imports it and it
is deliberately NOT exported from ``bba.preop_reservation.__init__`` — the
picker-v2 seams consume it directly once the wiring ticket lands (the
``name_match`` precedent). The packaged CSV is emitted by
``scripts/build_oprtact_icd9_bridge.py``; the raw hospital export is never
read at runtime.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from types import MappingProxyType

BRIDGE_REFERENCE_FILENAME = "oprtact_icd9_bridge.csv"
_REQUIRED_COLUMNS = frozenset(
    {"oprtact", "icd9", "icd9_nodot", "score", "human_index", "human_agreed", "name"}
)


class BridgeReferenceError(ValueError):
    """The vendored bridge CSV is malformed and cannot be used."""


@dataclass(frozen=True, slots=True)
class BridgeEntry:
    """One OPRTACT key's First-Choice ICD-9 mapping with human provenance."""

    icd9: str
    icd9_nodot: str
    score: float
    human_index: str
    human_agreed: bool
    name: str


@dataclass(frozen=True, slots=True)
class OprtactBridge:
    """Immutable, content-addressed OPRTACT -> BridgeEntry index."""

    content_hash: str
    _entries: Mapping[str, BridgeEntry] = field(repr=False)

    def get(self, oprtact: str) -> BridgeEntry | None:
        """Entry for a (whitespace-stripped) OPRTACT key; None if absent."""
        return self._entries.get(oprtact.strip())

    def __len__(self) -> int:
        return len(self._entries)


def _parse_score(raw: str, *, row_number: int) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise BridgeReferenceError(
            f"malformed bridge rejected at row {row_number}: invalid score {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise BridgeReferenceError(
            f"malformed bridge rejected at row {row_number}: non-finite score {raw!r}"
        )
    return value


def _bridge_from_rows(
    rows: Iterable[Mapping[str, str | None]], *, content_hash: str
) -> OprtactBridge:
    """Build a validated bridge from parsed rows (test seam)."""
    entries: dict[str, BridgeEntry] = {}
    for row_number, row in enumerate(rows, start=2):
        missing = _REQUIRED_COLUMNS.difference(row)
        if missing or any(row.get(column) is None for column in _REQUIRED_COLUMNS):
            missing_text = ", ".join(
                sorted(missing or {c for c in _REQUIRED_COLUMNS if row.get(c) is None})
            )
            raise BridgeReferenceError(
                f"malformed bridge rejected at row {row_number}: "
                f"missing columns {missing_text}"
            )
        oprtact = (row["oprtact"] or "").strip()
        if not oprtact:
            raise BridgeReferenceError(
                f"malformed bridge rejected at row {row_number}: blank oprtact"
            )
        if oprtact in entries:
            raise BridgeReferenceError(
                f"malformed bridge rejected at row {row_number}: "
                f"duplicate oprtact {oprtact!r}"
            )
        icd9 = (row["icd9"] or "").strip()
        if not icd9:
            raise BridgeReferenceError(
                f"malformed bridge rejected at row {row_number}: blank icd9"
            )
        icd9_nodot = (row["icd9_nodot"] or "").strip()
        if icd9_nodot != icd9.replace(".", ""):
            raise BridgeReferenceError(
                f"malformed bridge rejected at row {row_number}: icd9_nodot "
                f"{icd9_nodot!r} does not match icd9 {icd9!r}"
            )
        score = _parse_score((row["score"] or "").strip(), row_number=row_number)
        human_agreed_raw = row["human_agreed"] or ""
        if human_agreed_raw not in ("true", "false"):
            raise BridgeReferenceError(
                f"malformed bridge rejected at row {row_number}: human_agreed "
                f"must be 'true' or 'false', got {human_agreed_raw!r}"
            )
        entries[oprtact] = BridgeEntry(
            icd9=icd9,
            icd9_nodot=icd9_nodot,
            score=score,
            human_index=(row["human_index"] or "").strip(),
            human_agreed=human_agreed_raw == "true",
            name=(row["name"] or "").strip(),
        )
    frozen = MappingProxyType(dict(sorted(entries.items())))
    return OprtactBridge(content_hash=content_hash, _entries=frozen)


@lru_cache(maxsize=1)
def load_oprtact_bridge() -> OprtactBridge:
    """Load and validate the immutable vendored bridge once per process."""
    reference_path = resources.files("bba.preop_reservation").joinpath(
        "data", BRIDGE_REFERENCE_FILENAME
    )
    raw_bytes = reference_path.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    reader = csv.DictReader(io.StringIO(raw_bytes.decode("utf-8-sig")))
    if reader.fieldnames is None:
        raise BridgeReferenceError("malformed bridge rejected: missing header row")
    missing = _REQUIRED_COLUMNS.difference(reader.fieldnames)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise BridgeReferenceError(
            f"malformed bridge rejected: missing columns {missing_text}"
        )
    return _bridge_from_rows(reader, content_hash=content_hash)


__all__ = [
    "BRIDGE_REFERENCE_FILENAME",
    "BridgeEntry",
    "BridgeReferenceError",
    "OprtactBridge",
    "load_oprtact_bridge",
]
