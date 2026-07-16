"""Fail-loud loader for the vendored MSBOS reference schedule."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import Literal, cast

from bba.preop_reservation.models import MsbosRow, MsbosToken

MSBOS_REFERENCE_FILENAME = "OPBloodLimit_by_icd9.csv"
_REQUIRED_COLUMNS = frozenset({"icd9_code_nodot", "msbos", "recommended_units"})
_MSBOS_TOKENS = frozenset({"none", "G/M", "T/S"})
_INTEGER_RE = re.compile(r"\d+")
_RANGE_RE = re.compile(r"(\d+)-(\d+)")

Resolution = MsbosRow | Literal["ambiguous"] | None


class MsbosReferenceError(ValueError):
    """The vendored MSBOS schedule is malformed and cannot be evaluated."""


@dataclass(frozen=True, slots=True)
class MsbosReference:
    """Immutable, content-addressed ICD-9 to MSBOS recommendation index."""

    content_hash: str
    _rows_by_code: Mapping[str, frozenset[MsbosRow]]

    def resolve(self, icd9_nodot: str) -> Resolution:
        """Resolve a code to one recommendation, ambiguity, or no match."""
        matches = self._rows_by_code.get(icd9_nodot.strip())
        if not matches:
            return None
        if len(matches) > 1:
            return "ambiguous"
        return next(iter(matches))


def parse_recommended_units(raw: str) -> int:
    """Parse a blank, integer, or integer range to its recommended high end."""
    value = raw.strip()
    if not value:
        return 0
    if _INTEGER_RE.fullmatch(value):
        return int(value)
    range_match = _RANGE_RE.fullmatch(value)
    if range_match is not None:
        return int(range_match.group(2))
    raise MsbosReferenceError(f"invalid recommended_units value {raw!r}")


def _reference_from_rows(
    rows: Iterable[Mapping[str, str]], *, content_hash: str
) -> MsbosReference:
    """Build a validated reference from parsed rows (test seam)."""
    by_code: dict[str, set[MsbosRow]] = {}
    for row_number, row in enumerate(rows, start=2):
        missing = _REQUIRED_COLUMNS.difference(row)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise MsbosReferenceError(
                f"malformed schedule rejected at row {row_number}: "
                f"missing columns {missing_text}"
            )
        token = row["msbos"].strip()
        if token not in _MSBOS_TOKENS:
            raise MsbosReferenceError(
                f"malformed schedule rejected at row {row_number}: "
                f"unknown msbos token {token!r}"
            )
        try:
            units = parse_recommended_units(row["recommended_units"])
        except MsbosReferenceError as exc:
            raise MsbosReferenceError(
                f"malformed schedule rejected at row {row_number}: {exc}"
            ) from exc
        code = row["icd9_code_nodot"].strip()
        if not code:
            continue
        recommendation = MsbosRow(
            msbos=cast(MsbosToken, token),
            recommended_units=units,
        )
        by_code.setdefault(code, set()).add(recommendation)

    frozen_rows = MappingProxyType(
        {code: frozenset(values) for code, values in sorted(by_code.items())}
    )
    return MsbosReference(content_hash=content_hash, _rows_by_code=frozen_rows)


@lru_cache(maxsize=1)
def load_msbos_reference() -> MsbosReference:
    """Load and validate the immutable vendored schedule once per process."""
    reference_path = resources.files("bba.preop_reservation").joinpath(
        "data", MSBOS_REFERENCE_FILENAME
    )
    raw_bytes = reference_path.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    reader = csv.DictReader(io.StringIO(raw_bytes.decode("utf-8-sig")))
    if reader.fieldnames is None:
        raise MsbosReferenceError("malformed schedule rejected: missing header row")
    missing = _REQUIRED_COLUMNS.difference(reader.fieldnames)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise MsbosReferenceError(
            f"malformed schedule rejected: missing columns {missing_text}"
        )
    return _reference_from_rows(reader, content_hash=content_hash)


__all__ = [
    "MSBOS_REFERENCE_FILENAME",
    "MsbosReference",
    "MsbosReferenceError",
    "Resolution",
    "load_msbos_reference",
    "parse_recommended_units",
]
