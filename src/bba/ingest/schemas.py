"""Pandera schemas (version v1) for the 10 HOSxP CSV tables.

The schemas are the authoritative description of expected columns. Their joint
sha256 fingerprint feeds into ``run_id``, so a silent schema bump (forgetting
to bump v1 → v2) becomes a new ``run_id`` and triggers a fresh ingest rather
than silently mutating prior outputs.

Schema-drift policy (PRD §1, fix E29): an input CSV with an unknown column
MUST fail loud with :class:`SchemaDriftError` containing the offending columns
and the table name. Silent column drops are forbidden.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import cast, get_args

from pandera.pandas import Column, DataFrameSchema

from bba.ingest.models import CSVTable


class SchemaDriftError(Exception):
    """Raised when an input CSV has columns not declared in its pandera schema.

    The error message includes the table name and the unknown column(s) so
    operators can act without re-running with extra logging.
    """


def _str(nullable: bool = True) -> Column:
    return Column(str, nullable=nullable, coerce=True)


_REGISTRY_V1: Mapping[CSVTable, DataFrameSchema] = {
    "BDVST": DataFrameSchema(
        {
            "HN": _str(nullable=False),
            "AN": _str(),
            "REQNO": _str(nullable=False),
            "BDVSTST": _str(),
            "REQTYPE": _str(),
            "CANCELDATE": _str(),
        }
    ),
    "BDVSTDT": DataFrameSchema(
        {"REQNO": _str(nullable=False), "DATETIME": _str()},
    ),
    "BDTYPE": DataFrameSchema(
        {"REQNO": _str(nullable=False), "PRODUCT": _str()},
    ),
    "BDVSTST": DataFrameSchema(
        {"REQNO": _str(nullable=False), "STATUS": _str()},
    ),
    "Diagnosis": DataFrameSchema(
        {"HN": _str(nullable=False), "AN": _str(), "ICD10": _str()},
    ),
    "Lab": DataFrameSchema(
        {
            "HN": _str(nullable=False),
            "LABEXM": _str(),
            "LABRESULT": _str(),
            "DATETIME": _str(),
        },
    ),
    "MED": DataFrameSchema(
        {"HN": _str(nullable=False), "DRUG": _str(), "DATETIME": _str()},
    ),
    "IPDADMPROGRESS": DataFrameSchema(
        {"HN": _str(nullable=False), "OBJECTIVE": _str(), "DATETIME": _str()},
    ),
    "IPDNRFOCUSDT": DataFrameSchema(
        {"HN": _str(nullable=False), "FOCUS": _str(), "DATETIME": _str()},
    ),
    "UnUSE_Patient_Background": DataFrameSchema(
        {"HN": _str(nullable=False), "BIRTHDATE": _str(), "SEX": _str()},
    ),
}


def get_schema(table: CSVTable) -> DataFrameSchema:
    """Return the v1 pandera schema for ``table``.

    Raises ``KeyError`` if ``table`` is not a registered CSVTable literal.
    """
    return _REGISTRY_V1[table]


def all_tables() -> tuple[CSVTable, ...]:
    """Return the canonical tuple of all 10 required CSV tables.

    The order matches the :data:`bba.ingest.models.CSVTable` literal so callers
    can rely on a stable iteration order across releases.
    """
    return cast("tuple[CSVTable, ...]", get_args(CSVTable))


def schema_fingerprint() -> str:
    """Return a stable sha256 hex digest over all v1 schemas.

    The fingerprint is deterministic: schemas are visited in the canonical
    :func:`all_tables` order, and within each schema the column names are
    visited in their declared order. Changing a column name or order produces
    a new fingerprint, which in turn produces a new ``run_id``.
    """
    h = hashlib.sha256()
    for table in all_tables():
        h.update(table.encode("utf-8"))
        schema = _REGISTRY_V1[table]
        for col_name, col in schema.columns.items():
            h.update(b"|")
            h.update(col_name.encode("utf-8"))
            h.update(b":")
            h.update(repr(col.dtype).encode("utf-8"))
            h.update(b":nullable=")
            h.update(b"1" if col.nullable else b"0")
    return h.hexdigest()
