"""Pandera schemas (version v1) for the 12 HOSxP CSV tables.

The schemas are the authoritative description of expected columns. Their joint
sha256 fingerprint feeds into ``run_id``, so a silent schema bump (forgetting
to bump v1 → v2) becomes a new ``run_id`` and triggers a fresh ingest rather
than silently mutating prior outputs.

Schema-drift policy (PRD §1, fix E29): an input CSV with an unknown column
MUST fail loud with :class:`SchemaDriftError` containing the offending columns
and the table name. Silent column drops are forbidden — *except* through the
normalize layer's per-table allow-list (policy a), which projects the file
to the declared columns and logs the dropped extras to the run audit before
``validate_header`` runs. See ``docs/ingest-mapping.md`` for the per-table
column specs and normalize rules.
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


class IncompleteInputError(Exception):
    """Raised when the input directory is missing one or more canonical HOSxP CSVs.

    The error message names the missing tables so operators can fix the export
    before re-running. This is distinct from :class:`SchemaDriftError` (per-table
    column drift); ``IncompleteInputError`` is about the registry-level *set* of
    tables that must be present for an ingest to be meaningful.
    """


def _str(nullable: bool = True) -> Column:
    return Column(str, nullable=nullable, coerce=True)


_REGISTRY_V1: Mapping[CSVTable, DataFrameSchema] = {
    "BDVST": DataFrameSchema(
        {
            "HN": _str(nullable=False),
            "REQNO": _str(nullable=False),
            "AN": _str(),
            "BDVSTST": _str(),
            "REQTYPE": _str(),
            "CANCELDATE": _str(),
            "REQDATE": _str(),
            "REQTIME": _str(),
            "BDVSTDATE": _str(),
            "BDVSTTIME": _str(),
            "ICD10": _str(),
            "DIAGNOSIS": _str(),
            # Ordering-doctor code (~99.8% filled). Joined to DCT.csv
            # for physician / department attribution (Feature 2); the
            # export's separate DCT column is empty and stays dropped.
            "DCTREQ": _str(),
        }
    ),
    "BDVSTDT": DataFrameSchema(
        {
            "REQNO": _str(nullable=False),
            "HN": _str(),
            "BDVSTDATE": _str(),
            "BDVSTTIME": _str(),
            "USEDATE": _str(),
            "USETIME": _str(),
            "BDTYPE": _str(),
            "ITEMNO": _str(),
            "UNITAMT": _str(),
        }
    ),
    "BDVSTST": DataFrameSchema(
        {
            "BDVSTST": _str(nullable=False),
            "NAME": _str(),
        }
    ),
    "BDTYPE": DataFrameSchema(
        {
            "BDTYPE": _str(nullable=False),
            "NAME": _str(),
        }
    ),
    "Diagnosis": DataFrameSchema(
        {
            "HN": _str(nullable=False),
            "AN": _str(nullable=False),
            "NAME_DIAGTYPE": _str(),
            "ICD10": _str(),
            "ICD10WHO": _str(),
            "NAME_ICD10": _str(),
        }
    ),
    "Lab": DataFrameSchema(
        {
            "HN": _str(nullable=False),
            "AN": _str(nullable=False),
            "LVSTDATE": _str(),
            "LVSTTIME": _str(),
            "LABGRP": _str(),
            "NAME_LABGRP": _str(),
            "LABEXM": _str(),
            "NAME_LABEXM": _str(),
            "RESULT": _str(),
            "MINNRM": _str(),
            "MAXNRM": _str(),
            "NRMUNIT": _str(),
        }
    ),
    "Med": DataFrameSchema(
        {
            "HN": _str(nullable=False),
            "AN": _str(nullable=False),
            "PRSCDATE": _str(),
            "PRSCTIME": _str(),
            "MEDITEM": _str(),
            "NAME_MEDITEM": _str(),
            "GENERIC": _str(),
            "NAME_GENERIC": _str(),
            "STRENGTH": _str(),
            "STRENGTHUNIT": _str(),
            "MEDUSETYPE": _str(),
            "MEDUSEQTY": _str(),
        }
    ),
    "IPDADMPROGRESS": DataFrameSchema(
        {
            "HN": _str(nullable=False),
            "AN": _str(nullable=False),
            "PROGDATE": _str(),
            "SUBJECTIVE": _str(),
            "OBJECTIVE": _str(),
            "ASSESSMENT": _str(),
            "PLAN": _str(),
        }
    ),
    "IPDNRFOCUSDT": DataFrameSchema(
        {
            "HN": _str(nullable=False),
            "AN": _str(nullable=False),
            "PROGRESSDATE": _str(),
            "PROGRESSTIME": _str(),
            "ACTION": _str(),
            "RESPONSE": _str(),
        }
    ),
    "IPTSUMOPRT": DataFrameSchema(
        {
            "AN": _str(nullable=False),
            "ICD9CM": _str(),
            "INDATE": _str(),
            "INTIME": _str(),
        }
    ),
    "INCPT_OPRTACT": DataFrameSchema(
        {
            "HN": _str(nullable=False),
            "AN": _str(nullable=False),
            "INCDATE": _str(),
            "INCTIME": _str(),
            "ORDERCODE": _str(),
            "INCOME": _str(),
            "CANCELDATE": _str(),
            "INCGRP": _str(),
            # OPRTACT side, joined on INCPT.INCOME = OPRTACT.INCOME. The
            # ``O__`` prefix marks the right-hand table of the pre-joined
            # export so collisions with INCPT's own columns (e.g. INCOME,
            # CANCELDATE) stay unambiguous. ``O__ICD9CM`` is the per-
            # operative-act procedure code that issue #69 was filed to
            # surface — without it, IPTSUMOPRT alone silently under-
            # detects cardiac / ortho cohorts. ``O__NAME`` / ``O__NAME_EN``
            # carry the human-readable procedure name for LLM prompts.
            "O__OPRTACT": _str(),
            "O__NAME": _str(),
            "O__NAME_EN": _str(),
            "O__ICD9CM": _str(),
            "O__ICD9CMADD1": _str(),
            "O__ICD9CMADD2": _str(),
            "O__OPRTTYPE": _str(),
            "O__OPRTGRP": _str(),
        }
    ),
    "ICD9CM": DataFrameSchema(
        {
            "ICD9CM": _str(nullable=False),
            "NAME": _str(),
            "ORFLAG": _str(),
        }
    ),
}


def get_schema(table: CSVTable) -> DataFrameSchema:
    """Return the v1 pandera schema for ``table``.

    Raises ``KeyError`` if ``table`` is not a registered CSVTable literal.
    """
    return _REGISTRY_V1[table]


def all_tables() -> tuple[CSVTable, ...]:
    """Return the canonical tuple of all 12 required CSV tables.

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


def validate_header(table: CSVTable, header: list[str]) -> None:
    """Raise :class:`SchemaDriftError` if ``header`` doesn't match the v1 schema.

    Drift goes both ways and both are fatal:

    * **unknown** — a column in the CSV that the schema does not declare.
      Untrusted data could be carried into a downstream join.
    * **missing** — a declared column that the CSV omits. Joins downstream
      would silently null out, which a partial export can produce without
      any other warning.

    Returns ``None`` on a clean header. Callers should invoke this before any
    side-effect (run_id derivation, marker write).
    """
    schema = get_schema(table)
    declared = set(schema.columns)
    present = set(header)
    unknown = sorted(present - declared)
    missing = sorted(declared - present)
    if unknown or missing:
        raise SchemaDriftError(
            f"schema drift in table {table!r}: "
            f"unknown columns {unknown}, missing required columns {missing} "
            f"(declared columns: {sorted(declared)})"
        )
