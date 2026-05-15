"""Per-rule inclusion/exclusion predicates over a :class:`BloodOrderInput`.

The filter pipeline applies the rules in a fixed order (see
:mod:`bba.audit_orders.pipeline`). Each predicate here answers a single
yes/no question against one input record and returns the typed
:class:`bba.audit_orders.models.ExcludedRecord` it would emit — or ``None``
if the rule does not exclude.

Rules implemented as separate functions rather than a single mega-predicate
because each acceptance criterion in issue #4 names a specific subgroup,
and the test suite asserts one predicate at a time. Composing them in a
fixed order in :mod:`pipeline` keeps the public surface a single function
while letting tests target the exact rule the issue calls out.
"""

from __future__ import annotations

from datetime import date
from typing import cast

from bba.audit_orders.models import (
    BloodOrderInput,
    ExcludedRecord,
    RBCProduct,
)

# Allow-listed RBC products per PRD §2 / issue #4 acceptance criteria.
RBC_PRODUCTS: frozenset[str] = frozenset({"LPRC", "LDPRC", "SDR"})

# BDVSTST values eligible for audit per PRD §2. Status 6 is "refused" and
# every other code (cancelled / never-issued / etc.) is out of scope for
# Phase 1.
ELIGIBLE_STATUS: frozenset[str] = frozenset({"4", "5"})

# ICD-10 3-character prefixes that hard-exclude under the hemoglobinopathy
# rule. Per issue #4 AC, each of D55/D56/D57/D58 must produce its own
# golden-fixture test. (Round 2 clinical-agent G6PD discussion is documented
# in the issue references; the AC list is the source of truth and lists
# D55 in the excluded set.)
HEMOGLOBINOPATHY_PREFIXES: frozenset[str] = frozenset({"D55", "D56", "D57", "D58"})

# ICD-10 prefix for autoimmune hemolytic anemia (AIHA), hard-excluded per
# issue #4. The ``.x`` wildcard means any D59 subcode (D59.0 / D59.1 / D59.2
# / D59.3 / D59.4 / D59.8 / D59.9) is excluded.
AIHA_PREFIX: str = "D59"

# ICD-10 prefixes for thrombotic microangiopathy (TMA) cohorts. M31.1 is
# thrombotic thrombocytopenic purpura (TTP). M31.0 is hypersensitivity
# angiitis and is intentionally NOT in the set.
TMA_PREFIXES: frozenset[str] = frozenset({"M31.1"})

# ICD-10 letter for obstetric conditions (O00–O9A). One-letter prefix
# match is sufficient: the entire O-chapter is "Pregnancy, childbirth and
# the puerperium" and out of scope for Phase 1.
OBSTETRIC_PREFIX: str = "O"

# Minimum age in years for Phase 1 inclusion (PRD §"Out of scope":
# pediatric < 15 is excluded).
MIN_AGE_YEARS: int = 15


def is_rbc_product(product: str) -> bool:
    """True iff ``product`` is one of the allow-listed RBC products."""
    return product in RBC_PRODUCTS


def rbc_products_in(products: tuple[str, ...]) -> tuple[RBCProduct, ...]:
    """Return the subset of ``products`` that are allow-listed RBC products,
    preserving input order.
    """
    # cast is safe because the membership check restricts to the RBCProduct
    # literal members; mypy cannot narrow `str` to `Literal["LPRC", ...]`
    # from a runtime set check.
    return tuple(cast("RBCProduct", p) for p in products if p in RBC_PRODUCTS)


def check_rbc_product(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject records whose products are entirely outside :data:`RBC_PRODUCTS`.

    A record with at least one RBC product passes. Mixed-product orders
    (one RBC + one non-RBC) are passed at this gate; downstream stages
    care about which RBC product was actually issued.
    """
    if any(is_rbc_product(p) for p in record.products):
        return None
    return ExcludedRecord(
        hn=record.hn,
        reqno=record.reqno,
        reason="not_rbc_product",
        detail=",".join(record.products) if record.products else None,
    )


def check_status(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject records whose ``bdvstst`` is not in :data:`ELIGIBLE_STATUS`.

    Status 6 (refused) and every other sentinel are out of scope per the
    issue's excluded-subgroup list (``refused (BDVSTST=6)``).
    """
    if record.bdvstst in ELIGIBLE_STATUS:
        return None
    return ExcludedRecord(
        hn=record.hn,
        reqno=record.reqno,
        reason="status_not_eligible",
        detail=record.bdvstst,
    )


def check_cancelled(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject records with a non-null ``canceldate``.

    Per the issue: ``cancelled (CANCELDATE not null)`` is hard-excluded.
    """
    if record.canceldate is None:
        return None
    return ExcludedRecord(
        hn=record.hn,
        reqno=record.reqno,
        reason="cancelled",
        detail=record.canceldate,
    )


def check_an_scoped(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject records without an ``AN`` (admission number) — i.e., OPD.

    Outpatient transfusions are out of scope: ``OPD (AN=null)``. An empty
    or whitespace-only AN is also treated as missing — an empty string in
    a CSV column is the missing-value sentinel, not a real admission.
    """
    if record.an is not None and record.an.strip() != "":
        return None
    return ExcludedRecord(
        hn=record.hn,
        reqno=record.reqno,
        reason="no_an",
        detail=None,
    )


def check_request_type(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject records whose ``reqtype`` is not ``'P'`` (in-house request).

    ``REQTYPE='H'`` is an inter-hospital referral: clinical context lives
    outside KCMH and is out of scope. Any other non-``P`` value is also
    excluded under the same reason — Phase-1 conservative.
    """
    if record.reqtype == "P":
        return None
    return ExcludedRecord(
        hn=record.hn,
        reqno=record.reqno,
        reason="inter_hospital",
        detail=record.reqtype,
    )


def years_between(start: date, end: date) -> int:
    """Whole years from ``start`` to ``end`` (i.e., age in years).

    The boundary is open: a birthday on ``end`` counts as the new age; a
    birthday after ``end`` (same calendar year) counts as the previous age.
    """
    years = end.year - start.year
    if (end.month, end.day) < (start.month, start.day):
        years -= 1
    return years


def check_age(record: BloodOrderInput, anchor_date: date) -> ExcludedRecord | None:
    """Reject records whose patient age at the anchor date is < :data:`MIN_AGE_YEARS`.

    ``anchor_date`` is the date component of the resolved anchor datetime —
    age is computed relative to the order, not to today. A null birthdate
    is treated as missing-eligibility; per Phase-1 conservatism it counts
    as a pediatric exclusion.
    """
    if record.birthdate is None:
        return ExcludedRecord(
            hn=record.hn,
            reqno=record.reqno,
            reason="pediatric",
            detail=None,
        )
    age = years_between(record.birthdate, anchor_date)
    if age < MIN_AGE_YEARS:
        return ExcludedRecord(
            hn=record.hn,
            reqno=record.reqno,
            reason="pediatric",
            detail=str(age),
        )
    return None


def _first_matching_code(
    codes: tuple[str, ...], prefixes: frozenset[str]
) -> str | None:
    """Return the first ICD-10 code in ``codes`` that starts with any of
    ``prefixes``, or ``None`` if no match.

    Case-sensitive ``str.startswith``: ICD-10 is uppercase by convention,
    and tolerating lowercase would also tolerate other formatting drift
    (e.g., trailing whitespace, half-width digits) that we have not
    explicitly opted in to.
    """
    for code in codes:
        for prefix in prefixes:
            if code.startswith(prefix):
                return code
    return None


def check_hemoglobinopathy(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject if any diagnosis code prefix matches :data:`HEMOGLOBINOPATHY_PREFIXES`.

    The matched code is carried in ``ExcludedRecord.detail`` so the test
    suite can distinguish D55 vs D56 vs D57 vs D58 fixtures (the issue
    AC requires one fixture per code).
    """
    match = _first_matching_code(record.diagnosis_codes, HEMOGLOBINOPATHY_PREFIXES)
    if match is None:
        return None
    return ExcludedRecord(
        hn=record.hn,
        reqno=record.reqno,
        reason="hemoglobinopathy",
        detail=match,
    )


def check_aiha(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject if any diagnosis code starts with :data:`AIHA_PREFIX` (``D59``).

    The matched code is carried in ``ExcludedRecord.detail``.
    """
    match = _first_matching_code(record.diagnosis_codes, frozenset({AIHA_PREFIX}))
    if match is None:
        return None
    return ExcludedRecord(
        hn=record.hn,
        reqno=record.reqno,
        reason="aiha",
        detail=match,
    )


def check_tma(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject if any diagnosis code prefix matches :data:`TMA_PREFIXES`.

    The matched code is carried in ``ExcludedRecord.detail``.
    """
    match = _first_matching_code(record.diagnosis_codes, TMA_PREFIXES)
    if match is None:
        return None
    return ExcludedRecord(
        hn=record.hn,
        reqno=record.reqno,
        reason="tma",
        detail=match,
    )


def check_obstetric(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject if any diagnosis code starts with :data:`OBSTETRIC_PREFIX` (``O``).

    The whole ICD-10 O-chapter is out of scope per PRD §"Out of scope".
    The matched code is carried in ``ExcludedRecord.detail``.
    """
    match = _first_matching_code(record.diagnosis_codes, frozenset({OBSTETRIC_PREFIX}))
    if match is None:
        return None
    return ExcludedRecord(
        hn=record.hn,
        reqno=record.reqno,
        reason="obstetric",
        detail=match,
    )


__all__ = (
    "AIHA_PREFIX",
    "ELIGIBLE_STATUS",
    "HEMOGLOBINOPATHY_PREFIXES",
    "MIN_AGE_YEARS",
    "OBSTETRIC_PREFIX",
    "RBC_PRODUCTS",
    "TMA_PREFIXES",
    "check_age",
    "check_aiha",
    "check_an_scoped",
    "check_cancelled",
    "check_hemoglobinopathy",
    "check_obstetric",
    "check_rbc_product",
    "check_request_type",
    "check_status",
    "check_tma",
    "is_rbc_product",
    "rbc_products_in",
    "years_between",
)
