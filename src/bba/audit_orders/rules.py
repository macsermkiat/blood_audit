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
# thrombotic thrombocytopenic purpura (TTP); broader M31 is the
# polyarteritis/related-conditions block that includes TTP-HUS.
TMA_PREFIXES: frozenset[str] = frozenset({"M31.1"})

# ICD-10 letter for obstetric conditions (O00–O9A). One-letter prefix
# match is sufficient: the entire O-chapter is "Pregnancy, childbirth and
# the puerperium" and out of scope for Phase 1.
OBSTETRIC_PREFIX: str = "O"

# Minimum age in years for Phase 1 inclusion (PRD §"Out of scope":
# pediatric < 15 is excluded).
MIN_AGE_YEARS: int = 15


def check_rbc_product(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject records whose products are entirely outside :data:`RBC_PRODUCTS`.

    A record with at least one RBC product passes. Mixed-product orders
    (one RBC + one non-RBC) are passed at this gate; downstream stages
    care about which RBC product was actually issued.
    """
    raise NotImplementedError


def check_status(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject records whose ``bdvstst`` is not in :data:`ELIGIBLE_STATUS`.

    Status 6 (refused) and every other sentinel are out of scope per the
    issue's excluded-subgroup list (``refused (BDVSTST=6)``).
    """
    raise NotImplementedError


def check_cancelled(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject records with a non-null ``canceldate``.

    Per the issue: ``cancelled (CANCELDATE not null)`` is hard-excluded.
    """
    raise NotImplementedError


def check_an_scoped(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject records without an ``AN`` (admission number) — i.e., OPD.

    Outpatient transfusions are out of scope: ``OPD (AN=null)``.
    """
    raise NotImplementedError


def check_request_type(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject records whose ``reqtype`` is not ``'P'`` (in-house request).

    ``REQTYPE='H'`` is an inter-hospital referral: clinical context lives
    outside KCMH and is out of scope.
    """
    raise NotImplementedError


def check_age(record: BloodOrderInput, anchor_date: date) -> ExcludedRecord | None:
    """Reject records whose patient age at the anchor date is < :data:`MIN_AGE_YEARS`.

    ``anchor_date`` is the date component of the resolved anchor datetime —
    age is computed relative to the order, not to today. A null birthdate
    is treated as missing-eligibility; per Phase-1 conservatism it counts
    as a pediatric exclusion (the dashboard surfaces these for review).
    """
    raise NotImplementedError


def check_hemoglobinopathy(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject if any diagnosis code prefix matches :data:`HEMOGLOBINOPATHY_PREFIXES`.

    The matched code is carried in ``ExcludedRecord.detail`` so the test
    suite can distinguish D55 vs D56 vs D57 vs D58 fixtures (the issue
    AC requires one fixture per code).
    """
    raise NotImplementedError


def check_aiha(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject if any diagnosis code starts with :data:`AIHA_PREFIX` (``D59``).

    The matched code is carried in ``ExcludedRecord.detail``.
    """
    raise NotImplementedError


def check_tma(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject if any diagnosis code prefix matches :data:`TMA_PREFIXES`.

    The matched code is carried in ``ExcludedRecord.detail``.
    """
    raise NotImplementedError


def check_obstetric(record: BloodOrderInput) -> ExcludedRecord | None:
    """Reject if any diagnosis code starts with :data:`OBSTETRIC_PREFIX` (``O``).

    The whole ICD-10 O-chapter is out of scope per PRD §"Out of scope".
    The matched code is carried in ``ExcludedRecord.detail``.
    """
    raise NotImplementedError


def is_rbc_product(product: str) -> bool:
    """True iff ``product`` is one of the allow-listed RBC products."""
    raise NotImplementedError


def rbc_products_in(products: tuple[str, ...]) -> tuple[RBCProduct, ...]:
    """Return the subset of ``products`` that are allow-listed RBC products,
    preserving input order.
    """
    raise NotImplementedError


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
)
