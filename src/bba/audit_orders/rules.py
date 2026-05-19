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
    """Reject records with a non-null, non-empty ``canceldate``.

    Per the issue: ``cancelled (CANCELDATE not null)`` is hard-excluded.

    A whitespace-only or empty string is treated as the CSV missing-value
    sentinel (consistent with PRD §1's strict-loud convention) — an empty
    cell in HOSxP exports means "not cancelled", not "cancelled with no
    date". Only a non-empty value triggers the exclusion.
    """
    if record.canceldate is None or record.canceldate.strip() == "":
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


def _code_matches_prefix(code: str, prefix: str) -> bool:
    """True iff ``code`` belongs to the ICD-10 chapter denoted by ``prefix``.

    ICD-10 codes follow ``<letter><digit><digit>[.subcategory]`` (e.g.,
    ``D55``, ``D55.0``). A "chapter prefix" can be one of:

    * single letter (e.g., ``"O"``) — matches any code starting with that
      letter followed by a digit (the 3-char category form);
    * 3-char category code (e.g., ``"D55"``) — matches the bare code or
      the code followed by ``"."`` and a subcategory;
    * subcategory code (e.g., ``"M31.1"``) — matches the bare code or
      the code followed by ``"."`` and further subdivisions.

    The boundary check rejects malformed near-misses like ``"D550"``
    (which is NOT ``D55`` + dot — it's a different chapter), per the
    Codex review of issue #4. A raw ``startswith`` would collapse the
    boundary and silently broaden the hard-exclusion set.
    """
    if not code.startswith(prefix):
        return False
    if len(code) == len(prefix):
        return True
    next_char = code[len(prefix)]
    if next_char == ".":
        return True
    # Single-letter chapter prefixes (e.g., "O") can be continued by a
    # digit to form the 3-char category (O00, O80, etc.). Multi-char
    # prefixes do not get the digit-continuation pass — that's what
    # disambiguates "D550" from a real D55 subcategory.
    if len(prefix) == 1 and next_char.isdigit():
        return True
    return False


def _first_matching_code(
    codes: tuple[str, ...], prefixes: frozenset[str]
) -> str | None:
    """Return the first ICD-10 code in ``codes`` that matches any of
    ``prefixes`` under the boundary rules of :func:`_code_matches_prefix`,
    or ``None`` if no match.

    Case-sensitive: ICD-10 is uppercase by convention, and tolerating
    lowercase would also tolerate other formatting drift (e.g., trailing
    whitespace, half-width digits) that we have not explicitly opted
    in to.
    """
    for code in codes:
        for prefix in prefixes:
            if _code_matches_prefix(code, prefix):
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
    "OBSTETRIC_PREFIX",
    "RBC_PRODUCTS",
    "TMA_PREFIXES",
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
