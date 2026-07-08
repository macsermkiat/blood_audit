"""Pydantic v2 models for the audit_orders module — input rows, filter output, and exclusions.

All models are immutable (``frozen=True``). The module's contract is a pure
function from a sequence of :class:`BloodOrderInput` to a :class:`FilterResult`
that partitions every input into either an :class:`AuditOrder` (included) or
an :class:`ExcludedRecord` (excluded with a typed reason). No record is silently
dropped — the total-coverage invariant is asserted by the test suite.

Field groups on :class:`AuditOrder` match PRD §"Output schema" → Identity +
Anchor: ``audit_id``, ``hn``, ``an``, ``reqno``, ``order_datetime``,
``anchor_imputed``, ``products_ordered``. Downstream tickets (#5–#7) join
labs / vitals / cohort on ``hn`` + ``an`` + ``order_datetime``; later tickets
(#9, #19) replace ``hn``/``an`` with ``hn_hash``/``an_hash`` before persistence.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from bba.ingest.models import ParsedTimeOfDay


# Inclusion product allow-list (PRD §2 + issue #4): the three RBC products
# present in the BDTYPE table. Anything else fails the product-inclusion gate
# with reason ``"not_rbc_product"``.
RBCProduct = Literal["LPRC", "LDPRC", "SDR"]


# Blood-component family of an admitted order (Phase 2). Mirrors
# ``bba.audit_store.models.Component`` — kept local to avoid coupling
# audit_orders (a foundational intake module) to audit_store.
# ``red_cell`` is the default so existing RBC orders require no change.
Component = Literal["red_cell", "platelet"]


# Typed reason for excluding a (HN, REQNO). One enum value per excluded
# subgroup named in the issue acceptance criteria; the ``detail`` field on
# :class:`ExcludedRecord` carries the specific ICD-10 code (e.g., "D55.1")
# when the reason category is broader than a single code.
#
# ``"pediatric"`` is preserved as a back-compat value for runs that pre-date
# the 2026-05-19 schema lock (when the audit still received birthdate from a
# patient-demographics table). Current ingest bundles have no per-row age
# column, so the upstream IT pre-filter (``age > 15``) handles the gate
# exclusively and the audit pipeline never fires ``"pediatric"`` itself.
ExclusionReason = Literal[
    "not_rbc_product",
    "status_not_eligible",
    "cancelled",
    "pediatric",
    "no_an",
    "inter_hospital",
    "hemoglobinopathy",
]


class BloodOrderInput(BaseModel):
    """One pre-joined (HN, REQNO) input row.

    Carries the BDVST identity + status fields, the anchor candidates from
    both BDVST (``req_date`` / ``req_time``) and BDVSTDT (``bdvst_date`` /
    ``bdvst_time``), the joined BDTYPE products, and the joined Diagnosis
    ICD-10 codes scoped to the same admission (AN).

    The input shape is "already joined" so the filter logic does not have to
    perform DuckDB joins itself — that is the orchestrator's job. Keeping the
    filter pure-Python over a flat input model makes every rule unit-testable
    without standing up DuckDB in the test, and it keeps the module's
    public surface honest: the input is everything the filter needs, the
    output is exactly what the canonical table holds.
    """

    model_config = ConfigDict(frozen=True)

    # BDVST identity + gates
    hn: str
    an: str | None
    reqno: str
    bdvstst: str
    reqtype: str
    canceldate: str | None

    # Anchor candidates (primary: REQ; fallback: BDVST)
    req_date: date | None
    req_time: ParsedTimeOfDay | None
    bdvst_date: date | None
    bdvst_time: ParsedTimeOfDay | None

    # Joined products from BDTYPE (per REQNO)
    products: tuple[str, ...]

    # Joined ICD-10 codes from Diagnosis (AN-scoped)
    diagnosis_codes: tuple[str, ...]


class AuditOrder(BaseModel):
    """One canonical audit_orders row — output of the filter for an included input.

    Identity (``audit_id``) is deterministic: same ``(hn, reqno)`` always
    yields the same ``audit_id`` across runs, which is the prerequisite for
    run-level idempotency in downstream stages (#9, #19, #24).

    Anchor (``order_datetime``, ``anchor_imputed``): ``order_datetime`` is
    tz-aware UTC; ``anchor_imputed`` is True iff the primary REQ pair was
    unusable and the BDVST pair supplied the moment. Per the PRD's tz
    contract, naive datetimes are forbidden on the audit-side of the system.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: str
    hn: str
    an: str
    reqno: str
    order_datetime: datetime
    anchor_imputed: bool
    # Widened from tuple[RBCProduct, ...] in Phase 2 to accommodate platelet
    # product codes. RBC orders still carry exactly the same string values as
    # before (LPRC / LDPRC / SDR); only platelet-only orders carry platelet codes.
    products_ordered: tuple[str, ...]
    diagnosis_codes: tuple[str, ...]
    # Component axis (Phase 2). Defaults to "red_cell" so existing RBC order
    # construction is byte-identical — callers that do not set this field get
    # the same AuditOrder they would have before Phase 2.
    component: Component = "red_cell"


class ExcludedRecord(BaseModel):
    """One (HN, REQNO) that the filter dropped, with the reason and optional detail.

    ``reason`` is the broad category (matches the issue's excluded-subgroup
    list one-to-one). ``detail`` carries the specific evidence that fired
    the rule: e.g. the ICD-10 code that matched a hemoglobinopathy block,
    or the BDVSTST sentinel that failed the status gate. Reviewers reading
    the excluded set should be able to audit the rule trigger without
    re-joining the source CSVs.
    """

    model_config = ConfigDict(frozen=True)

    hn: str
    reqno: str
    reason: ExclusionReason
    detail: str | None


class FilterResult(BaseModel):
    """Outcome of :func:`bba.audit_orders.build_audit_orders`.

    Invariant (asserted by the test suite): every input record lands in
    exactly one of ``included`` or ``excluded``. The two tuples together
    partition the input set; no record is silently dropped.

    The orderings are stable across runs given a stable input ordering —
    the filter is a pure transformation, not a set operation that loses
    insertion order.
    """

    model_config = ConfigDict(frozen=True)

    included: tuple[AuditOrder, ...]
    excluded: tuple[ExcludedRecord, ...]


class AuditOrdersConfig(BaseModel):
    """Configuration for a single :func:`build_audit_orders` invocation.

    ``code_version`` is reserved for future participation in a per-row
    idempotency contract analogous to :class:`bba.ingest.RunIdentity`; at
    Phase-1 RED the field exists so callers wire it through without a later
    breaking change.

    ``tz_source`` is the wall-clock zone of the HOSxP anchor columns
    (``Asia/Bangkok``); :class:`~bba.ingest.RowTimestamp` consumes it when
    combining a date + parsed time into a UTC moment.
    """

    model_config = ConfigDict(frozen=True)

    code_version: str
    tz_source: str = "Asia/Bangkok"


__all__: Sequence[str] = (
    "AuditOrder",
    "AuditOrdersConfig",
    "BloodOrderInput",
    "Component",
    "ExcludedRecord",
    "ExclusionReason",
    "FilterResult",
    "RBCProduct",
)
