"""Pure read-only aggregation for the MSBOS reservation pilot report."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict

from bba.audit_store.models import AuditRow, LlmCall
from bba.preop_reservation.models import ReservationReason
from bba.preop_reservation.platelet_evaluate import PlateletReservationReason

_COVERAGE_LIMITATION = (
    "Resolution rate among reservation-terminal rows, not population coverage; "
    "resolved-to-within reservations persist no marker and are invisible to "
    "this read-only report."
)
_ZERO_DENOMINATOR_NOTE = "Rate is null because no eligible terminal rows exist."
_PRECISION_STATUS: Literal["PENDING_CLINICIAN_VALIDATED_SAMPLE"] = (
    "PENDING_CLINICIAN_VALIDATED_SAMPLE"
)

_RBC_MODEL_ID = "msbos-reservation"
_PLATELET_MODEL_ID = "msbos-platelet-reservation"
_RBC_MARKER_KEYS = frozenset({"over_reservation", "operation_unresolved"})
_PLATELET_MARKER_KEYS = frozenset(
    {"platelet_over_reservation", "platelet_reservation_review"}
)
_MARKER_KEYS = _RBC_MARKER_KEYS | _PLATELET_MARKER_KEYS
_OVER_MARKER_KEYS = frozenset({"over_reservation", "platelet_over_reservation"})

_RBC_REASONS = frozenset(get_args(ReservationReason))
_PLATELET_REASONS = frozenset(get_args(PlateletReservationReason))
_CATEGORY_UNRESOLVED_REASONS = frozenset(
    {"uncategorised_procedure", "ambiguous_category"}
)
_OTHER_PLATELET_REVIEW_REASONS = (
    "missing_pre_op_count",
    "no_planned_op",
    "ambiguous_planned_op",
)
_PLATELET_OVER_REASONS = (
    "over_major_non_neuraxial",
    "over_neuraxial",
    "over_cardiac_cpb",
)

# Source of truth: audit_pipeline/replay.py:158. Kept local to avoid the
# preop_reservation <-> audit_pipeline import cycle.
_RETURNS_TERMINALS = frozenset({"RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"})


class PilotReportError(Exception):
    """Persisted pilot data violates a report integrity invariant."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class RbcNoteResolutionRate(_FrozenModel):
    """RBC note-disambiguation resolution among reservation terminals."""

    resolved: int
    unresolved: int
    denominator: int
    rate: float | None
    note: str | None
    limitation: str


class PlateletCategoryResolutionRate(_FrozenModel):
    """Platelet category resolution among reservation terminals."""

    resolved: int
    category_unresolved: int
    denominator: int
    rate: float | None
    note: str | None
    limitation: str


class PlateletOtherReviewBreakdown(_FrozenModel):
    """Platelet review outcomes excluded from category-resolution failures."""

    missing_pre_op_count: int
    no_planned_op: int
    ambiguous_planned_op: int


class CoverageSection(_FrozenModel):
    """Persisted-terminal coverage metrics for both blood-component legs."""

    rbc_note_resolution_rate: RbcNoteResolutionRate
    platelet_category_resolution_rate: PlateletCategoryResolutionRate
    platelet_other_review_reasons: PlateletOtherReviewBreakdown


class RbcPrecisionDenominators(_FrozenModel):
    """RBC over-assertion counts awaiting independent adjudication."""

    none_bucket_over_assertions: int
    note_resolved_over_assertions: int
    over_gm_excess: int
    over_type_and_screen_crossmatched: int
    over_ceiling: int


class PlateletPrecisionDenominators(_FrozenModel):
    """Platelet over-assertion counts awaiting independent adjudication."""

    over_major_non_neuraxial: int
    over_neuraxial: int
    over_cardiac_cpb: int


class PrecisionSection(_FrozenModel):
    """Data-blocked precision structure; deliberately contains no PPV figure."""

    status: Literal["PENDING_CLINICIAN_VALIDATED_SAMPLE"]
    rbc_assertion_denominators: RbcPrecisionDenominators
    platelet_assertion_denominators: PlateletPrecisionDenominators
    rbc_none_and_note_resolved_non_additive: Literal[True]
    overlap_note: str
    pending_note: str


class ReconciliationSection(_FrozenModel):
    """Over-marker versus returns-terminal double-fire check."""

    status: Literal["PASS", "FAIL"]
    over_marker_count: int
    returns_terminal_count: int
    double_fire_count: int
    double_fire_ids: tuple[str, ...]


class Provenance(_FrozenModel):
    """Pinned persisted-data scope for a deterministic report artifact."""

    run_id: str
    code_version: str
    scoped_audit_rows: int
    total_reservation_markers: int


class MsbosReservationPilotReport(_FrozenModel):
    """Committee-facing, read-only MSBOS reservation pilot report."""

    provenance: Provenance
    coverage: CoverageSection
    precision: PrecisionSection
    reconciliation: ReconciliationSection


@dataclass(frozen=True)
class _Marker:
    audit_id: str
    key: str
    reason: str
    note_resolved: bool = False


def reconcile_returns(
    over_marker_ids: set[str], returns_terminal_ids: set[str]
) -> ReconciliationSection:
    """Reconcile independently supplied over-marker and returns-terminal ids."""
    overlap = tuple(sorted(over_marker_ids & returns_terminal_ids))
    return ReconciliationSection(
        status="FAIL" if overlap else "PASS",
        over_marker_count=len(over_marker_ids),
        returns_terminal_count=len(returns_terminal_ids),
        double_fire_count=len(overlap),
        double_fire_ids=overlap,
    )


def _required(payload: Mapping[str, Any], key: str, expected: type[object]) -> Any:
    if key not in payload or type(payload[key]) is not expected:
        raise PilotReportError(f"marker field {key!r} must be {expected.__name__}")
    return payload[key]


def _validate_marker_key(call: LlmCall) -> str:
    payload: Mapping[str, Any] = call.request_json
    true_keys = tuple(key for key in _MARKER_KEYS if payload.get(key) is True)
    if len(true_keys) != 1:
        raise PilotReportError(
            f"marker call {call.call_id!r} must have exactly one true marker key"
        )
    (key,) = true_keys
    expected_keys = {
        _RBC_MODEL_ID: _RBC_MARKER_KEYS,
        _PLATELET_MODEL_ID: _PLATELET_MARKER_KEYS,
    }.get(call.model_id)
    if expected_keys is None or key not in expected_keys:
        raise PilotReportError(
            f"marker key {key!r} does not match model_id {call.model_id!r}"
        )
    if payload.get("audit_id") != call.audit_id:
        raise PilotReportError(
            f"marker payload audit_id does not match call audit_id {call.audit_id!r}"
        )
    return key


def _parse_marker(call: LlmCall) -> _Marker:
    payload: Mapping[str, Any] = call.request_json
    key = _validate_marker_key(call)
    reason = _required(payload, "reason", str)
    _required(payload, "resolved_icd9", str)
    if key in _RBC_MARKER_KEYS:
        note_resolved = _required(payload, "note_resolved", bool)
        if reason not in _RBC_REASONS:
            raise PilotReportError(f"unknown RBC reservation reason {reason!r}")
        return _Marker(call.audit_id, key, reason, note_resolved)
    _required(payload, "category", str)
    count = payload.get("pre_op_count_k_ul")
    if "pre_op_count_k_ul" not in payload or (
        count is not None and type(count) is not float
    ):
        raise PilotReportError("marker field 'pre_op_count_k_ul' must be float or None")
    _required(payload, "reserved_units", int)
    _required(payload, "clinician_signed", bool)
    if reason not in _PLATELET_REASONS:
        raise PilotReportError(f"unknown platelet reservation reason {reason!r}")
    return _Marker(call.audit_id, key, reason)


def _marker_calls(calls: Sequence[LlmCall]) -> tuple[LlmCall, ...]:
    return tuple(call for call in calls if call.model_id.startswith("msbos-"))


def _validate_scope(
    rows: Sequence[AuditRow], calls: Sequence[LlmCall]
) -> tuple[dict[str, AuditRow], tuple[LlmCall, ...]]:
    row_counts = Counter(row.audit_id for row in rows)
    duplicate_rows = sorted(audit_id for audit_id, n in row_counts.items() if n > 1)
    if duplicate_rows:
        raise PilotReportError(f"duplicate scoped audit row ids: {duplicate_rows}")
    rows_by_id = {row.audit_id: row for row in rows}
    markers = _marker_calls(calls)
    orphans = sorted({call.audit_id for call in markers} - rows_by_id.keys())
    if orphans:
        raise PilotReportError(f"orphan reservation marker audit ids: {orphans}")
    marker_counts = Counter(call.audit_id for call in markers)
    duplicates = sorted(audit_id for audit_id, n in marker_counts.items() if n > 1)
    if duplicates:
        raise PilotReportError(f"duplicate reservation marker audit ids: {duplicates}")
    return rows_by_id, markers


def _validate_join(marker: _Marker, row: AuditRow) -> None:
    is_over = marker.key in _OVER_MARKER_KEYS
    expected_final = "PREOP_OVER_RESERVATION" if is_over else "NEEDS_REVIEW"
    # An over-marker whose committed row is a returns terminal is the exact
    # double-fire the returns reconciliation exists to REPORT (a returned /
    # peri-op-exempt order also billed as over-reservation). Let it through so
    # reconcile_returns surfaces it as FAIL + double_fire_ids instead of
    # aborting the whole gating report on a generic integrity error. Any OTHER
    # final-classification mismatch is still corruption and fails loud.
    returns_double_fire = is_over and row.final_classification in _RETURNS_TERMINALS
    if row.final_classification != expected_final and not returns_double_fire:
        raise PilotReportError(
            f"marker {marker.audit_id!r} requires final_classification "
            f"{expected_final!r}, got {row.final_classification!r}"
        )
    expected_component = "red_cell" if marker.key in _RBC_MARKER_KEYS else "platelet"
    if row.component != expected_component:
        raise PilotReportError(
            f"marker {marker.audit_id!r} requires component "
            f"{expected_component!r}, got {row.component!r}"
        )


def _rate(numerator: int, denominator: int) -> tuple[float | None, str | None]:
    if denominator == 0:
        return None, _ZERO_DENOMINATOR_NOTE
    return numerator / denominator, None


def _rbc_coverage(markers: Sequence[_Marker]) -> RbcNoteResolutionRate:
    rbc_resolved = sum(
        marker.key == "over_reservation" and marker.note_resolved for marker in markers
    )
    rbc_unresolved = sum(marker.key == "operation_unresolved" for marker in markers)
    rbc_denominator = rbc_resolved + rbc_unresolved
    rbc_rate, rbc_note = _rate(rbc_resolved, rbc_denominator)
    return RbcNoteResolutionRate(
        resolved=rbc_resolved,
        unresolved=rbc_unresolved,
        denominator=rbc_denominator,
        rate=rbc_rate,
        note=rbc_note,
        limitation=_COVERAGE_LIMITATION,
    )


def _platelet_coverage(
    markers: Sequence[_Marker],
) -> tuple[PlateletCategoryResolutionRate, PlateletOtherReviewBreakdown]:
    platelet_resolved = sum(
        marker.key == "platelet_over_reservation" for marker in markers
    )
    category_unresolved = sum(
        marker.key == "platelet_reservation_review"
        and marker.reason in _CATEGORY_UNRESOLVED_REASONS
        for marker in markers
    )
    platelet_denominator = platelet_resolved + category_unresolved
    platelet_rate, platelet_note = _rate(platelet_resolved, platelet_denominator)
    other = Counter(
        marker.reason
        for marker in markers
        if marker.key == "platelet_reservation_review"
        and marker.reason in _OTHER_PLATELET_REVIEW_REASONS
    )
    rate = PlateletCategoryResolutionRate(
        resolved=platelet_resolved,
        category_unresolved=category_unresolved,
        denominator=platelet_denominator,
        rate=platelet_rate,
        note=platelet_note,
        limitation=_COVERAGE_LIMITATION,
    )
    breakdown = PlateletOtherReviewBreakdown(
        **{reason: other[reason] for reason in _OTHER_PLATELET_REVIEW_REASONS}
    )
    return rate, breakdown


def _coverage(markers: Sequence[_Marker]) -> CoverageSection:
    platelet_rate, platelet_other = _platelet_coverage(markers)
    return CoverageSection(
        rbc_note_resolution_rate=_rbc_coverage(markers),
        platelet_category_resolution_rate=platelet_rate,
        platelet_other_review_reasons=platelet_other,
    )


def _precision(markers: Sequence[_Marker]) -> PrecisionSection:
    rbc_over_reasons = Counter(
        marker.reason for marker in markers if marker.key == "over_reservation"
    )
    platelet_over_reasons = Counter(
        marker.reason for marker in markers if marker.key == "platelet_over_reservation"
    )
    note_resolved = sum(
        marker.key == "over_reservation" and marker.note_resolved for marker in markers
    )
    return PrecisionSection(
        status=_PRECISION_STATUS,
        rbc_assertion_denominators=RbcPrecisionDenominators(
            none_bucket_over_assertions=sum(
                marker.key == "over_reservation" and marker.reason == "over_none"
                for marker in markers
            ),
            note_resolved_over_assertions=note_resolved,
            over_gm_excess=rbc_over_reasons["over_gm_excess"],
            over_type_and_screen_crossmatched=rbc_over_reasons[
                "over_type_and_screen_crossmatched"
            ],
            over_ceiling=rbc_over_reasons["over_ceiling"],
        ),
        platelet_assertion_denominators=PlateletPrecisionDenominators(
            **{
                reason: platelet_over_reasons[reason]
                for reason in _PLATELET_OVER_REASONS
            }
        ),
        rbc_none_and_note_resolved_non_additive=True,
        overlap_note=(
            "The RBC none-bucket and note-resolved denominators overlap and are "
            "non-additive; an over_none assertion can be note-resolved. Do not sum them."
        ),
        pending_note=(
            "Precision/PPV remains pending until a versioned, independently "
            "adjudicated clinician-signed sample is available."
        ),
    )


def build_pilot_report(
    rows: Sequence[AuditRow],
    calls: Sequence[LlmCall],
    *,
    run_id: str,
    code_version: str,
) -> MsbosReservationPilotReport:
    """Validate and aggregate persisted pilot data without recomputing verdicts."""
    rows_by_id, marker_calls = _validate_scope(rows, calls)
    markers = tuple(_parse_marker(call) for call in marker_calls)
    for marker in markers:
        _validate_join(marker, rows_by_id[marker.audit_id])
    over_marker_ids = {
        marker.audit_id for marker in markers if marker.key in _OVER_MARKER_KEYS
    }
    returns_terminal_ids = {
        row.audit_id for row in rows if row.final_classification in _RETURNS_TERMINALS
    }
    return MsbosReservationPilotReport(
        provenance=Provenance(
            run_id=run_id,
            code_version=code_version,
            scoped_audit_rows=len(rows),
            total_reservation_markers=len(markers),
        ),
        coverage=_coverage(markers),
        precision=_precision(markers),
        reconciliation=reconcile_returns(over_marker_ids, returns_terminal_ids),
    )


__all__ = [
    "CoverageSection",
    "MsbosReservationPilotReport",
    "PilotReportError",
    "PrecisionSection",
    "ReconciliationSection",
    "build_pilot_report",
    "reconcile_returns",
]
