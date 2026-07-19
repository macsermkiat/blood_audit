"""Single home for classifying an MSBOS reservation reason into a render bucket.

The canonical reason vocabularies are the library Literals ``ReservationReason``
(RBC, :mod:`bba.preop_reservation.models`) and ``PlateletReservationReason``
(platelet, :mod:`bba.preop_reservation.platelet_evaluate`). The pilot renderer
``scripts/pilot/build_review.py`` groups those reasons into scannable count
buckets and pill colours; this module owns the ``reason -> bucket`` classification
so the pilot cannot silently misbucket a newly added reason. A reason added to
either Literal without a bucket here fails ``_validate_exhaustive`` at import.
"""

from __future__ import annotations

from typing import Literal, get_args

from bba.preop_reservation.models import ReservationReason
from bba.preop_reservation.platelet_evaluate import PlateletReservationReason

Bucket = Literal["above", "within", "within_ceiling", "unresolved"]

# Pilot-injected reason (scripts/pilot/run_pipeline.py) that is rendered by
# build_review but is NOT a member of either Literal.
LEG_LOCAL_REASONS: tuple[str, ...] = ("reservation_lookup_miss",)

RESERVATION_REASON_VOCAB: frozenset[str] = (
    frozenset(get_args(ReservationReason))
    | frozenset(get_args(PlateletReservationReason))
    | frozenset(LEG_LOCAL_REASONS)
)

# reason -> bucket. Exhaustive over RESERVATION_REASON_VOCAB (enforced below).
RESERVATION_REASON_BUCKET: dict[str, Bucket] = {
    # above — a reserved quantity exceeds the MSBOS tariff / platelet cutoff
    "over_none": "above",
    "over_gm_excess": "above",
    "over_type_and_screen_crossmatched": "above",
    "over_ceiling": "above",
    "over_major_non_neuraxial": "above",
    "over_neuraxial": "above",
    "over_cardiac_cpb": "above",
    # within_ceiling — its own scannable bucket (the shadow-over exposure, #210/#214)
    "within_ceiling": "within_ceiling",
    # within — at/below tariff, or nothing to judge
    "within_recommendation": "within",
    "type_and_screen_screen_only": "within",
    "within_major_non_neuraxial": "within",
    "within_neuraxial": "within",
    "within_cardiac_cpb": "within",
    "no_reserved_units": "within",
    # unresolved — code/op/category/count could not be resolved (warn, not an over)
    "unresolved_code": "unresolved",
    "ambiguous_code": "unresolved",
    "operation_unresolved": "unresolved",
    "no_planned_op": "unresolved",
    "ambiguous_planned_op": "unresolved",
    "uncategorised_procedure": "unresolved",
    "ambiguous_category": "unresolved",
    "missing_pre_op_count": "unresolved",
    "reservation_lookup_miss": "unresolved",
}


def _validate_exhaustive() -> None:
    """Fail loud at import if the bucket table drifts from the reason vocabulary."""
    keys = frozenset(RESERVATION_REASON_BUCKET)
    missing = RESERVATION_REASON_VOCAB - keys
    extra = keys - RESERVATION_REASON_VOCAB
    if missing or extra:
        raise ValueError(
            "RESERVATION_REASON_BUCKET is out of sync with the reason vocabulary: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )


_validate_exhaustive()

# Derived membership sets for consumers that classify by set (build_review).
# WITHIN_REASONS deliberately EXCLUDES within_ceiling (its own bucket); every
# build_review use of the "within" set is guarded by an earlier within_ceiling
# branch, so the exclusion is behaviour-identical to the historical frozenset.
ABOVE_REASONS: frozenset[str] = frozenset(
    reason for reason, bucket in RESERVATION_REASON_BUCKET.items() if bucket == "above"
)
WITHIN_REASONS: frozenset[str] = frozenset(
    reason for reason, bucket in RESERVATION_REASON_BUCKET.items() if bucket == "within"
)

PILL_OK = "cls-msbos-ok"
PILL_WARN = "cls-msbos-warn"

_UNRESOLVED: Bucket = "unresolved"


def bucket_for(reason: str) -> Bucket:
    """Bucket for a reservation reason; unknown/blank strings -> ``"unresolved"``."""
    return RESERVATION_REASON_BUCKET.get(reason, _UNRESOLVED)


def pill_class_for_bucket(bucket: Bucket) -> str:
    """Pill colour class for a bucket: within/within_ceiling read OK, else WARN."""
    return PILL_OK if bucket in ("within", "within_ceiling") else PILL_WARN


__all__ = [
    "ABOVE_REASONS",
    "Bucket",
    "LEG_LOCAL_REASONS",
    "PILL_OK",
    "PILL_WARN",
    "RESERVATION_REASON_BUCKET",
    "RESERVATION_REASON_VOCAB",
    "WITHIN_REASONS",
    "bucket_for",
    "pill_class_for_bucket",
]
