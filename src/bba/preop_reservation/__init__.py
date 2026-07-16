"""Pre-op blood-component reservation evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from bba.preop_reservation.evaluate import (
    evaluate_reservation,
    evaluate_reservation_with_notes,
)
from bba.preop_reservation.models import (
    CandidateOperation,
    MsbosRow,
    ReservationDecision,
)
from bba.preop_reservation.platelet_evaluate import (
    REVIEW_REASONS,
    PlateletReservationDecision,
    PlateletReservationReason,
    evaluate_platelet_reservation,
    platelet_reservation_verdict_for_category,
)
from bba.preop_reservation.platelet_thresholds import (
    CARDIAC_CPB_OVER_ABOVE_PER_UL,
    CATEGORY_OVER_ABOVE_PER_UL,
    MAJOR_NON_NEURAXIAL_OVER_ABOVE_PER_UL,
    NEURAXIAL_OVER_ABOVE_PER_UL,
    PROCEDURE_GROUP_TO_CATEGORY,
    PlateletCategory,
    category_for_groups,
)
from bba.preop_reservation.reference import (
    MsbosReference,
    MsbosReferenceError,
    load_msbos_reference,
)
from bba.preop_reservation.reserved_units import reserved_units_by_component

__all__: Sequence[str] = (
    "CandidateOperation",
    "MsbosReference",
    "MsbosReferenceError",
    "MsbosRow",
    "CARDIAC_CPB_OVER_ABOVE_PER_UL",
    "CATEGORY_OVER_ABOVE_PER_UL",
    "MAJOR_NON_NEURAXIAL_OVER_ABOVE_PER_UL",
    "NEURAXIAL_OVER_ABOVE_PER_UL",
    "PROCEDURE_GROUP_TO_CATEGORY",
    "PlateletCategory",
    "PlateletReservationDecision",
    "PlateletReservationReason",
    "REVIEW_REASONS",
    "ReservationDecision",
    "category_for_groups",
    "evaluate_platelet_reservation",
    "evaluate_reservation",
    "evaluate_reservation_with_notes",
    "load_msbos_reference",
    "platelet_reservation_verdict_for_category",
    "reserved_units_by_component",
)
