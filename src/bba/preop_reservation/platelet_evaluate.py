"""Pure judgment over a reserved platelet quantity and planned operation.

Applies the clinician-signed reservation rule (worksheet T4/#166): for a
resolved procedure category, reserving platelets is APPROPRIATE when the pre-op
count is at/below the category cutoff and OVER when it is strictly above. There
is no gray band (signed Open question A-i). A missing count, an uncategorisable
or category-ambiguous code, or an unresolved plan routes to NEEDS_REVIEW; a
zero/absent reserved quantity is never a reservation to judge.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bba.preop_reservation.platelet_thresholds import (
    CATEGORY_OVER_ABOVE_PER_UL,
    PlateletCategory,
    category_for_groups,
)

PlateletReservationReason = Literal[
    "within_major_non_neuraxial",
    "within_neuraxial",
    "within_cardiac_cpb",
    "no_reserved_units",
    "over_major_non_neuraxial",
    "over_neuraxial",
    "over_cardiac_cpb",
    "uncategorised_procedure",
    "ambiguous_category",
    "missing_pre_op_count",
    "no_planned_op",
    "ambiguous_planned_op",
]

REVIEW_REASONS: frozenset[PlateletReservationReason] = frozenset(
    {
        "uncategorised_procedure",
        "ambiguous_category",
        "missing_pre_op_count",
        "no_planned_op",
        "ambiguous_planned_op",
    }
)

_WITHIN_REASON: dict[PlateletCategory, PlateletReservationReason] = {
    PlateletCategory.MAJOR_NON_NEURAXIAL: "within_major_non_neuraxial",
    PlateletCategory.NEURAXIAL: "within_neuraxial",
    PlateletCategory.CARDIAC_CPB: "within_cardiac_cpb",
}
_OVER_REASON: dict[PlateletCategory, PlateletReservationReason] = {
    PlateletCategory.MAJOR_NON_NEURAXIAL: "over_major_non_neuraxial",
    PlateletCategory.NEURAXIAL: "over_neuraxial",
    PlateletCategory.CARDIAC_CPB: "over_cardiac_cpb",
}


class PlateletReservationDecision(BaseModel):
    """Frozen per-order snapshot of the platelet reservation judgment."""

    model_config = ConfigDict(frozen=True)

    resolved_icd9: str = ""
    category: str = ""
    pre_op_count_k_ul: float | None = None
    over_above_per_ul: int | None = None
    reserved_units: int = Field(default=0, ge=0)
    is_over: bool = False
    reason: PlateletReservationReason
    reference_hash: str
    clinician_signed: bool = True


def platelet_reservation_verdict_for_category(
    *,
    category: PlateletCategory,
    pre_op_count_k_ul: float | None,
    reserved_units: int,
) -> tuple[bool, PlateletReservationReason]:
    """Apply the signed count cutoff for a resolved platelet category.

    Reserving is over when the pre-op count is strictly above the category
    cutoff, and appropriate at/below it (no gray band). A zero/absent reserved
    quantity can never be over.
    """
    if reserved_units <= 0:
        return False, "no_reserved_units"
    if pre_op_count_k_ul is None:
        raise ValueError("platelet reservation verdict requires a pre-op count")
    cutoff = CATEGORY_OVER_ABOVE_PER_UL[category]
    count_per_ul = pre_op_count_k_ul * 1000.0
    if count_per_ul > cutoff:
        return True, _OVER_REASON[category]
    return False, _WITHIN_REASON[category]


def evaluate_platelet_reservation(
    *,
    reserved_units: int,
    pre_op_count_k_ul: float | None,
    planned_icd9_nodot: str,
    procedure_groups: Sequence[str],
    reference_hash: str,
) -> PlateletReservationDecision:
    """Evaluate the clinician-signed platelet reservation rules."""
    code = planned_icd9_nodot.strip()

    def decision(
        *,
        reason: PlateletReservationReason,
        category: PlateletCategory | None = None,
        is_over: bool = False,
    ) -> PlateletReservationDecision:
        return PlateletReservationDecision(
            resolved_icd9=code,
            category=category.value if category is not None else "",
            pre_op_count_k_ul=pre_op_count_k_ul,
            over_above_per_ul=(
                CATEGORY_OVER_ABOVE_PER_UL[category] if category is not None else None
            ),
            reserved_units=reserved_units,
            is_over=is_over,
            reason=reason,
            reference_hash=reference_hash,
            clinician_signed=True,
        )

    if reserved_units <= 0:
        # No reserved platelet units means there is no reservation to judge — it
        # can be neither over nor a review case, whatever the count, plan, or
        # category. Short-circuits before every terminal branch below so a
        # zero-unit order (incl. a missing count) proceeds to the normal floor /
        # LLM path instead of a spurious NEEDS_REVIEW.
        return decision(reason="no_reserved_units")
    if code == "":
        return decision(reason="no_planned_op")
    if code == "\x00AMBIG":
        return decision(reason="ambiguous_planned_op")

    categories = category_for_groups(procedure_groups)
    if not categories:
        return decision(reason="uncategorised_procedure")
    if len(categories) > 1:
        return decision(reason="ambiguous_category")

    (category,) = categories
    if pre_op_count_k_ul is None:
        # Reserved-but-uncounted: the reservation cannot be judged numerically
        # and must reach clinician review rather than be silently absorbed.
        return decision(reason="missing_pre_op_count", category=category)

    is_over, reason = platelet_reservation_verdict_for_category(
        category=category,
        pre_op_count_k_ul=pre_op_count_k_ul,
        reserved_units=reserved_units,
    )
    return decision(reason=reason, category=category, is_over=is_over)


__all__ = [
    "REVIEW_REASONS",
    "PlateletReservationDecision",
    "PlateletReservationReason",
    "evaluate_platelet_reservation",
    "platelet_reservation_verdict_for_category",
]
