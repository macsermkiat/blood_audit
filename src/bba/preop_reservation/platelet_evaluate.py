"""Pure judgment over a reserved platelet quantity and planned operation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bba.preop_reservation.platelet_thresholds import (
    CATEGORY_SEED_STATUS,
    MNS_HIGH_RISK_CEILING_PER_UL,
    MNS_THRESHOLD_PER_UL,
    PlateletCategory,
    SeedStatus,
    category_for_groups,
)

PlateletReservationReason = Literal[
    "within_major_non_neuraxial",
    "no_reserved_units",
    "over_major_non_neuraxial",
    "over_cardiac_cpb_any_units",
    "gray_band_major_non_neuraxial",
    "cardiothoracic_split_unresolved",
    "neuraxial_rule_unresolved",
    "uncategorised_procedure",
    "ambiguous_category",
    "missing_pre_op_count",
    "no_planned_op",
    "ambiguous_planned_op",
]

REVIEW_REASONS: frozenset[PlateletReservationReason] = frozenset(
    {
        "gray_band_major_non_neuraxial",
        "cardiothoracic_split_unresolved",
        "neuraxial_rule_unresolved",
        "uncategorised_procedure",
        "ambiguous_category",
        "missing_pre_op_count",
        "no_planned_op",
        "ambiguous_planned_op",
    }
)


class PlateletReservationDecision(BaseModel):
    """Frozen per-order snapshot of the platelet reservation judgment."""

    model_config = ConfigDict(frozen=True)

    resolved_icd9: str = ""
    category: str = ""
    pre_op_count_k_ul: float | None = None
    threshold_per_ul: int | None = None
    high_risk_ceiling_per_ul: int | None = None
    reserved_units: int = Field(default=0, ge=0)
    is_over: bool = False
    reason: PlateletReservationReason
    reference_hash: str
    seed_pending_signoff: bool = True


def platelet_reservation_verdict_for_category(
    *,
    category: PlateletCategory,
    pre_op_count_k_ul: float | None,
    reserved_units: int,
) -> tuple[bool, PlateletReservationReason]:
    """Apply the count and unit rule for a resolved platelet category."""
    if reserved_units <= 0:
        return False, "no_reserved_units"
    if category is PlateletCategory.CARDIAC_CPB:
        return True, "over_cardiac_cpb_any_units"
    if category is PlateletCategory.MAJOR_NON_NEURAXIAL:
        if pre_op_count_k_ul is None:
            raise ValueError("major non-neuraxial verdict requires a pre-op count")
        count_per_ul = pre_op_count_k_ul * 1000.0
        if count_per_ul < MNS_THRESHOLD_PER_UL:
            return False, "within_major_non_neuraxial"
        if count_per_ul >= MNS_HIGH_RISK_CEILING_PER_UL:
            return True, "over_major_non_neuraxial"
        return False, "gray_band_major_non_neuraxial"
    raise ValueError(f"unresolved platelet category {category.value!r}")


def evaluate_platelet_reservation(
    *,
    reserved_units: int,
    pre_op_count_k_ul: float | None,
    planned_icd9_nodot: str,
    procedure_groups: Sequence[str],
    reference_hash: str,
) -> PlateletReservationDecision:
    """Evaluate the frozen platelet reservation SEED rules."""
    code = planned_icd9_nodot.strip()

    def decision(
        *,
        reason: PlateletReservationReason,
        category: PlateletCategory | None = None,
        is_over: bool = False,
        stamp_mns_thresholds: bool = False,
    ) -> PlateletReservationDecision:
        return PlateletReservationDecision(
            resolved_icd9=code,
            category=category.value if category is not None else "",
            pre_op_count_k_ul=pre_op_count_k_ul,
            threshold_per_ul=MNS_THRESHOLD_PER_UL if stamp_mns_thresholds else None,
            high_risk_ceiling_per_ul=(
                MNS_HIGH_RISK_CEILING_PER_UL if stamp_mns_thresholds else None
            ),
            reserved_units=reserved_units,
            is_over=is_over,
            reason=reason,
            reference_hash=reference_hash,
            seed_pending_signoff=True,
        )

    if reserved_units <= 0:
        # No reserved platelet units means there is no reservation to judge — it
        # can be neither over nor a review case, whatever the count, plan, or
        # category. Short-circuits before every terminal branch below so a
        # zero-unit order (incl. a missing count) proceeds to the normal floor /
        # LLM path instead of a spurious NEEDS_REVIEW.
        return decision(reason="no_reserved_units")
    if code == "":
        # TODO(worksheet T4/#166): a reservation without one planned operation requires review.
        return decision(reason="no_planned_op")
    if code == "\x00AMBIG":
        # TODO(worksheet T4/#166): equally near planned operations require review.
        return decision(reason="ambiguous_planned_op")

    categories = category_for_groups(procedure_groups)
    if not categories:
        # TODO(worksheet T4/#166): an unknown procedure group has no signed category.
        return decision(reason="uncategorised_procedure")
    if len(categories) > 1:
        # TODO(worksheet T4/#166): codes spanning categories must never guess one.
        return decision(reason="ambiguous_category")

    (category,) = categories
    if category is PlateletCategory.UNCATEGORISED:
        # TODO(worksheet T4/#166): categorise Tumor/TR/Pediatric before numeric use.
        return decision(reason="uncategorised_procedure", category=category)
    if category is PlateletCategory.NEURAXIAL:
        # TODO(worksheet T4/#166): the DRAFT supplies no neuraxial rule.
        return decision(reason="neuraxial_rule_unresolved", category=category)
    if (
        category is PlateletCategory.CARDIAC_CPB
        and CATEGORY_SEED_STATUS[category] is SeedStatus.UNRESOLVED_ROUTE_REVIEW
    ):
        # TODO(worksheet T4/#166): split merged cardiac/thoracic reference rows.
        return decision(reason="cardiothoracic_split_unresolved", category=category)
    if category is PlateletCategory.MAJOR_NON_NEURAXIAL and pre_op_count_k_ul is None:
        # TODO(worksheet T4/#166): reservation appropriateness needs a pre-op count.
        return decision(
            reason="missing_pre_op_count",
            category=category,
            stamp_mns_thresholds=True,
        )

    is_over, reason = platelet_reservation_verdict_for_category(
        category=category,
        pre_op_count_k_ul=pre_op_count_k_ul,
        reserved_units=reserved_units,
    )
    return decision(
        reason=reason,
        category=category,
        is_over=is_over,
        stamp_mns_thresholds=category is PlateletCategory.MAJOR_NON_NEURAXIAL,
    )


__all__ = [
    "REVIEW_REASONS",
    "PlateletReservationDecision",
    "PlateletReservationReason",
    "evaluate_platelet_reservation",
    "platelet_reservation_verdict_for_category",
]
