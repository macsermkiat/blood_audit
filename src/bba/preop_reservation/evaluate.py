"""Pure judgment over a reserved RBC quantity and resolved planned operation."""

from __future__ import annotations

from bba.preop_reservation.models import (
    MsbosRow,
    ReservationDecision,
    ReservationReason,
)
from bba.preop_reservation.reference import MsbosReference


def _resolved_decision(
    *,
    code: str,
    reserved_units: int,
    recommendation: MsbosRow,
    reference: MsbosReference,
    reason: ReservationReason,
    is_over: bool = False,
) -> ReservationDecision:
    return ReservationDecision(
        resolved_icd9=code,
        msbos=recommendation.msbos,
        recommended_units=recommendation.recommended_units,
        reserved_units=reserved_units,
        is_over=is_over,
        reason=reason,
        reference_hash=reference.content_hash,
    )


def evaluate_reservation(
    *,
    reserved_units: int,
    planned_icd9_nodot: str,
    reference: MsbosReference,
) -> ReservationDecision:
    """Evaluate the frozen MSBOS over-reservation rules table."""
    code = planned_icd9_nodot.strip()
    if not code:
        return ReservationDecision(
            reserved_units=reserved_units,
            reason="no_planned_op",
            reference_hash=reference.content_hash,
        )

    recommendation = reference.resolve(code)
    if recommendation is None:
        return ReservationDecision(
            resolved_icd9=code,
            reserved_units=reserved_units,
            reason="unresolved_code",
            reference_hash=reference.content_hash,
        )
    if recommendation == "ambiguous":
        return ReservationDecision(
            resolved_icd9=code,
            reserved_units=reserved_units,
            reason="ambiguous_code",
            reference_hash=reference.content_hash,
        )

    if recommendation.msbos == "T/S":
        # T2 (#164): a Type & Screen recommends screening only, no crossmatch.
        # The chosen crossmatch signal is the reserved-unit count (reserved_units
        # proxy): reserving any physical RBC unit is over-preparation, while zero
        # units is a compliant screen-only reservation. This makes the
        # crossmatch-vs-screen status always establishable, so it never asserts
        # over on absent unit data.
        if reserved_units > 0:
            return _resolved_decision(
                code=code,
                reserved_units=reserved_units,
                recommendation=recommendation,
                reference=reference,
                is_over=True,
                reason="over_type_and_screen_crossmatched",
            )
        return _resolved_decision(
            code=code,
            reserved_units=reserved_units,
            recommendation=recommendation,
            reference=reference,
            reason="type_and_screen_screen_only",
        )
    if recommendation.msbos == "none" and reserved_units > 0:
        return _resolved_decision(
            code=code,
            reserved_units=reserved_units,
            recommendation=recommendation,
            reference=reference,
            is_over=True,
            reason="over_none",
        )
    if (
        recommendation.msbos == "G/M"
        and reserved_units > recommendation.recommended_units
    ):
        return _resolved_decision(
            code=code,
            reserved_units=reserved_units,
            recommendation=recommendation,
            reference=reference,
            is_over=True,
            reason="over_gm_excess",
        )
    return _resolved_decision(
        code=code,
        reserved_units=reserved_units,
        recommendation=recommendation,
        reference=reference,
        reason="within_recommendation",
    )


__all__ = ["evaluate_reservation"]
