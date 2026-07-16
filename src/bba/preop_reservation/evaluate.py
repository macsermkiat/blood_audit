"""Pure judgment over a reserved RBC quantity and resolved planned operation."""

from __future__ import annotations

from collections.abc import Sequence

from bba.preop_reservation.models import (
    MsbosRow,
    ReservationDecision,
    ReservationReason,
)
from bba.preop_reservation.note_operation import resolve_operation_from_notes
from bba.preop_reservation.reference import MsbosReference


def _decide_from_row(
    *,
    row: MsbosRow,
    reserved_units: int,
    resolved_icd9: str,
    reference_hash: str,
    note_resolved: bool,
) -> ReservationDecision:
    def decision(
        *,
        reason: ReservationReason,
        is_over: bool = False,
        recommended_units: int = row.recommended_units,
    ) -> ReservationDecision:
        return ReservationDecision(
            resolved_icd9=resolved_icd9,
            msbos=row.msbos,
            recommended_units=recommended_units,
            reserved_units=reserved_units,
            is_over=is_over,
            reason=reason,
            reference_hash=reference_hash,
            note_resolved=note_resolved,
        )

    if row.msbos == "T/S":
        # T2 (#164): a Type & Screen recommends screening only, no crossmatch.
        # The chosen crossmatch signal is the reserved-unit count (reserved_units
        # proxy): reserving any physical RBC unit is over-preparation, while zero
        # units is a compliant screen-only reservation. This makes the
        # crossmatch-vs-screen status always establishable, so it never asserts
        # over on absent unit data.
        #
        # Committee ruling (T2 wrinkle resolved): keep the strict >0 rule and
        # IGNORE any recommended_units the reference carries for a T/S item (some
        # rows list "1"/"2"/"1-2"). It is never a crossmatch ceiling — T/S means
        # zero units should be crossmatched — so the snapshot records
        # recommended_units=0, never the reference figure.
        if reserved_units > 0:
            return decision(
                reason="over_type_and_screen_crossmatched",
                is_over=True,
                recommended_units=0,
            )
        return decision(reason="type_and_screen_screen_only", recommended_units=0)
    if row.msbos == "none" and reserved_units > 0:
        return decision(reason="over_none", is_over=True)
    if row.msbos == "G/M" and reserved_units > row.recommended_units:
        return decision(reason="over_gm_excess", is_over=True)
    return decision(reason="within_recommendation")


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

    return _decide_from_row(
        row=recommendation,
        reserved_units=reserved_units,
        resolved_icd9=code,
        reference_hash=reference.content_hash,
        note_resolved=False,
    )


def evaluate_reservation_with_notes(
    *,
    reserved_units: int,
    planned_icd9_nodot: str,
    reference: MsbosReference,
    note_texts: Sequence[str],
) -> ReservationDecision:
    """As evaluate_reservation, but disambiguate an ambiguous code via notes."""
    decision = evaluate_reservation(
        reserved_units=reserved_units,
        planned_icd9_nodot=planned_icd9_nodot,
        reference=reference,
    )
    if decision.reason != "ambiguous_code":
        return decision

    code = planned_icd9_nodot.strip()
    row = resolve_operation_from_notes(
        candidates=reference.candidates_for(code), note_texts=note_texts
    )
    if row is None:
        return ReservationDecision(
            resolved_icd9=code,
            reserved_units=reserved_units,
            is_over=False,
            reason="operation_unresolved",
            reference_hash=reference.content_hash,
        )
    return _decide_from_row(
        row=row,
        reserved_units=reserved_units,
        resolved_icd9=code,
        reference_hash=reference.content_hash,
        note_resolved=True,
    )


__all__ = ["evaluate_reservation", "evaluate_reservation_with_notes"]
