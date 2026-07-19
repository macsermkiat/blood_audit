"""Pin the single-home MSBOS reservation reason -> bucket table.

These tests encode WHY the table matters: build_review groups reasons into count
buckets and pill colours, and a reason added to a library Literal must not
silently misbucket to "unresolved". The exhaustiveness test makes such a drift a
loud failure instead of a wrong bucket.
"""

from __future__ import annotations

from typing import get_args

from bba.preop_reservation.models import ReservationReason
from bba.preop_reservation.platelet_evaluate import PlateletReservationReason
from bba.preop_reservation.reason_presentation import (
    ABOVE_REASONS,
    PILL_OK,
    PILL_WARN,
    RESERVATION_REASON_BUCKET,
    RESERVATION_REASON_VOCAB,
    WITHIN_REASONS,
    bucket_for,
    pill_class_for_bucket,
)


def test_vocab_is_both_literals_plus_leg_local_reason() -> None:
    expected = (
        set(get_args(ReservationReason))
        | set(get_args(PlateletReservationReason))
        | {"reservation_lookup_miss"}
    )
    assert set(RESERVATION_REASON_VOCAB) == expected


def test_bucket_table_is_exhaustive_over_vocab() -> None:
    # A new Literal member with no bucket entry fails here (and at import).
    assert set(RESERVATION_REASON_BUCKET) == set(RESERVATION_REASON_VOCAB)


def test_tricky_bucket_assignments_match_build_review_history() -> None:
    assert bucket_for("within_ceiling") == "within_ceiling"
    assert bucket_for("over_ceiling") == "above"
    assert bucket_for("no_reserved_units") == "within"
    assert bucket_for("reservation_lookup_miss") == "unresolved"


def test_unknown_or_blank_reason_falls_to_unresolved() -> None:
    assert bucket_for("totally_unknown_reason") == "unresolved"
    assert bucket_for("") == "unresolved"


def test_above_reasons_are_the_seven_overs() -> None:
    assert ABOVE_REASONS == {
        "over_none",
        "over_gm_excess",
        "over_type_and_screen_crossmatched",
        "over_ceiling",
        "over_major_non_neuraxial",
        "over_neuraxial",
        "over_cardiac_cpb",
    }


def test_within_reasons_exclude_within_ceiling_by_design() -> None:
    # within_ceiling is its OWN bucket; the derived within set must not contain it,
    # which is byte-parity-safe because build_review intercepts within_ceiling first.
    assert "within_ceiling" not in WITHIN_REASONS
    assert WITHIN_REASONS == {
        "within_recommendation",
        "type_and_screen_screen_only",
        "within_major_non_neuraxial",
        "within_neuraxial",
        "within_cardiac_cpb",
        "no_reserved_units",
    }


def test_pill_class_is_a_pure_function_of_bucket() -> None:
    assert pill_class_for_bucket("within") == PILL_OK
    assert pill_class_for_bucket("within_ceiling") == PILL_OK
    assert pill_class_for_bucket("above") == PILL_WARN
    assert pill_class_for_bucket("unresolved") == PILL_WARN
