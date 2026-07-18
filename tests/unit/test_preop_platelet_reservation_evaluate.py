"""Pure platelet reservation judgment tests for ticket #166 (clinician-signed)."""

from __future__ import annotations

import pytest

from bba.preop_reservation import (
    CARDIAC_CPB_OVER_ABOVE_PER_UL,
    MAJOR_NON_NEURAXIAL_OVER_ABOVE_PER_UL,
    NEURAXIAL_OVER_ABOVE_PER_UL,
    REVIEW_REASONS,
    PlateletCategory,
    evaluate_platelet_reservation,
    platelet_reservation_verdict_for_category,
)


def test_review_reasons_is_exactly_the_five_needs_review_reasons() -> None:
    # WHY: the review overlay fires iff decision.reason in REVIEW_REASONS.
    # Membership is load-bearing — accidentally adding a benign (within_*,
    # no_reserved_units) or an over_* reason here would route a non-review verdict
    # to NEEDS_REVIEW (or silently swallow an over row). Pin the set exactly.
    assert REVIEW_REASONS == frozenset(
        {
            "uncategorised_procedure",
            "ambiguous_category",
            "missing_pre_op_count",
            "no_planned_op",
            "ambiguous_planned_op",
        }
    )
    for reason in (
        "within_major_non_neuraxial",
        "within_neuraxial",
        "within_cardiac_cpb",
        "no_reserved_units",
        "over_major_non_neuraxial",
        "over_neuraxial",
        "over_cardiac_cpb",
    ):
        assert reason not in REVIEW_REASONS


@pytest.mark.parametrize(
    ("category", "count_k_ul", "expected"),
    [
        # Signed binary rule (no gray band): count <= cutoff -> within;
        # count > cutoff -> over. MNS cutoff 80,000; NEURAX/CARD cutoff 100,000.
        (
            PlateletCategory.MAJOR_NON_NEURAXIAL,
            80.0,
            (False, "within_major_non_neuraxial"),
        ),
        (
            PlateletCategory.MAJOR_NON_NEURAXIAL,
            80.001,
            (True, "over_major_non_neuraxial"),
        ),
        (
            PlateletCategory.MAJOR_NON_NEURAXIAL,
            150.0,
            (True, "over_major_non_neuraxial"),
        ),
        (PlateletCategory.NEURAXIAL, 100.0, (False, "within_neuraxial")),
        (PlateletCategory.NEURAXIAL, 100.001, (True, "over_neuraxial")),
        (PlateletCategory.CARDIAC_CPB, 100.0, (False, "within_cardiac_cpb")),
        (PlateletCategory.CARDIAC_CPB, 120.0, (True, "over_cardiac_cpb")),
    ],
)
def test_seam_applies_signed_cutoff_per_category(
    category: PlateletCategory, count_k_ul: float, expected: tuple[bool, str]
) -> None:
    assert (
        platelet_reservation_verdict_for_category(
            category=category,
            pre_op_count_k_ul=count_k_ul,
            reserved_units=1,
        )
        == expected
    )


@pytest.mark.parametrize(
    "category",
    [
        PlateletCategory.MAJOR_NON_NEURAXIAL,
        PlateletCategory.NEURAXIAL,
        PlateletCategory.CARDIAC_CPB,
    ],
)
def test_zero_reserved_units_can_never_be_over(category: PlateletCategory) -> None:
    assert platelet_reservation_verdict_for_category(
        category=category,
        pre_op_count_k_ul=150.0,
        reserved_units=0,
    ) == (False, "no_reserved_units")


@pytest.mark.parametrize(
    ("groups", "count", "expected_reason"),
    [
        # Signed resolutions: cardiothoracic and neuraxial now carry a count
        # cutoff (100k), so they produce within/over rather than a review reason.
        (("ศัลยกรรมหัวใจและทรวงอก",), 120.0, "over_cardiac_cpb"),
        (("ศัลยกรรมหัวใจและทรวงอก",), 90.0, "within_cardiac_cpb"),
        (("ศัลยกรรมระบบประสาท",), 120.0, "over_neuraxial"),
        (("ศัลยกรรมระบบประสาท",), 90.0, "within_neuraxial"),
        # Tumor is now signed to MNS (cutoff 80k).
        (("Tumor",), 90.0, "over_major_non_neuraxial"),
        (("ศัลยกรรมทั่วไป",), 70.0, "within_major_non_neuraxial"),
    ],
)
def test_evaluator_applies_signed_category_verdicts(
    groups: tuple[str, ...], count: float, expected_reason: str
) -> None:
    decision = evaluate_platelet_reservation(
        reserved_units=2,
        pre_op_count_k_ul=count,
        planned_icd9_nodot="0124",
        procedure_groups=groups,
        reference_hash="a" * 64,
    )

    assert decision.reason == expected_reason
    assert decision.clinician_signed is True


@pytest.mark.parametrize(
    ("groups", "count", "expected_reason"),
    [
        # A code whose groups span DISTINCT categories -> never guess.
        (("ศัลยกรรมทั่วไป", "C Spine"), 70.0, "ambiguous_category"),
        # A resolved category with no usable pre-op count -> review.
        (("ศัลยกรรมทั่วไป",), None, "missing_pre_op_count"),
    ],
)
def test_evaluator_routes_clinical_ambiguity_to_typed_review(
    groups: tuple[str, ...], count: float | None, expected_reason: str
) -> None:
    decision = evaluate_platelet_reservation(
        reserved_units=2,
        pre_op_count_k_ul=count,
        planned_icd9_nodot="0124",
        procedure_groups=groups,
        reference_hash="a" * 64,
    )

    assert decision.reason == expected_reason
    assert decision.reason in REVIEW_REASONS
    assert decision.is_over is False


def test_absent_code_with_no_known_group_routes_to_uncategorised() -> None:
    decision = evaluate_platelet_reservation(
        reserved_units=2,
        pre_op_count_k_ul=70.0,
        planned_icd9_nodot="9999",
        procedure_groups=(),
        reference_hash="a" * 64,
    )
    assert decision.reason == "uncategorised_procedure"
    assert decision.reason in REVIEW_REASONS


@pytest.mark.parametrize(
    ("groups", "count", "planned"),
    [
        (("ศัลยกรรมทั่วไป",), 150.0, "0613"),
        (("ศัลยกรรมหัวใจและทรวงอก",), 120.0, "3220"),
        (("ศัลยกรรมทั่วไป",), None, "0613"),
        (("ศัลยกรรมทั่วไป",), 70.0, "   "),
    ],
)
def test_zero_reserved_units_evaluate_is_never_terminal(
    groups: tuple[str, ...], count: float | None, planned: str
) -> None:
    # WHY: with no platelet units reserved there is no reservation to judge, so
    # the order must proceed to the normal floor/LLM path (never over, never
    # review) — otherwise a bare platelet order with no BDVSTDT detail line would
    # be spuriously flagged. Guards the reserved-units<=0 short-circuit.
    decision = evaluate_platelet_reservation(
        reserved_units=0,
        pre_op_count_k_ul=count,
        planned_icd9_nodot=planned,
        procedure_groups=groups,
        reference_hash="a" * 64,
    )

    assert decision.reason == "no_reserved_units"
    assert decision.is_over is False
    assert decision.reason not in REVIEW_REASONS


def test_reserved_but_uncounted_routes_to_review() -> None:
    # The never-guess missing-count case: platelets ARE reserved but no pre-op
    # count exists, so the reservation cannot be judged and must reach review.
    decision = evaluate_platelet_reservation(
        reserved_units=2,
        pre_op_count_k_ul=None,
        planned_icd9_nodot="0613",
        procedure_groups=("ศัลยกรรมทั่วไป",),
        reference_hash="a" * 64,
    )

    assert decision.reason == "missing_pre_op_count"
    assert decision.reason in REVIEW_REASONS
    assert decision.is_over is False


@pytest.mark.parametrize(
    ("planned", "expected_reason"),
    [("   ", "no_planned_op"), ("\x00AMBIG", "ambiguous_planned_op")],
)
def test_evaluator_defensively_routes_missing_or_ambiguous_plan_to_review(
    planned: str, expected_reason: str
) -> None:
    decision = evaluate_platelet_reservation(
        reserved_units=1,
        pre_op_count_k_ul=120.0,
        planned_icd9_nodot=planned,
        procedure_groups=("ศัลยกรรมทั่วไป",),
        reference_hash="b" * 64,
    )

    assert decision.reason == expected_reason
    assert decision.resolved_icd9 == planned.strip()


def test_evaluator_is_deterministic_across_group_order_and_duplicates() -> None:
    kwargs = {
        "reserved_units": 1,
        "pre_op_count_k_ul": 120.0,
        "planned_icd9_nodot": "0124",
        "reference_hash": "c" * 64,
    }

    first = evaluate_platelet_reservation(
        **kwargs, procedure_groups=("C Spine", "ศัลยกรรมทั่วไป", "C Spine")
    )
    second = evaluate_platelet_reservation(
        **kwargs, procedure_groups=("ศัลยกรรมทั่วไป", "C Spine")
    )

    assert first == second
    assert first.reason == "ambiguous_category"


def test_evaluator_stamps_resolved_major_non_neuraxial_snapshot() -> None:
    decision = evaluate_platelet_reservation(
        reserved_units=2,
        pre_op_count_k_ul=120.0,
        planned_icd9_nodot="0613",
        procedure_groups=("ศัลยกรรมทั่วไป",),
        reference_hash="d" * 64,
    )

    assert decision.model_dump() == {
        "resolved_icd9": "0613",
        "category": "major_non_neuraxial",
        "pre_op_count_k_ul": 120.0,
        "over_above_per_ul": 80_000,
        "reserved_units": 2,
        "is_over": True,
        "reason": "over_major_non_neuraxial",
        "reference_hash": "d" * 64,
        "clinician_signed": True,
        "planned_op": None,
    }


def test_signed_cutoffs_match_the_worksheet_values() -> None:
    assert MAJOR_NON_NEURAXIAL_OVER_ABOVE_PER_UL == 80_000
    assert NEURAXIAL_OVER_ABOVE_PER_UL == 100_000
    assert CARDIAC_CPB_OVER_ABOVE_PER_UL == 100_000
