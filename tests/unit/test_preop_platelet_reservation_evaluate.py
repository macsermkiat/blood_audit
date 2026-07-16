"""Pure platelet reservation judgment tests for ticket #166."""

from __future__ import annotations

import pytest

from bba.preop_reservation import (
    MNS_HIGH_RISK_CEILING_PER_UL,
    MNS_THRESHOLD_PER_UL,
    REVIEW_REASONS,
    PlateletCategory,
    evaluate_platelet_reservation,
    platelet_reservation_verdict_for_category,
)


def test_review_reasons_is_exactly_the_eight_needs_review_reasons() -> None:
    # WHY: the review overlay fires iff decision.reason in REVIEW_REASONS.
    # Membership is load-bearing — accidentally adding a benign reason
    # (within_*, no_reserved_units) or an over_* reason here would route a
    # non-review verdict to NEEDS_REVIEW (or silently swallow an over row).
    # Pin the frozen set exactly so such a drift fails loudly.
    assert REVIEW_REASONS == frozenset(
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
    # The benign non-over and over reasons must NEVER be review reasons.
    for reason in (
        "within_major_non_neuraxial",
        "no_reserved_units",
        "over_major_non_neuraxial",
        "over_cardiac_cpb_any_units",
    ):
        assert reason not in REVIEW_REASONS


@pytest.mark.parametrize(
    ("count_k_ul", "expected"),
    [
        (79.999, (False, "within_major_non_neuraxial")),
        (80.0, (False, "gray_band_major_non_neuraxial")),
        (99.999, (False, "gray_band_major_non_neuraxial")),
        (100.0, (True, "over_major_non_neuraxial")),
        (150.0, (True, "over_major_non_neuraxial")),
    ],
)
def test_major_non_neuraxial_seam_uses_draft_boundaries(
    count_k_ul: float, expected: tuple[bool, str]
) -> None:
    assert (
        platelet_reservation_verdict_for_category(
            category=PlateletCategory.MAJOR_NON_NEURAXIAL,
            pre_op_count_k_ul=count_k_ul,
            reserved_units=1,
        )
        == expected
    )


@pytest.mark.parametrize(
    "category",
    [PlateletCategory.MAJOR_NON_NEURAXIAL, PlateletCategory.CARDIAC_CPB],
)
def test_zero_reserved_units_can_never_be_over(category: PlateletCategory) -> None:
    assert platelet_reservation_verdict_for_category(
        category=category,
        pre_op_count_k_ul=150.0,
        reserved_units=0,
    ) == (False, "no_reserved_units")


def test_resolved_card_rule_seam_treats_any_reserved_unit_as_over() -> None:
    # The CARDIAC_CPB category is still UNRESOLVED_ROUTE_REVIEW pending clinician
    # sign-off (worksheet Open B-i), so evaluate_platelet_reservation routes
    # cardiothoracic to NEEDS_REVIEW. This asserts only the pure count-independent
    # SEAM that becomes reachable once the cardiac/thoracic split is signed off;
    # it does NOT imply any clinician approval has occurred.
    assert platelet_reservation_verdict_for_category(
        category=PlateletCategory.CARDIAC_CPB,
        pre_op_count_k_ul=None,
        reserved_units=1,
    ) == (True, "over_cardiac_cpb_any_units")


@pytest.mark.parametrize(
    ("groups", "count", "expected_reason"),
    [
        (("ศัลยกรรมหัวใจและทรวงอก",), None, "cardiothoracic_split_unresolved"),
        (("ศัลยกรรมระบบประสาท",), 70.0, "neuraxial_rule_unresolved"),
        (("Tumor",), 70.0, "uncategorised_procedure"),
        (("ศัลยกรรมทั่วไป", "C Spine"), 70.0, "ambiguous_category"),
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
    assert decision.is_over is False
    assert decision.seed_pending_signoff is True


@pytest.mark.parametrize(
    ("groups", "count", "planned"),
    [
        # Every terminal-bearing shape, but with zero reserved units: an MNS
        # over-count, a category that would otherwise route to review, a missing
        # count (would-be missing_pre_op_count), and a blank plan.
        (("ศัลยกรรมทั่วไป",), 150.0, "0613"),
        (("ศัลยกรรมหัวใจและทรวงอก",), 70.0, "3220"),
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
    # be spuriously flagged. This guards the reserved-units<=0 short-circuit.
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


def test_reserved_but_uncounted_major_non_neuraxial_routes_to_review() -> None:
    # The never-guess missing-count case: platelets ARE reserved (2 units) but no
    # pre-op count exists, so the reservation cannot be judged numerically and
    # must reach clinician review rather than be silently absorbed.
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
        "threshold_per_ul": 80_000,
        "high_risk_ceiling_per_ul": 100_000,
        "reserved_units": 2,
        "is_over": True,
        "reason": "over_major_non_neuraxial",
        "reference_hash": "d" * 64,
        "seed_pending_signoff": True,
    }


def test_seed_thresholds_match_the_draft_values() -> None:
    assert MNS_THRESHOLD_PER_UL == 80_000
    assert MNS_HIGH_RISK_CEILING_PER_UL == 100_000
