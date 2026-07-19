"""Pin the derived (not persisted) MSBOS reservation-eligibility classifier.

WHY: reserve-ahead / declared-exempt routing hinges on the free-form ``rationale``
slug. These tests pin the typed ``reservation_eligibility`` seam and prove
``is_msbos_eligible`` is byte-for-byte the historical predicate, so a slug rename
(now a single edit in ``deterministic_classifier.rationales``) cannot silently
change which rows enter MSBOS screening.
"""

from __future__ import annotations

import pytest

from bba.audit_pipeline.replay import (
    ReservationEligibility,
    is_msbos_eligible,
    reservation_eligibility,
)
from bba.deterministic_classifier.models import BypassReason, ClassifierResult
from bba.deterministic_classifier.rationales import (
    PREOP_DECLARED_EXEMPT,
    PREOP_DEFER_LLM,
    PREOP_DEFER_LLM_DECLARED,
    RESERVE_AHEAD_RATIONALES,
)


def _result(classification: str, rationale: str) -> ClassifierResult:
    return ClassifierResult(
        classification=classification,  # type: ignore[arg-type]
        bypass_reason=BypassReason.NONE,
        cohort_threshold=None,
        rationale=rationale,
    )


def test_reserve_ahead_set_is_the_two_deferral_slugs() -> None:
    assert RESERVE_AHEAD_RATIONALES == {PREOP_DEFER_LLM, PREOP_DEFER_LLM_DECLARED}
    assert PREOP_DECLARED_EXEMPT == "preop_declared_exempt"
    assert PREOP_DEFER_LLM == "preop_defer_llm"
    assert PREOP_DEFER_LLM_DECLARED == "preop_defer_llm_declared"


@pytest.mark.parametrize(
    ("classification", "rationale", "expected"),
    [
        # declared pre-op exemption -> declared-exempt, still screen-eligible
        (
            "PERIOP_TRANSFUSION_EXEMPT",
            PREOP_DECLARED_EXEMPT,
            ReservationEligibility.DECLARED_EXEMPT,
        ),
        # legacy peri-op exemption (any other rationale) -> terminal
        (
            "PERIOP_TRANSFUSION_EXEMPT",
            "some_other_rationale",
            ReservationEligibility.NONE,
        ),
        # factual returns -> terminal ahead of MSBOS
        ("RETURNED_NOT_TRANSFUSED", PREOP_DEFER_LLM, ReservationEligibility.NONE),
        # reserve-ahead deferrals -> reserve-ahead, screen-eligible
        ("NEEDS_REVIEW", PREOP_DEFER_LLM, ReservationEligibility.RESERVE_AHEAD),
        (
            "NEEDS_REVIEW",
            PREOP_DEFER_LLM_DECLARED,
            ReservationEligibility.RESERVE_AHEAD,
        ),
        # non-reserve NEEDS_REVIEW -> not eligible
        ("NEEDS_REVIEW", "hb_7_to_10", ReservationEligibility.NONE),
        # plain verdicts -> not eligible
        ("APPROPRIATE", "hb_lt_7_universal", ReservationEligibility.NONE),
    ],
)
def test_reservation_eligibility_matrix(
    classification: str, rationale: str, expected: ReservationEligibility
) -> None:
    result = _result(classification, rationale)
    assert reservation_eligibility(result) is expected
    # is_msbos_eligible is exactly "not NONE" — the historical predicate.
    assert is_msbos_eligible(result) is (expected is not ReservationEligibility.NONE)


def test_is_msbos_eligible_matches_legacy_predicate_for_every_classification() -> None:
    # Reproduces the pre-refactor branch logic verbatim across the classification
    # space, so any behaviour drift in the enum function fails here.
    classifications = [
        "APPROPRIATE",
        "NEEDS_REVIEW",
        "INSUFFICIENT_EVIDENCE",
        "POTENTIALLY_INAPPROPRIATE",
        "PREOP_OVER_RESERVATION",
        "RETURNED_NOT_TRANSFUSED",
        "PERIOP_TRANSFUSION_EXEMPT",
    ]
    rationales = [
        PREOP_DECLARED_EXEMPT,
        PREOP_DEFER_LLM,
        PREOP_DEFER_LLM_DECLARED,
        "hb_lt_7_universal",
        "bypass_mtp",
    ]
    for classification in classifications:
        for rationale in rationales:
            result = _result(classification, rationale)
            if classification == "PERIOP_TRANSFUSION_EXEMPT":
                expected = rationale == PREOP_DECLARED_EXEMPT
            elif classification in (
                "RETURNED_NOT_TRANSFUSED",
                "PERIOP_TRANSFUSION_EXEMPT",
            ):
                expected = False
            else:
                expected = rationale in RESERVE_AHEAD_RATIONALES
            assert is_msbos_eligible(result) is expected, (classification, rationale)
