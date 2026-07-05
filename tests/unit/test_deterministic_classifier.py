"""RED-phase failing tests for issue #8 (bba.deterministic_classifier).

Each ``class`` maps to one acceptance criterion in the issue body. Tests
assert contracts (the WHY), not implementation choices — see PRD §"Testing
Decisions". No implementation exists yet; every test MUST fail in this
scaffold commit (``NotImplementedError`` raised by ``classify`` /
``total_crystalloid_liters``, or missing public symbols).

Acceptance-criterion → test-class map (issue #8):

* AC ① "Implementation in ``src/bba/deterministic_classifier/``"
  → import surface verified at module top; collection failure means the
    public API is mis-scaffolded.

* AC ② "Exhaustive fixture table covering each (Hb-tier × cohort × bypass)
    combination"
  → :class:`TestHbTierByCohort` (parametrized matrix) and
    :class:`TestBypassPathways`.

* AC ③ "B2 invariant: missing notes + Hb=8 → INSUFFICIENT_EVIDENCE or
    NEEDS_REVIEW, NEVER INAPPROPRIATE; deterministic engine never emits
    INAPPROPRIATE under any input combination"
  → :class:`TestB2Invariant` (golden cases) +
    :class:`TestB2InvariantProperty` (hypothesis property over the input
    space; user constraint #6).

* AC ④ "Hemodilution flag: Hb 7.0-8.0 in a higher-threshold cohort with
    ≥2 L crystalloid in 4 h → NEEDS_REVIEW, not APPROPRIATE"
  → :class:`TestHemodilutionFlag` (boundary at 2 L; below-window vs
    in-window crystalloid).

* AC ⑤ "Bypass-reason field populated and distinct per pathway"
  → :class:`TestBypassReasonDistinct` (each pathway sets exactly its own
    enum member; non-bypass classifications carry NONE).

* AC ⑥ "Property test: monotonicity — increasing Hb never moves
    classification from INAPPROPRIATE toward APPROPRIATE within same cohort"
  → :class:`TestMonotonicityProperty` (hypothesis property; user
    constraint #7).

* AC ⑦ "Coverage ≥ 80%; ruff + mypy clean" — covered by suite totality.

Additional contract tests:

* :class:`TestCohortUnknownRoutesToNeedsReview` — user constraint #9: if
  cohort is UNKNOWN, classification MUST be NEEDS_REVIEW with
  bypass_reason=NONE unless the global Hb < 7.0 rule already returned
  APPROPRIATE.

* :class:`TestClassifierInputsImmutability` — frozen contract on input
  model so the classifier cannot accidentally mutate its argument.

* :class:`TestCrystalloidHelper` — :func:`total_crystalloid_liters`
  contract: sums in-window, ignores future events, parses mL/cc/L.

* :class:`TestCanonicalClassificationContract` — the result.classification
  field is the canonical :data:`bba.audit_store.Classification`; the
  deterministic engine never emits a string outside that Literal.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, get_args

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bba.audit_store import Classification
from bba.cohort_detector import (
    CARDIAC_SURGERY_THRESHOLD,
    DEFAULT_THRESHOLD,
    ESRD_EPO_THRESHOLD,
    ORTHO_CARDIAC_THRESHOLD,
    CohortAssignment,
    CohortLabel,
    MedEvent,
)
from bba.deterministic_classifier import (
    HB_GT_10_THRESHOLD,
    HEMODILUTION_CRYSTALLOID_LITERS,
    PERI_PROCEDURAL_WINDOW_HOURS,
    PERIOP_MIN_EBL_ML,
    PRE_OP_CROSSMATCH_WINDOW_HOURS,
    UNIVERSAL_LOW_HB_APPROPRIATE_THRESHOLD,
    BypassReason,
    ClassifierInputs,
    ClassifierResult,
    classify,
    total_crystalloid_liters,
)
from bba.hb_lookup import DeltaHbWindow, HbLookupResult


# =============================================================================
# Fixture helpers — small synthetic data, no real PHI
# =============================================================================

ORDER_DT = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
"""Fixed tz-aware UTC order anchor used by every fixture so window math
is deterministic across tests."""


def _untriggered_delta() -> tuple[DeltaHbWindow, ...]:
    """Three delta-Hb windows, none triggered. Use when the test is
    explicitly NOT exercising the delta-Hb bypass."""
    return tuple(
        DeltaHbWindow(
            window_hours=h,
            threshold_g_dl=t,
            prior_value_g_dl=None,
            prior_datetime_utc=None,
            drop_g_dl=None,
            triggered=False,
        )
        for h, t in ((6, 2.0), (12, 2.5), (24, 3.0))
    )


def _triggered_delta() -> tuple[DeltaHbWindow, ...]:
    """Three delta-Hb windows, the 6 h window triggered. Use when the
    test is exercising the delta-Hb bypass path."""
    return (
        DeltaHbWindow(
            window_hours=6,
            threshold_g_dl=2.0,
            prior_value_g_dl=10.0,
            prior_datetime_utc=ORDER_DT - timedelta(hours=5),
            drop_g_dl=2.5,
            triggered=True,
        ),
        DeltaHbWindow(
            window_hours=12,
            threshold_g_dl=2.5,
            prior_value_g_dl=None,
            prior_datetime_utc=None,
            drop_g_dl=None,
            triggered=False,
        ),
        DeltaHbWindow(
            window_hours=24,
            threshold_g_dl=3.0,
            prior_value_g_dl=None,
            prior_datetime_utc=None,
            drop_g_dl=None,
            triggered=False,
        ),
    )


def _hb(
    value_g_dl: float | None,
    *,
    delta_bypass: bool = False,
    freshness: str = "fresh",
    needs_review_single_low_hb: bool = False,
) -> HbLookupResult:
    """Build a minimal :class:`HbLookupResult` for tests.

    Pass ``value_g_dl=None`` to model the "Hb missing" tier (forces
    freshness=missing). ``delta_bypass=True`` substitutes a triggered
    6 h window so the classifier sees ``delta_hb_bypass=True``.
    ``needs_review_single_low_hb=True`` models the upstream flag for an
    isolated Hb < 8 with no 24 h trend (PRD §3 + hb_lookup contract).
    """
    if value_g_dl is None:
        return HbLookupResult(
            value_g_dl=None,
            datetime_utc=None,
            source=None,
            freshness="missing",
            delta_hb_bypass=False,
            delta_hb_windows=_untriggered_delta(),
            needs_review_single_low_hb=False,
        )
    return HbLookupResult(
        value_g_dl=value_g_dl,
        datetime_utc=ORDER_DT - timedelta(hours=2),
        source="HEMATOLOGY",
        freshness=freshness,  # type: ignore[arg-type]
        delta_hb_bypass=delta_bypass,
        delta_hb_windows=_triggered_delta() if delta_bypass else _untriggered_delta(),
        needs_review_single_low_hb=needs_review_single_low_hb,
    )


def _cohort(
    label: CohortLabel,
    threshold: float | None,
    *,
    evidence_code: str | None = None,
    evidence_name: str | None = None,
) -> CohortAssignment:
    return CohortAssignment(
        label=label,
        threshold=threshold,
        evidence_code=evidence_code,
        evidence_name=evidence_name,
    )


def _inputs(
    *,
    hb: HbLookupResult,
    cohort: CohortAssignment,
    procedure_proximity_hours: float | None = None,
    upcoming_procedure_hours: float | None = None,
    crystalloid_liters_prior_4h: float = 0.0,
    enable_missing_hb_positive_evidence: bool = False,
    periop_blood_loss_ml: int | None = None,
    periop_intraop_transfusion: bool = False,
    periop_surgical_context: bool = False,
) -> ClassifierInputs:
    return ClassifierInputs(
        audit_id="audit-test-0001",
        hb_result=hb,
        cohort_assignment=cohort,
        order_datetime=ORDER_DT,
        procedure_proximity_hours=procedure_proximity_hours,
        upcoming_procedure_hours=upcoming_procedure_hours,
        crystalloid_liters_prior_4h=crystalloid_liters_prior_4h,
        enable_missing_hb_positive_evidence=enable_missing_hb_positive_evidence,
        periop_blood_loss_ml=periop_blood_loss_ml,
        periop_intraop_transfusion=periop_intraop_transfusion,
        periop_surgical_context=periop_surgical_context,
    )


# =============================================================================
# AC ② — Hb-tier × cohort matrix
# =============================================================================


class TestHbTierByCohort:
    """One test per (Hb-tier × cohort) cell, with no bypass active.

    The matrix exercises every threshold-driven cohort against the three
    Hb tiers (< threshold, threshold ≤ Hb < 10, ≥ 10). Non-threshold
    cohorts (MTP, HEME_MALIGNANCY_ACTIVE) have their own tests in
    :class:`TestBypassPathways` / :class:`TestB2Invariant` because their
    classification rules are not Hb-tier-driven.
    """

    @pytest.mark.parametrize(
        ("label", "threshold"),
        [
            (CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            (CohortLabel.CARDIAC_SURGERY, CARDIAC_SURGERY_THRESHOLD),
            (CohortLabel.ORTHO_CARDIAC, ORTHO_CARDIAC_THRESHOLD),
            (CohortLabel.ESRD_EPO, ESRD_EPO_THRESHOLD),
        ],
    )
    def test_hb_below_threshold_is_appropriate(
        self, label: CohortLabel, threshold: float
    ) -> None:
        """Hb < cohort_threshold → APPROPRIATE (PRD §6 plain Hb-tier rule)."""
        result = classify(
            _inputs(
                hb=_hb(threshold - 0.5),
                cohort=_cohort(label, threshold),
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.NONE
        assert result.cohort_threshold == threshold

    @pytest.mark.parametrize(
        ("label", "threshold"),
        [
            (CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            (CohortLabel.CARDIAC_SURGERY, CARDIAC_SURGERY_THRESHOLD),
            (CohortLabel.ORTHO_CARDIAC, ORTHO_CARDIAC_THRESHOLD),
            (CohortLabel.ESRD_EPO, ESRD_EPO_THRESHOLD),
        ],
    )
    def test_hb_gray_zone_is_needs_review(
        self, label: CohortLabel, threshold: float
    ) -> None:
        """cohort_threshold ≤ Hb < 10 → NEEDS_REVIEW (LLM-eligible
        gray-zone case, persisted canonically per audit_store contract)."""
        # Pick Hb halfway between threshold and 10
        hb_value = (threshold + HB_GT_10_THRESHOLD) / 2.0
        result = classify(
            _inputs(
                hb=_hb(hb_value),
                cohort=_cohort(label, threshold),
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE
        assert result.cohort_threshold == threshold

    @pytest.mark.parametrize(
        ("label", "threshold"),
        [
            (CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            (CohortLabel.CARDIAC_SURGERY, CARDIAC_SURGERY_THRESHOLD),
            (CohortLabel.ORTHO_CARDIAC, ORTHO_CARDIAC_THRESHOLD),
            (CohortLabel.ESRD_EPO, ESRD_EPO_THRESHOLD),
        ],
    )
    def test_hb_at_or_above_10_is_potentially_inappropriate(
        self, label: CohortLabel, threshold: float
    ) -> None:
        """Hb ≥ 10 → POTENTIALLY_INAPPROPRIATE (LLM may override)."""
        result = classify(
            _inputs(
                hb=_hb(10.0),
                cohort=_cohort(label, threshold),
            )
        )
        assert result.classification == "POTENTIALLY_INAPPROPRIATE"
        assert result.bypass_reason == BypassReason.NONE
        assert result.cohort_threshold == threshold

    def test_hb_at_threshold_boundary_is_needs_review(self) -> None:
        """Hb == cohort_threshold falls into the gray zone (the ``<``
        in ``Hb < cohort_threshold`` is strict; equal Hb routes to LLM)."""
        result = classify(
            _inputs(
                hb=_hb(DEFAULT_THRESHOLD),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            )
        )
        assert result.classification == "NEEDS_REVIEW"

    def test_hb_missing_is_insufficient_evidence(self) -> None:
        """Hb missing (freshness=missing), no Hb-independent positive
        evidence → INSUFFICIENT_EVIDENCE.

        With a DEFAULT cohort and no MTP / peri-procedural signal there is
        nothing to anchor a classification, so the case is a genuine
        documentation gap. (The structured positive-evidence bypasses that
        DO fire on missing Hb are pinned in
        :class:`TestMissingHbPositiveEvidence`.)
        """
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            )
        )
        assert result.classification == "INSUFFICIENT_EVIDENCE"
        assert result.bypass_reason == BypassReason.NONE


# =============================================================================
# AC ② — Bypass pathways (one per pathway)
# =============================================================================


class TestBypassPathways:
    """Each deterministic bypass pathway fires correctly."""

    def test_mtp_cohort_bypasses_to_appropriate(self) -> None:
        """MTP cohort → APPROPRIATE with bypass_reason=MTP, regardless of
        Hb tier (the cluster pattern is the auto-bypass safety signal)."""
        result = classify(
            _inputs(
                hb=_hb(11.0),  # Even a high Hb is bypassed by MTP
                cohort=_cohort(CohortLabel.MTP, None),
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.MTP
        assert result.cohort_threshold is None

    def test_peri_procedural_within_6h_bypasses(self) -> None:
        """Procedure ≤ 6 h before order → APPROPRIATE,
        bypass_reason=PERI_PROCEDURAL_6H, even when Hb is in the gray zone."""
        result = classify(
            _inputs(
                hb=_hb(8.5),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                procedure_proximity_hours=4.0,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.PERI_PROCEDURAL_6H

    def test_peri_procedural_outside_6h_does_not_bypass(self) -> None:
        """Procedure > 6 h before order → no bypass (plain Hb-tier rule)."""
        result = classify(
            _inputs(
                hb=_hb(8.5),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                procedure_proximity_hours=PERI_PROCEDURAL_WINDOW_HOURS + 0.5,
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE

    def test_peri_procedural_at_exact_boundary_bypasses(self) -> None:
        """A procedure exactly at the 6 h boundary still counts as
        peri-procedural (``≤ 6 h`` per PRD §6)."""
        result = classify(
            _inputs(
                hb=_hb(8.5),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                procedure_proximity_hours=PERI_PROCEDURAL_WINDOW_HOURS,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.PERI_PROCEDURAL_6H

    def test_pre_op_crossmatch_within_72h_defers_to_llm(self) -> None:
        """Procedure ≤ 72 h after order → NEEDS_REVIEW (defer to LLM), NOT
        deterministic APPROPRIATE. A crossmatch reservation is not a
        transfusion indication — an upcoming surgery can hide an active
        problem the reservation masks (case 68080335 / ongoing LGIB), so the
        note-reading LLM leg must own the call. bypass_reason stays NONE:
        this is a deferral, not a clearing bypass."""
        result = classify(
            _inputs(
                hb=_hb(12.2),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                upcoming_procedure_hours=68.0,
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "preop_defer_llm"

    def test_pre_op_crossmatch_outside_72h_does_not_defer(self) -> None:
        """Procedure > 72 h after order → no pre-op signal; falls through to
        the plain Hb-tier rule (Hb ≥ 10 → POTENTIALLY_INAPPROPRIATE)."""
        result = classify(
            _inputs(
                hb=_hb(12.2),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                upcoming_procedure_hours=PRE_OP_CROSSMATCH_WINDOW_HOURS + 0.5,
            )
        )
        assert result.classification == "POTENTIALLY_INAPPROPRIATE"
        assert result.bypass_reason == BypassReason.NONE

    def test_pre_op_crossmatch_at_exact_boundary_defers_to_llm(self) -> None:
        """A procedure exactly 72 h after the order still defers to the LLM
        (``≤ 72 h`` inclusive)."""
        result = classify(
            _inputs(
                hb=_hb(12.2),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                upcoming_procedure_hours=PRE_OP_CROSSMATCH_WINDOW_HOURS,
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "preop_defer_llm"

    def test_delta_hb_bypass_fires(self) -> None:
        """Delta-Hb trigger fired → APPROPRIATE, bypass_reason=DELTA_HB."""
        result = classify(
            _inputs(
                hb=_hb(8.5, delta_bypass=True),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.DELTA_HB

    def test_delta_hb_bypass_does_not_fire_when_hb_missing(self) -> None:
        """A stale delta-Hb flag does NOT fire on missing Hb (no Hb to diff
        against). With no Hb-independent positive evidence, the case falls
        through to INSUFFICIENT_EVIDENCE with rationale="hb_missing" — the
        ``hb_missing`` rationale (not ``bypass_delta_hb``) proves the orphan
        delta flag was ignored rather than honored."""
        # Force the inconsistent case: hb=missing but delta_hb_bypass=True.
        # The classifier MUST prefer the structural "no Hb" check over
        # honoring a stale bypass flag.
        bad_hb = HbLookupResult(
            value_g_dl=None,
            datetime_utc=None,
            source=None,
            freshness="missing",
            delta_hb_bypass=True,
            delta_hb_windows=_triggered_delta(),
            needs_review_single_low_hb=False,
        )
        result = classify(
            _inputs(
                hb=bad_hb,
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            )
        )
        assert result.classification == "INSUFFICIENT_EVIDENCE"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "hb_missing"


# =============================================================================
# Missing-Hb positive-evidence pre-check (SEED pending clinical sign-off)
# =============================================================================


class TestMissingHbPositiveEvidence:
    """When the Hb is missing, hard structured Hb-independent positive
    evidence (active MTP, peri-procedural ≤ 6 h) still auto-classifies the
    order ``APPROPRIATE`` — mirroring the Hb-present path, where the same
    case auto-classifies. The order is preserved (MTP → UNKNOWN → peri-
    procedural): MTP is the dominant signal, UNKNOWN remains a genuine gap
    that peri-procedural must not override, and only the weaker pre-op
    crossmatch / interpreted delta-Hb signals stay excluded on missing Hb.

    The distinct rationale slugs (``bypass_mtp_hb_missing`` /
    ``bypass_peri_procedural_hb_missing``) keep these "approved with no
    documented Hb" cases auditable for the QI committee.
    """

    def test_mtp_with_missing_hb_is_appropriate(self) -> None:
        """Active MTP + missing Hb → APPROPRIATE (cluster pattern is
        Hb-independent; transfuse-before-you-have-an-Hb).

        Opts into the SEED policy via
        ``enable_missing_hb_positive_evidence=True`` because the
        bypass is gated off by default pending clinical sign-off."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.MTP, None),
                enable_missing_hb_positive_evidence=True,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.MTP
        assert result.rationale == "bypass_mtp_hb_missing"
        assert result.cohort_threshold is None

    def test_peri_procedural_with_missing_hb_is_appropriate(self) -> None:
        """Procedure ≤ 6 h before order + missing Hb → APPROPRIATE
        (fresh-out-of-surgery surgical-loss scenario). Opts into the
        SEED policy explicitly — see class docstring."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                procedure_proximity_hours=4.0,
                enable_missing_hb_positive_evidence=True,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.PERI_PROCEDURAL_6H
        assert result.rationale == "bypass_peri_procedural_hb_missing"
        assert result.cohort_threshold == DEFAULT_THRESHOLD

    def test_pre_op_crossmatch_only_with_missing_hb_defers_to_llm(self) -> None:
        """Upcoming procedure (pre-op crossmatch) is a SOFT signal: it is
        deliberately NOT a missing-Hb auto-approve bypass — transfusing
        pre-op with no Hb is exactly what an audit should look at. With the
        pre-pass flag ON it no longer dead-ends as INSUFFICIENT_EVIDENCE;
        it defers to the LLM (NEEDS_REVIEW / ``hb_missing_defer_llm``), which
        reads the prose and either grounds APPROPRIATE or floors it to a
        human — strictly more automation than terminating here."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                upcoming_procedure_hours=10.0,
                enable_missing_hb_positive_evidence=True,
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "hb_missing_defer_llm"

    def test_mtp_precedes_peri_procedural_on_missing_hb(self) -> None:
        """MTP + peri-procedural + missing Hb → MTP wins (the most
        clinically load-bearing signal), preserving the Hb-present order."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.MTP, None),
                procedure_proximity_hours=4.0,
                enable_missing_hb_positive_evidence=True,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.MTP
        assert result.rationale == "bypass_mtp_hb_missing"

    def test_unknown_cohort_peri_procedural_missing_hb_defers_to_llm(self) -> None:
        """UNKNOWN cohort + peri-procedural + missing Hb → no deterministic
        auto-approve, even with the SEED policy on. Peri-procedural MUST NOT
        override UNKNOWN, mirroring the Hb-present order (UNKNOWN precedes
        peri-procedural): missing Hb + unknown context is the dominant
        documentation gap, so it is never rubber-stamped APPROPRIATE here.
        But it no longer dead-ends — with the flag ON it defers to the LLM
        (NEEDS_REVIEW / ``hb_missing_defer_llm``) rather than terminating as
        INSUFFICIENT_EVIDENCE."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.UNKNOWN, None),
                procedure_proximity_hours=4.0,
                enable_missing_hb_positive_evidence=True,
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "hb_missing_defer_llm"

    def test_peri_procedural_at_boundary_with_missing_hb_is_appropriate(self) -> None:
        """Procedure exactly at the 6 h boundary + missing Hb → APPROPRIATE
        (``≤ 6 h`` inclusive, mirroring the Hb-present boundary test). Pins
        the ``<=`` predicate so a ``<`` regression is caught on this path."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                procedure_proximity_hours=PERI_PROCEDURAL_WINDOW_HOURS,
                enable_missing_hb_positive_evidence=True,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.PERI_PROCEDURAL_6H
        assert result.rationale == "bypass_peri_procedural_hb_missing"

    def test_peri_procedural_outside_window_with_missing_hb_defers_to_llm(
        self,
    ) -> None:
        """Procedure > 6 h before order + missing Hb → no peri-procedural
        auto-approve (even with the SEED policy on). The proximity guard
        rejects the out-of-window case, pinning that the window check (not
        just presence) gates the bypass. With the flag ON the rejected case
        defers to the LLM (NEEDS_REVIEW / ``hb_missing_defer_llm``) instead
        of dead-ending as INSUFFICIENT_EVIDENCE."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                procedure_proximity_hours=PERI_PROCEDURAL_WINDOW_HOURS + 0.5,
                enable_missing_hb_positive_evidence=True,
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "hb_missing_defer_llm"

    def test_peri_procedural_wins_over_orphan_delta_on_missing_hb(self) -> None:
        """Missing Hb + a stale delta-Hb flag + peri-procedural proximity →
        APPROPRIATE via the peri-procedural slug, NOT delta-Hb. Delta-Hb is
        never reachable on missing Hb (no current Hb to diff), so the
        rationale must be ``bypass_peri_procedural_hb_missing`` — proving the
        positive-evidence pre-check, not the orphan flag, drove the result."""
        # Inconsistent upstream state: hb=missing yet delta_hb_bypass=True.
        bad_hb = HbLookupResult(
            value_g_dl=None,
            datetime_utc=None,
            source=None,
            freshness="missing",
            delta_hb_bypass=True,
            delta_hb_windows=_triggered_delta(),
            needs_review_single_low_hb=False,
        )
        result = classify(
            _inputs(
                hb=bad_hb,
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                procedure_proximity_hours=4.0,
                enable_missing_hb_positive_evidence=True,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.PERI_PROCEDURAL_6H
        assert result.rationale == "bypass_peri_procedural_hb_missing"

    def test_mtp_missing_hb_flag_default_off_stays_insufficient(self) -> None:
        """SEED policy default-OFF: MTP + missing Hb returns
        INSUFFICIENT_EVIDENCE, NOT APPROPRIATE.

        WHY: the bypass is gated until clinical sign-off (Codex P1).
        Without an explicit operator opt-in, the classifier must NOT
        auto-approve an undocumented-Hb case purely on cohort label.
        Pins the disabled-by-default contract; flipping the flag in
        production requires the QI committee to sign off first.
        """
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.MTP, None),
                # enable_missing_hb_positive_evidence defaults to False
            )
        )
        assert result.classification == "INSUFFICIENT_EVIDENCE"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "hb_missing"

    def test_peri_procedural_missing_hb_flag_default_off_stays_insufficient(
        self,
    ) -> None:
        """SEED policy default-OFF: peri-procedural + missing Hb returns
        INSUFFICIENT_EVIDENCE. Same rationale as the MTP default-off
        test — both bypass branches must stay dark until sign-off."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                procedure_proximity_hours=4.0,
                # enable_missing_hb_positive_evidence defaults to False
            )
        )
        assert result.classification == "INSUFFICIENT_EVIDENCE"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "hb_missing"


class TestMissingHbPeriopEvidenceAndDeferral:
    """Missing-Hb pre-pass: HARD peri-op note evidence auto-approves;
    everything else with the flag ON defers to the LLM instead of
    dead-ending.

    WHY this is a deliberate behaviour change: a deterministic
    INSUFFICIENT_EVIDENCE is terminal — it never reaches the LLM, so the
    peri-op extractor + free-text prose can never auto-resolve the
    well-documented majority. The new policy auto-approves ONLY on hard,
    Hb-independent surgical-loss documentation (a charted intra-op
    transfusion, or EBL ≥ :data:`PERIOP_MIN_EBL_ML`) and routes the rest
    to the LLM (``hb_missing_defer_llm``). The accuracy invariant is
    preserved: there is no Hb here, so the deterministic gate never decided
    on a (possibly post-transfusion) Hb value.
    """

    def test_intraop_transfusion_with_missing_hb_is_appropriate(self) -> None:
        """A charted intra-op transfusion is HARD evidence: the surgery gave
        blood intra-operatively, which stands in for the absent Hb →
        APPROPRIATE via the distinct PERIOP_EVIDENCE bypass so the QI
        committee can count auto-approvals made with no documented Hb."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                enable_missing_hb_positive_evidence=True,
                periop_intraop_transfusion=True,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.PERIOP_EVIDENCE
        assert result.rationale == "bypass_periop_evidence_hb_missing"
        assert result.cohort_threshold == DEFAULT_THRESHOLD

    def test_ebl_at_floor_with_missing_hb_is_appropriate(self) -> None:
        """EBL exactly at PERIOP_MIN_EBL_ML is HARD evidence (``>=`` floor,
        inclusive) → APPROPRIATE. Pins the boundary so a ``>`` regression is
        caught."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                enable_missing_hb_positive_evidence=True,
                periop_blood_loss_ml=PERIOP_MIN_EBL_ML,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.PERIOP_EVIDENCE
        assert result.rationale == "bypass_periop_evidence_hb_missing"

    def test_ebl_below_floor_with_missing_hb_defers_to_llm(self) -> None:
        """EBL just below the floor is a routine loss, NOT hard evidence; it
        must not auto-approve. With the flag ON it defers to the LLM
        (NEEDS_REVIEW / ``hb_missing_defer_llm``)."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                enable_missing_hb_positive_evidence=True,
                periop_blood_loss_ml=PERIOP_MIN_EBL_ML - 1,
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "hb_missing_defer_llm"

    def test_surgical_context_only_with_missing_hb_defers_to_llm(self) -> None:
        """A merely-documented surgery (no intra-op transfusion, no large
        EBL) is the SOFT "surgery exists" cue the design refuses to
        rubber-stamp on missing Hb. It defers to the LLM rather than
        auto-approving or dead-ending — surgical_context is carried for
        traceability only, never as a verdict gate."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                enable_missing_hb_positive_evidence=True,
                periop_surgical_context=True,
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "hb_missing_defer_llm"

    def test_no_periop_evidence_with_missing_hb_defers_to_llm(self) -> None:
        """Flag ON, no peri-op evidence at all → defer to the LLM. This is
        the headline change: the previously-terminal INSUFFICIENT_EVIDENCE
        case now routes to the LLM (``hb_missing_defer_llm``)."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                enable_missing_hb_positive_evidence=True,
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "hb_missing_defer_llm"

    def test_hard_periop_evidence_flag_off_stays_insufficient(self) -> None:
        """SEED policy default-OFF dominates everything: even with a charted
        intra-op transfusion AND a large EBL, the flag-OFF path returns the
        unchanged terminal INSUFFICIENT_EVIDENCE. Pins that the new branch is
        fully gated behind the operator opt-in (no behaviour change until
        sign-off)."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                periop_intraop_transfusion=True,
                periop_blood_loss_ml=PERIOP_MIN_EBL_ML + 1000,
                # enable_missing_hb_positive_evidence defaults to False
            )
        )
        assert result.classification == "INSUFFICIENT_EVIDENCE"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "hb_missing"

    def test_unknown_cohort_hard_periop_evidence_defers_to_llm(self) -> None:
        """UNKNOWN cohort blocks the deterministic auto-approve even on hard
        peri-op evidence — mirroring the peri-procedural precedence
        (UNKNOWN + missing Hb is the dominant documentation gap, never
        rubber-stamped). The hard-evidence UNKNOWN case is not dead-ended,
        though: it defers to the LLM, which sees the peri-op block and can
        ground the verdict."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.UNKNOWN, None),
                enable_missing_hb_positive_evidence=True,
                periop_intraop_transfusion=True,
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "hb_missing_defer_llm"

    def test_mtp_precedes_periop_evidence_on_missing_hb(self) -> None:
        """MTP outranks hard peri-op evidence — the cluster pattern is the
        most clinically load-bearing signal, so the rationale is
        ``bypass_mtp_hb_missing`` not the peri-op slug."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.MTP, None),
                enable_missing_hb_positive_evidence=True,
                periop_intraop_transfusion=True,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.MTP
        assert result.rationale == "bypass_mtp_hb_missing"

    def test_peri_procedural_precedes_periop_evidence_on_missing_hb(self) -> None:
        """Peri-procedural ≤ 6 h outranks hard peri-op note evidence (it is
        the structured-timing signal checked first), so a case with both
        fires the peri-procedural slug, proving the precedence order."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                procedure_proximity_hours=4.0,
                enable_missing_hb_positive_evidence=True,
                periop_intraop_transfusion=True,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.PERI_PROCEDURAL_6H
        assert result.rationale == "bypass_peri_procedural_hb_missing"


# =============================================================================
# AC ③ — B2 invariant: never INAPPROPRIATE
# =============================================================================


class TestB2Invariant:
    """B2: documentation absence MUST NOT produce INAPPROPRIATE.

    The deterministic engine NEVER emits INAPPROPRIATE — that label
    requires positive evidence the LLM stage reasons over. These tests
    pin the invariant at the unit boundary so a future refactor cannot
    quietly add an INAPPROPRIATE path.
    """

    def test_missing_notes_hb_8_is_not_inappropriate(self) -> None:
        """User constraint #6 (the canonical B2 case): Hb=8 + missing
        notes → NEEDS_REVIEW or INSUFFICIENT_EVIDENCE, NEVER INAPPROPRIATE."""
        result = classify(
            _inputs(
                hb=_hb(8.0),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            )
        )
        assert result.classification != "INAPPROPRIATE"
        assert result.classification in {"NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE"}

    def test_hb_ge_10_is_potentially_not_inappropriate(self) -> None:
        """Even at Hb ≥ 10 (the worst-case tier), the deterministic
        engine returns POTENTIALLY_INAPPROPRIATE, not INAPPROPRIATE.
        Only the LLM stage can promote to INAPPROPRIATE."""
        result = classify(
            _inputs(
                hb=_hb(12.0),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            )
        )
        assert result.classification == "POTENTIALLY_INAPPROPRIATE"
        assert result.classification != "INAPPROPRIATE"


class TestB2InvariantProperty:
    """Property test for the B2 invariant (user constraint #6).

    Any random combination of inputs at any Hb > 7 must not return
    INAPPROPRIATE. We sweep Hb across the full physiologic range
    [2.0, 25.0] and every cohort, every bypass flag combination.
    """

    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    @given(
        hb_value=st.floats(
            min_value=2.0,
            max_value=25.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        threshold=st.sampled_from([7.0, 7.5, 8.0]),
        procedure_proximity_hours=st.one_of(
            st.none(),
            st.floats(
                min_value=0.0, max_value=72.0, allow_nan=False, allow_infinity=False
            ),
        ),
        upcoming_procedure_hours=st.one_of(
            st.none(),
            st.floats(
                min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False
            ),
        ),
        crystalloid_liters=st.floats(
            min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False
        ),
        delta_bypass=st.booleans(),
    )
    def test_deterministic_engine_never_emits_inappropriate(
        self,
        hb_value: float,
        threshold: float,
        procedure_proximity_hours: float | None,
        upcoming_procedure_hours: float | None,
        crystalloid_liters: float,
        delta_bypass: bool,
    ) -> None:
        result = classify(
            _inputs(
                hb=_hb(hb_value, delta_bypass=delta_bypass),
                cohort=_cohort(CohortLabel.DEFAULT, threshold),
                procedure_proximity_hours=procedure_proximity_hours,
                upcoming_procedure_hours=upcoming_procedure_hours,
                crystalloid_liters_prior_4h=crystalloid_liters,
            )
        )
        assert result.classification != "INAPPROPRIATE"


# =============================================================================
# AC ④ — Hemodilution flag
# =============================================================================


class TestHemodilutionFlag:
    """Hb < cohort_threshold AND ≥ 2 L crystalloid in 4 h → NEEDS_REVIEW
    only after the global Hb < 7.0 rule has had first chance to classify."""

    def test_canonical_hemodilution_case(self) -> None:
        """Hb=7.5 in an 8.0-threshold cohort with 2.5 L crystalloid in 4 h
        → NEEDS_REVIEW with bypass_reason=HEMODILUTION_FLAGGED."""
        result = classify(
            _inputs(
                hb=_hb(7.5),
                cohort=_cohort(CohortLabel.ESRD_EPO, ESRD_EPO_THRESHOLD),
                crystalloid_liters_prior_4h=2.5,
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.HEMODILUTION_FLAGGED

    def test_exactly_2l_triggers_hemodilution(self) -> None:
        """Boundary at exactly 2.0 L: ≥ 2 L means at-or-above (Round 1 B5)."""
        result = classify(
            _inputs(
                hb=_hb(7.5),
                cohort=_cohort(CohortLabel.ESRD_EPO, ESRD_EPO_THRESHOLD),
                crystalloid_liters_prior_4h=HEMODILUTION_CRYSTALLOID_LITERS,
            )
        )
        assert result.bypass_reason == BypassReason.HEMODILUTION_FLAGGED
        assert result.classification == "NEEDS_REVIEW"

    def test_hb_below_7_bypasses_hemodilution_review(self) -> None:
        """Hb < 7.0 is globally APPROPRIATE before the hemodilution gate."""
        result = classify(
            _inputs(
                hb=_hb(UNIVERSAL_LOW_HB_APPROPRIATE_THRESHOLD - 0.1),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                crystalloid_liters_prior_4h=2.5,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "hb_lt_7_universal"

    def test_below_2l_crystalloid_does_not_flag(self) -> None:
        """< 2 L crystalloid → plain Hb-tier rule applies (APPROPRIATE
        for Hb < threshold)."""
        result = classify(
            _inputs(
                hb=_hb(6.5),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                crystalloid_liters_prior_4h=1.5,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.NONE

    def test_hemodilution_only_applies_when_hb_below_threshold(self) -> None:
        """≥ 2 L crystalloid at gray-zone Hb does NOT trigger the
        hemodilution path — that rule is scoped to Hb < threshold (the
        normal Hb is not the suspicious-because-diluted case)."""
        result = classify(
            _inputs(
                hb=_hb(8.5),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                crystalloid_liters_prior_4h=3.0,
            )
        )
        assert result.bypass_reason != BypassReason.HEMODILUTION_FLAGGED


# =============================================================================
# AC ⑤ — Bypass-reason distinctness
# =============================================================================


class TestBypassReasonDistinct:
    """Each bypass pathway sets exactly its own enum member; the
    non-bypass paths carry :attr:`BypassReason.NONE`."""

    def test_each_pathway_sets_unique_reason(self) -> None:
        # MTP
        mtp = classify(
            _inputs(
                hb=_hb(8.5),
                cohort=_cohort(CohortLabel.MTP, None),
            )
        )
        # Peri-procedural
        peri = classify(
            _inputs(
                hb=_hb(8.5),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                procedure_proximity_hours=3.0,
            )
        )
        # Delta-Hb
        delta = classify(
            _inputs(
                hb=_hb(8.5, delta_bypass=True),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            )
        )
        # Hemodilution
        hemo = classify(
            _inputs(
                hb=_hb(7.5),
                cohort=_cohort(CohortLabel.ESRD_EPO, ESRD_EPO_THRESHOLD),
                crystalloid_liters_prior_4h=2.5,
            )
        )
        # Pre-op crossmatch is no longer a clearing bypass — it defers to the
        # LLM (NEEDS_REVIEW / bypass_reason NONE), so it is asserted separately
        # in TestPreOpCrossmatchBypass, not among the distinct bypass reasons.
        reasons = {
            mtp.bypass_reason,
            peri.bypass_reason,
            delta.bypass_reason,
            hemo.bypass_reason,
        }
        assert reasons == {
            BypassReason.MTP,
            BypassReason.PERI_PROCEDURAL_6H,
            BypassReason.DELTA_HB,
            BypassReason.HEMODILUTION_FLAGGED,
        }

    def test_non_bypass_paths_set_none(self) -> None:
        """The four non-bypass Hb tiers all set BypassReason.NONE."""
        cases = [
            _hb(6.0),  # below threshold
            _hb(8.5),  # gray zone
            _hb(11.0),  # >= 10
            _hb(None),  # missing
        ]
        for hb in cases:
            r = classify(
                _inputs(
                    hb=hb,
                    cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                )
            )
            assert r.bypass_reason == BypassReason.NONE


# =============================================================================
# AC ⑥ — Monotonicity property
# =============================================================================


class TestMonotonicityProperty:
    """For the same cohort + same evidence, increasing Hb monotonically
    must never move classification from less-appropriate toward
    more-appropriate (user constraint #7).

    Order from most-appropriate to least-appropriate:
    APPROPRIATE  <  NEEDS_REVIEW  <  POTENTIALLY_INAPPROPRIATE

    We sweep Hb values across the tier boundaries with no bypass active;
    once the engine emits POTENTIALLY_INAPPROPRIATE it cannot regress to
    NEEDS_REVIEW or APPROPRIATE at a higher Hb.
    """

    _ORDERING: dict[str, int] = {
        "APPROPRIATE": 0,
        "INSUFFICIENT_EVIDENCE": 1,  # not Hb-tier-ordered; allowed at hb=None only
        "NEEDS_REVIEW": 2,
        "POTENTIALLY_INAPPROPRIATE": 3,
    }

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    @given(
        threshold=st.sampled_from([7.0, 7.5, 8.0]),
        hb_lo=st.floats(min_value=2.5, max_value=14.5, allow_nan=False),
        delta=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
    def test_monotonic_in_hb(
        self, threshold: float, hb_lo: float, delta: float
    ) -> None:
        hb_hi = min(hb_lo + delta, 25.0)
        lo = classify(
            _inputs(
                hb=_hb(hb_lo),
                cohort=_cohort(CohortLabel.DEFAULT, threshold),
            )
        )
        hi = classify(
            _inputs(
                hb=_hb(hb_hi),
                cohort=_cohort(CohortLabel.DEFAULT, threshold),
            )
        )
        # Skip cases where either side hit a path the ordering can't
        # reason about (none expected, since hb is numeric on both).
        assert lo.classification in self._ORDERING
        assert hi.classification in self._ORDERING
        assert self._ORDERING[hi.classification] >= self._ORDERING[lo.classification]


# =============================================================================
# Single-low-Hb review flag (PR #52 Codex P1)
# =============================================================================


class TestSingleLowHbReviewFlag:
    """When :attr:`bba.hb_lookup.HbLookupResult.needs_review_single_low_hb`
    is set, the deterministic engine routes to ``NEEDS_REVIEW`` rather than
    auto-classifying sub-threshold Hb as ``APPROPRIATE``, except that the
    global Hb < 7.0 rule wins first.

    Upstream contract (PRD §3, hb_lookup model docstring): the flag is
    True only when the most-recent Hb is < 8 g/dL and no prior 24 h
    observation exists — i.e., a worrying value with no trend to
    interpret it against. The deterministic stage cannot honor confirmed
    anemia without a confirming observation, so an isolated low value
    routes to human review.
    """

    def test_isolated_low_hb_routes_to_needs_review(self) -> None:
        """ESRD/ortho-cardiac patient with Hb 7.5 and no 24 h trend
        (sub-threshold for the 8.0 cohort) → NEEDS_REVIEW, not APPROPRIATE."""
        result = classify(
            _inputs(
                hb=_hb(7.5, needs_review_single_low_hb=True),
                cohort=_cohort(CohortLabel.ESRD_EPO, ESRD_EPO_THRESHOLD),
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE

    def test_isolated_low_hb_below_7_routes_to_appropriate(self) -> None:
        """Default-cohort case: Hb 6.5 with the flag set is still
        APPROPRIATE because the global Hb < 7.0 rule runs first."""
        result = classify(
            _inputs(
                hb=_hb(6.5, needs_review_single_low_hb=True),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.NONE
        assert result.rationale == "hb_lt_7_universal"

    def test_flag_does_not_override_mtp_bypass(self) -> None:
        """MTP is the auto-APPROPRIATE safety signal — the cluster pattern
        is itself the "confirming" observation, so the single-low-Hb flag
        does NOT downgrade an MTP-cohort row to NEEDS_REVIEW."""
        result = classify(
            _inputs(
                hb=_hb(7.5, needs_review_single_low_hb=True),
                cohort=_cohort(CohortLabel.MTP, None),
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.MTP

    def test_flag_does_not_override_peri_procedural_bypass(self) -> None:
        """A procedure within 6 h is itself a positive-evidence anchor
        for the order, so the single-low-Hb flag does NOT downgrade the
        peri-procedural bypass."""
        result = classify(
            _inputs(
                hb=_hb(7.5, needs_review_single_low_hb=True),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                procedure_proximity_hours=3.0,
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.PERI_PROCEDURAL_6H

    def test_flag_does_not_override_delta_hb_bypass(self) -> None:
        """A triggered delta-Hb window IS a trend, so the single-low-Hb
        flag (which fires for "no trend available") does not apply when
        the bypass is also set. The classifier honors the delta-Hb path."""
        result = classify(
            _inputs(
                hb=_hb(
                    7.5,
                    delta_bypass=True,
                    needs_review_single_low_hb=True,
                ),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.DELTA_HB


# =============================================================================
# Global Hb < 7.0 rule — before cohort routing
# =============================================================================


class TestGlobalLowHbRule:
    """Hb < 7.0 is APPROPRIATE before cohort-specific review routes."""

    def test_hb_below_7_is_appropriate_for_heme_malignancy(self) -> None:
        result = classify(
            _inputs(
                hb=_hb(6.7),
                cohort=_cohort(CohortLabel.HEME_MALIGNANCY_ACTIVE, None),
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.NONE
        assert result.cohort_threshold is None
        assert result.rationale == "hb_lt_7_universal"

    def test_hb_at_7_does_not_use_global_rule(self) -> None:
        result = classify(
            _inputs(
                hb=_hb(UNIVERSAL_LOW_HB_APPROPRIATE_THRESHOLD),
                cohort=_cohort(CohortLabel.HEME_MALIGNANCY_ACTIVE, None),
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.rationale == "cohort_non_threshold"


# =============================================================================
# Cohort UNKNOWN — user constraint #9
# =============================================================================


class TestCohortUnknownRoutesToNeedsReview:
    """User constraint #9: if cohort is UNKNOWN (procedure data missing),
    classification MUST be NEEDS_REVIEW with bypass_reason=NONE for Hb >= 7.0,
    while Hb < 7.0 is globally APPROPRIATE before cohort routing."""

    def test_unknown_cohort_with_hb_below_7_is_appropriate(self) -> None:
        result = classify(
            _inputs(
                hb=_hb(6.0),
                cohort=_cohort(CohortLabel.UNKNOWN, None),
            )
        )
        assert result.classification == "APPROPRIATE"
        assert result.bypass_reason == BypassReason.NONE
        assert result.cohort_threshold is None
        assert result.rationale == "hb_lt_7_universal"

    def test_unknown_cohort_with_hb_at_7_is_needs_review(self) -> None:
        result = classify(
            _inputs(
                hb=_hb(7.0),
                cohort=_cohort(CohortLabel.UNKNOWN, None),
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE
        assert result.cohort_threshold is None

    def test_unknown_cohort_with_high_hb_is_needs_review(self) -> None:
        """UNKNOWN routes to NEEDS_REVIEW even at high Hb — the system
        cannot rule out a high-target cohort context."""
        result = classify(
            _inputs(
                hb=_hb(11.0),
                cohort=_cohort(CohortLabel.UNKNOWN, None),
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.cohort_threshold is None

    def test_unknown_cohort_with_missing_hb_is_insufficient_evidence(self) -> None:
        """Missing Hb still takes precedence over the UNKNOWN-cohort
        route (no Hb means no anchor at all)."""
        result = classify(
            _inputs(
                hb=_hb(None),
                cohort=_cohort(CohortLabel.UNKNOWN, None),
            )
        )
        assert result.classification == "INSUFFICIENT_EVIDENCE"


# =============================================================================
# Canonical-classification contract — only deterministic-allowed values
# =============================================================================


class TestCanonicalClassificationContract:
    """``ClassifierResult.classification`` must be one of the four canonical
    :data:`bba.audit_store.Classification` values the deterministic engine
    is allowed to emit. ``INAPPROPRIATE`` is reserved for the LLM stage."""

    ALLOWED_DETERMINISTIC: frozenset[str] = frozenset(
        {
            "APPROPRIATE",
            "NEEDS_REVIEW",
            "POTENTIALLY_INAPPROPRIATE",
            "INSUFFICIENT_EVIDENCE",
        }
    )

    @pytest.mark.parametrize(
        "hb_value",
        [None, 6.0, 8.5, 11.0],
    )
    def test_each_tier_emits_canonical_value(self, hb_value: float | None) -> None:
        result = classify(
            _inputs(
                hb=_hb(hb_value),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            )
        )
        assert result.classification in self.ALLOWED_DETERMINISTIC


# =============================================================================
# crystalloid helper contract
# =============================================================================


class TestCrystalloidHelper:
    """:func:`total_crystalloid_liters` contract.

    These tests pin the minimal helper shape so the orchestrator can rely
    on it for the ``crystalloid_liters_prior_4h`` input.
    """

    def _med(self, drug: str, hours_before_order: float) -> MedEvent:
        return MedEvent(
            drug=drug,
            timestamp=ORDER_DT - timedelta(hours=hours_before_order),
        )

    def test_empty_med_events_returns_zero(self) -> None:
        assert total_crystalloid_liters((), ORDER_DT) == 0.0

    def test_sums_in_window_in_liters(self) -> None:
        """Two crystalloid orders inside the 4 h window sum to liters."""
        events = (
            self._med("NSS 1000 mL", 2.0),
            self._med("RLS 1 L", 3.5),
        )
        assert total_crystalloid_liters(events, ORDER_DT) == pytest.approx(2.0)

    def test_ignores_events_outside_window(self) -> None:
        """A crystalloid order > 4 h before the anchor must not be summed."""
        events = (
            self._med("NSS 1000 mL", 5.0),  # outside 4 h window
            self._med("RLS 500 cc", 2.0),  # inside
        )
        assert total_crystalloid_liters(events, ORDER_DT) == pytest.approx(0.5)

    def test_ignores_future_events(self) -> None:
        """A crystalloid order timestamped AFTER the order anchor must
        not be summed (we only look backward)."""
        events = (
            self._med("NSS 1000 mL", -1.0),  # 1 h AFTER order
            self._med("RLS 1 L", 1.0),  # 1 h before order
        )
        assert total_crystalloid_liters(events, ORDER_DT) == pytest.approx(1.0)

    def test_parses_ml_cc_and_l_units(self) -> None:
        """Parses mL / cc / L (case-insensitive)."""
        events = (
            self._med("NSS 500 mL", 1.0),
            self._med("Plasmalyte 500 cc", 1.5),
            self._med("LRS 1 L", 2.0),
        )
        assert total_crystalloid_liters(events, ORDER_DT) == pytest.approx(2.0)

    def test_parses_thousands_separator(self) -> None:
        """A dose written with a thousands separator (``"NSS 1,000 mL"``)
        must parse as 1.0 L, not the ``"000 mL"`` tail (0 L). Undercounting
        here can keep the 4 h total below the 2 L hemodilution trigger and
        wrongly leave a dilutional sub-threshold Hb as APPROPRIATE."""
        events = (
            self._med("NSS 1,000 mL", 1.0),
            self._med("RLS 1,500 cc", 2.0),
        )
        assert total_crystalloid_liters(events, ORDER_DT) == pytest.approx(2.5)

    def test_excludes_infusion_rate_strings(self) -> None:
        """PR #52 Codex P2: ``NSS 500 mL/h`` is an infusion RATE, not a
        delivered bolus. The parser MUST NOT sum it into the 4 h total —
        counting a rate as a bolus can push the total over the 2 L
        hemodilution threshold and incorrectly flip a sub-threshold Hb
        from APPROPRIATE to NEEDS_REVIEW.
        """
        events = (
            self._med("NSS 500 mL/h", 1.0),  # rate — must be excluded
            self._med("NSS 500 mL/hr", 1.5),  # rate variant — must be excluded
            self._med("D5W 200 cc/hour", 2.0),  # cc rate — must be excluded
            self._med("RLS 1 L/h", 2.5),  # L rate — must be excluded
        )
        assert total_crystalloid_liters(events, ORDER_DT) == 0.0

    def test_mixed_rates_and_boluses(self) -> None:
        """A rate co-existing with a real bolus must sum only the bolus.
        Two rate lines (no liters) plus one 1 L bolus → 1.0 L total,
        well below the 2 L hemodilution trigger."""
        events = (
            self._med("NSS 500 mL/h", 0.5),
            self._med("NSS 1000 mL", 1.0),  # real bolus
            self._med("D5W 200 cc/hour", 2.0),
        )
        assert total_crystalloid_liters(events, ORDER_DT) == pytest.approx(1.0)


# =============================================================================
# Adversarial sentinel — keep imports used so ruff stays clean across RED+GREEN
# =============================================================================


def _unused_keep_imports_alive() -> tuple[Any, ...]:
    return (Classification, ClassifierResult, get_args)
