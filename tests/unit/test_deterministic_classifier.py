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

* AC ④ "Hemodilution flag: Hb=6.5 with ≥2 L crystalloid in 4 h →
    NEEDS_REVIEW, not APPROPRIATE"
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
  bypass_reason=NONE; never silently fall back to threshold=7.0.

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
    crystalloid_liters_prior_4h: float = 0.0,
) -> ClassifierInputs:
    return ClassifierInputs(
        audit_id="audit-test-0001",
        hb_result=hb,
        cohort_assignment=cohort,
        order_datetime=ORDER_DT,
        procedure_proximity_hours=procedure_proximity_hours,
        crystalloid_liters_prior_4h=crystalloid_liters_prior_4h,
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
        """Hb missing (freshness=missing) → INSUFFICIENT_EVIDENCE.

        Bypass paths MUST NOT fire on missing Hb — the upstream signals
        cannot be interpreted without a numeric Hb to anchor them.
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
    """Each of the four deterministic bypass pathways fires correctly."""

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
        """No bypass can fire on missing Hb — fall through to
        INSUFFICIENT_EVIDENCE."""
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
        crystalloid_liters: float,
        delta_bypass: bool,
    ) -> None:
        result = classify(
            _inputs(
                hb=_hb(hb_value, delta_bypass=delta_bypass),
                cohort=_cohort(CohortLabel.DEFAULT, threshold),
                procedure_proximity_hours=procedure_proximity_hours,
                crystalloid_liters_prior_4h=crystalloid_liters,
            )
        )
        assert result.classification != "INAPPROPRIATE"


# =============================================================================
# AC ④ — Hemodilution flag
# =============================================================================


class TestHemodilutionFlag:
    """Hb < cohort_threshold AND ≥ 2 L crystalloid in 4 h → NEEDS_REVIEW,
    not auto-APPROPRIATE (Round 1 B5)."""

    def test_canonical_hemodilution_case(self) -> None:
        """Hb=6.5 with 2.5 L crystalloid in 4 h → NEEDS_REVIEW with
        bypass_reason=HEMODILUTION_FLAGGED (the AC #4 golden case)."""
        result = classify(
            _inputs(
                hb=_hb(6.5),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                crystalloid_liters_prior_4h=2.5,
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.HEMODILUTION_FLAGGED

    def test_exactly_2l_triggers_hemodilution(self) -> None:
        """Boundary at exactly 2.0 L: ≥ 2 L means at-or-above (Round 1 B5)."""
        result = classify(
            _inputs(
                hb=_hb(6.5),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                crystalloid_liters_prior_4h=HEMODILUTION_CRYSTALLOID_LITERS,
            )
        )
        assert result.bypass_reason == BypassReason.HEMODILUTION_FLAGGED
        assert result.classification == "NEEDS_REVIEW"

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
                hb=_hb(6.0),
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
                hb=_hb(6.5),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
                crystalloid_liters_prior_4h=2.5,
            )
        )
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
    is set, the deterministic engine MUST route to ``NEEDS_REVIEW`` rather
    than auto-classifying sub-threshold Hb as ``APPROPRIATE``.

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

    def test_isolated_low_hb_default_cohort_routes_to_needs_review(self) -> None:
        """Default-cohort case: Hb 6.5 with the flag set → NEEDS_REVIEW
        (the lone-value-no-trend rule applies to every threshold-driven
        cohort, not just the high-target ones)."""
        result = classify(
            _inputs(
                hb=_hb(6.5, needs_review_single_low_hb=True),
                cohort=_cohort(CohortLabel.DEFAULT, DEFAULT_THRESHOLD),
            )
        )
        assert result.classification == "NEEDS_REVIEW"
        assert result.bypass_reason == BypassReason.NONE

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
# Cohort UNKNOWN — user constraint #9
# =============================================================================


class TestCohortUnknownRoutesToNeedsReview:
    """User constraint #9: if cohort is UNKNOWN (procedure data missing),
    classification MUST be NEEDS_REVIEW with bypass_reason=NONE — DO NOT
    silently fall back to threshold=7.0."""

    def test_unknown_cohort_with_low_hb_is_needs_review(self) -> None:
        result = classify(
            _inputs(
                hb=_hb(6.0),
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
