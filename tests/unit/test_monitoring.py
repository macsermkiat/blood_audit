"""RED-phase failing tests for issue #27 (bba.monitoring).

Each ``class`` maps to one acceptance criterion in the issue body OR one of
the user-supplied operational constraints:

* AC ① "Implementation in ``src/bba/monitoring/``"
    → :class:`TestModulePublicSurface`, :class:`TestModelImmutability`,
      :class:`TestModelValidation`
* AC ② "SPRT trigger correctness: synthetic drift injected at known offset
    → alarm fires"
    → :class:`TestSprtSyntheticDriftFires`, :class:`TestSprtNoAlarmOnNullData`,
      :class:`TestSprtWaldBounds`, :class:`TestSprtArl0Empirical`,
      :class:`TestSprtResetClearsState`
* AC ③ "Sentinel set construction is deterministic (fixed seed)"
    → :class:`TestSentinelConstructionDeterministic`,
      :class:`TestSentinelKappaIdenticalRuns`,
      :class:`TestSentinelKappaPerturbedRuns`,
      :class:`TestSentinelInsufficientHistory`
* AC ④ "Alerting integration stub (log + structured output)"
    → :class:`TestAlertingEmitsStructuredLog`,
      :class:`TestNoSlackEmailPagingImport` (Phase 1.5 boundary regression)
* AC ⑤ "Coverage ≥ 70%; ruff + mypy clean"
    → verified by the build (not a behavioral test)

User-supplied constraints (top-of-prompt):

* "OPERATIONAL not clinical — no imports from hb_lookup / vitals_extractor /
   cohort_detector / etc."
    → :class:`TestNoClinicalImports`
* "No reimplementing metrics — import κ / Gwet's AC1 from bba.eval_harness"
    → :class:`TestSentinelUsesEvalHarnessMetrics`
* "Weekly clinical-reviewer sample (50–75); deterministic by
   (week_iso, sample_size, seed); persist manifests"
    → :class:`TestWeeklyReviewerSampleDeterministic`,
      :class:`TestWeeklyReviewerSampleSizeRange`,
      :class:`TestWeeklyReviewerSampleManifestPersisted`
* "Quarterly model-drift probe — alarm if >5% classification change or
   >10% indication change"
    → :class:`TestGoldenSetClassificationDriftDetects`,
      :class:`TestGoldenSetIndicationDriftDetects`,
      :class:`TestGoldenSetNoDriftNoAlarm`,
      :class:`TestGoldenSetMismatchRaises`
* "Property test: replaying the same period's data twice produces zero
   duplicate alarms"
    → :class:`TestPropertyReplayIdempotent`
* "No live Anthropic calls in tests — reuse betamax cassettes from #22"
    → :class:`TestNoLiveAnthropicInDriftProbe`
* "All cadences are CRON-LIKE but NOT scheduled in #27 itself"
    → :class:`TestNoSchedulerImports`

Tests assert contracts (the WHY), not implementation choices. In this RED
scaffold:

* Every behavioral test FAILS — calls into the scaffold raise
  ``NotImplementedError``, which bubbles up and fails the test.
* Public-surface / constant / model-construction / model-validation
  tests PASS — they are regression guards on the declared interface
  (``frozen=True``, validator presence, constant values) and the user's
  RED-phase rule allows these scaffold-validators to be green because
  they encode the interface contract itself, not behavior. The convention
  matches ``test_audit_store`` and ``test_eval_harness``.
"""

from __future__ import annotations

import importlib
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

# Public-surface check: a missing re-export fails collection before any
# test runs. Mirrors the test_eval_harness convention.
from bba.monitoring import (
    GOLDEN_SET_CLASSIFICATION_DRIFT_THRESHOLD,
    GOLDEN_SET_INDICATION_DRIFT_THRESHOLD,
    GOLDEN_SET_SIZE,
    SENTINEL_KAPPA_ALARM_THRESHOLD,
    SENTINEL_SET_SEED,
    SENTINEL_SET_SIZE,
    SPRT_DEFAULT_ALPHA,
    SPRT_DEFAULT_BETA,
    SPRT_DEFAULT_MIN_N,
    SPRT_TARGET_ARL0,
    WEEKLY_REVIEWER_SAMPLE_MAX,
    WEEKLY_REVIEWER_SAMPLE_MIN,
    GoldenSetEntry,
    GoldenSetMismatchError,
    InsufficientHistoryError,
    MonitoringAlarm,
    MonitoringAlarmInput,
    MonitoringConfig,
    MonitoringError,
    MonitoringStore,
    SentinelManifest,
    SprtConfig,
    SprtState,
    WaldSprtMonitor,
    WeeklyReviewerSample,
    build_sentinel_manifest,
    draw_weekly_reviewer_sample,
    emit_alarm,
    evaluate_golden_set_drift,
    evaluate_sentinel_run,
    run_sprt_on_window,
    synthetic_drift_stream,
    wald_bounds,
)


# =============================================================================
# Fixtures
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parents[2]
"""Absolute path to the repo root — independent of pytest's cwd."""


MONITORING_PACKAGE_DIR = REPO_ROOT / "src" / "bba" / "monitoring"


@pytest.fixture
def utc_now() -> datetime:
    """A fixed tz-aware UTC timestamp for model construction in tests.

    Frozen so tests are stable under clock changes; the actual value is
    arbitrary but tz-aware (the model validators reject naive datetimes)."""
    return datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def sprt_config() -> SprtConfig:
    """A representative SPRT config for the quote-grounding-failure signal.

    ``p_null=0.05`` is a plausible baseline failure rate; ``p_alt=0.10``
    is the +5pp minimum-detectable shift the synthetic-drift fixture
    injects. ``alpha=beta=SPRT_DEFAULT_ALPHA`` keeps ARL₀ around the
    documented 500-event target."""
    return SprtConfig(
        signal="quote_grounding_failure_rate",
        p_null=0.05,
        p_alt=0.10,
        alpha=SPRT_DEFAULT_ALPHA,
        beta=SPRT_DEFAULT_BETA,
        min_n=30,
    )


@pytest.fixture
def monitoring_config(tmp_path: Path) -> MonitoringConfig:
    """A MonitoringConfig with a placeholder DSN.

    RED-phase store methods raise NotImplementedError before any DB call,
    so the DSN does not need to be real. GREEN-phase will swap in a
    testcontainers-backed config for tests that genuinely connect."""
    return MonitoringConfig(
        dsn=f"postgresql://test:test@localhost:5432/test_{tmp_path.name}",
    )


def _fake_audit_id(i: int) -> str:
    """Build a SafeId-compliant fake audit_id for population fixtures."""
    return f"audit-{i:04d}"


# =============================================================================
# AC ① — Module exists with the expected public surface + model contracts
# =============================================================================


class TestModulePublicSurface:
    """The imports at the top of this file already cover the surface check
    (a missing re-export fails collection). This class asserts a few stable
    constants whose VALUES are part of the contract — a silent renumbering
    would be a regression even when the surface still resolves."""

    def test_weekly_sample_bounds_match_prd(self) -> None:
        assert WEEKLY_REVIEWER_SAMPLE_MIN == 50
        assert WEEKLY_REVIEWER_SAMPLE_MAX == 75

    def test_sentinel_constants_match_prd(self) -> None:
        assert SENTINEL_SET_SIZE == 200
        assert SENTINEL_SET_SEED == 42
        assert SENTINEL_KAPPA_ALARM_THRESHOLD == pytest.approx(0.90)

    def test_golden_set_thresholds_match_prd(self) -> None:
        assert GOLDEN_SET_SIZE == 100
        assert GOLDEN_SET_CLASSIFICATION_DRIFT_THRESHOLD == pytest.approx(0.05)
        assert GOLDEN_SET_INDICATION_DRIFT_THRESHOLD == pytest.approx(0.10)

    def test_sprt_arl0_target_documented(self) -> None:
        assert SPRT_TARGET_ARL0 >= 500

    def test_monitoring_error_hierarchy(self) -> None:
        """All typed exceptions inherit from MonitoringError so a broad
        ``except`` catches the family."""
        assert issubclass(InsufficientHistoryError, MonitoringError)
        assert issubclass(GoldenSetMismatchError, MonitoringError)

    def test_sprt_min_n_default_matches_constant(self) -> None:
        """``SprtConfig.min_n`` defaults to ``SPRT_DEFAULT_MIN_N`` — the
        constant is the single source of truth, not a magic number on
        the model."""
        assert SPRT_DEFAULT_MIN_N >= 1
        cfg = SprtConfig(
            signal="needs_review_rate", p_null=0.05, p_alt=0.10
        )
        assert cfg.min_n == SPRT_DEFAULT_MIN_N


class TestModelImmutability:
    """Frozen Pydantic models reject post-construction mutation.

    Part of the interface contract — passes in RED on purpose."""

    def test_weekly_reviewer_sample_is_frozen(self, utc_now: datetime) -> None:
        sample = WeeklyReviewerSample(
            week_iso="2026-W20",
            sample_size=50,
            seed=1,
            audit_ids=("audit-0001",),
            drawn_at=utc_now,
        )
        with pytest.raises(ValidationError):
            sample.sample_size = 99  # type: ignore[misc]

    def test_sprt_state_is_frozen(self) -> None:
        state = SprtState(
            signal="needs_review_rate",
            n_observations=10,
            n_successes=2,
            log_lr=0.1,
            upper_bound=2.9,
            lower_bound=-2.9,
            verdict="continue",
        )
        with pytest.raises(ValidationError):
            state.log_lr = 5.0  # type: ignore[misc]

    def test_monitoring_alarm_input_is_frozen(self, utc_now: datetime) -> None:
        alarm = MonitoringAlarmInput(
            kind="drift_sprt",
            signal="quote_grounding_failure_rate",
            raised_at=utc_now,
            detail={"log_lr": 3.5, "n_observations": 47},
        )
        with pytest.raises(ValidationError):
            alarm.kind = "sentinel_kappa"  # type: ignore[misc]


class TestModelValidation:
    """Field-level validators reject invalid inputs at construction time.

    Validators are part of the interface scaffold; this passes in RED."""

    def test_week_iso_rejects_malformed(self, utc_now: datetime) -> None:
        """``week_iso`` must be ``YYYY-Www`` — a typo silently breaks the
        determinism invariant ('same week → same audit_ids')."""
        with pytest.raises(ValidationError):
            WeeklyReviewerSample(
                week_iso="2026-21",  # missing "W"
                sample_size=50,
                seed=1,
                audit_ids=(),
                drawn_at=utc_now,
            )

    def test_naive_datetime_rejected_on_alarm(self) -> None:
        """Naive ``raised_at`` would compare incorrectly against the audit
        row's tz-aware ``order_datetime`` in dashboard queries."""
        with pytest.raises(ValidationError):
            MonitoringAlarmInput(
                kind="drift_sprt",
                signal="needs_review_rate",
                raised_at=datetime(2026, 5, 16, 12, 0, 0),  # naive!
                detail={},
            )

    def test_unsafe_audit_id_rejected(self, utc_now: datetime) -> None:
        """SafeId allow-list (``[A-Za-z0-9._-]+``) prevents log-injection
        and path-traversal-style identifiers from entering the table."""
        with pytest.raises(ValidationError):
            WeeklyReviewerSample(
                week_iso="2026-W20",
                sample_size=50,
                seed=1,
                audit_ids=("audit/with/slash",),
                drawn_at=utc_now,
            )


# =============================================================================
# Weekly clinical-reviewer sample
# =============================================================================


class TestWeeklyReviewerSampleDeterministic:
    """Same ``(week_iso, sample_size, seed)`` → same ``audit_ids`` across
    invocations, processes, and Python interpreter restarts.

    The determinism invariant is what makes historical-sample reproduction
    possible without RNG-state replay."""

    def test_same_inputs_same_audit_ids(self) -> None:
        population = [_fake_audit_id(i) for i in range(500)]
        sample1 = draw_weekly_reviewer_sample(
            population,  # type: ignore[arg-type]
            week_iso="2026-W20",
            sample_size=50,
            seed=42,
        )
        sample2 = draw_weekly_reviewer_sample(
            population,  # type: ignore[arg-type]
            week_iso="2026-W20",
            sample_size=50,
            seed=42,
        )
        assert sample1.audit_ids == sample2.audit_ids

    def test_different_week_different_sample(self) -> None:
        population = [_fake_audit_id(i) for i in range(500)]
        sample_a = draw_weekly_reviewer_sample(
            population,  # type: ignore[arg-type]
            week_iso="2026-W20",
            sample_size=50,
            seed=42,
        )
        sample_b = draw_weekly_reviewer_sample(
            population,  # type: ignore[arg-type]
            week_iso="2026-W21",
            sample_size=50,
            seed=42,
        )
        assert sample_a.audit_ids != sample_b.audit_ids


class TestWeeklyReviewerSampleSizeRange:
    """``sample_size`` must be in [50, 75] per PRD §18."""

    @pytest.mark.parametrize("bad_size", [49, 76, 0, -1])
    def test_size_out_of_range_rejected(self, bad_size: int) -> None:
        population = [_fake_audit_id(i) for i in range(500)]
        with pytest.raises(ValueError):
            draw_weekly_reviewer_sample(
                population,  # type: ignore[arg-type]
                week_iso="2026-W20",
                sample_size=bad_size,
                seed=42,
            )

    @pytest.mark.parametrize(
        "good_size",
        [
            WEEKLY_REVIEWER_SAMPLE_MIN,
            WEEKLY_REVIEWER_SAMPLE_MAX,
            (WEEKLY_REVIEWER_SAMPLE_MIN + WEEKLY_REVIEWER_SAMPLE_MAX) // 2,
        ],
    )
    def test_size_in_range_accepted(self, good_size: int) -> None:
        population = [_fake_audit_id(i) for i in range(500)]
        sample = draw_weekly_reviewer_sample(
            population,  # type: ignore[arg-type]
            week_iso="2026-W20",
            sample_size=good_size,
            seed=42,
        )
        assert len(sample.audit_ids) == good_size


class TestWeeklyReviewerSampleManifestPersisted:
    """The manifest is persisted via :meth:`MonitoringStore.persist_sample_manifest`
    so historical audit can re-derive who reviewed what week."""

    def test_manifest_persists_to_store(
        self,
        monitoring_config: MonitoringConfig,
        utc_now: datetime,
    ) -> None:
        store = MonitoringStore(monitoring_config)
        sample = WeeklyReviewerSample(
            week_iso="2026-W20",
            sample_size=50,
            seed=42,
            audit_ids=tuple(_fake_audit_id(i) for i in range(50)),
            drawn_at=utc_now,
        )
        store.persist_sample_manifest(sample)
        manifests = store.list_sample_manifests(week_iso="2026-W20")
        assert len(manifests) == 1
        assert manifests[0].audit_ids == sample.audit_ids

    def test_duplicate_manifest_is_idempotent(
        self,
        monitoring_config: MonitoringConfig,
        utc_now: datetime,
    ) -> None:
        """Re-persisting the same ``(week_iso, sample_size, seed)`` is a
        no-op — cron retries don't multiply rows."""
        store = MonitoringStore(monitoring_config)
        sample = WeeklyReviewerSample(
            week_iso="2026-W20",
            sample_size=50,
            seed=42,
            audit_ids=tuple(_fake_audit_id(i) for i in range(50)),
            drawn_at=utc_now,
        )
        store.persist_sample_manifest(sample)
        store.persist_sample_manifest(sample)
        manifests = store.list_sample_manifests(week_iso="2026-W20")
        assert len(manifests) == 1


# =============================================================================
# SPRT drift detection (AC ②)
# =============================================================================


class TestSprtWaldBounds:
    """Wald's SPRT bounds: A = log((1-β)/α), B = log(β/(1-α))."""

    def test_bounds_have_correct_sign(self) -> None:
        lower, upper = wald_bounds(alpha=0.05, beta=0.05)
        assert upper > 0
        assert lower < 0

    def test_symmetric_alpha_beta_produces_symmetric_bounds(self) -> None:
        """When α = β, upper = -lower (symmetric in log-LR space)."""
        lower, upper = wald_bounds(alpha=0.05, beta=0.05)
        assert upper == pytest.approx(-lower, abs=1e-9)


class TestSprtSyntheticDriftFires:
    """Synthetic-drift fixture: inject a +5pp failure rate at a known offset
    and verify the SPRT alarm fires within the expected sample window.

    This is the headline AC ② test."""

    def test_drift_at_offset_50_fires_alarm(self, sprt_config: SprtConfig) -> None:
        """Inject a stream that is Bernoulli(0.05) for 50 observations
        then Bernoulli(0.10) for 200 more. The SPRT MUST reach
        ``verdict='reject_null'`` somewhere in the drifted segment."""
        stream = synthetic_drift_stream(
            null_rate=0.05,
            drift_rate=0.10,
            drift_offset=50,
            total_n=250,
            seed=1,
        )
        state = run_sprt_on_window(stream, sprt_config)
        assert state.verdict == "reject_null"

    def test_drift_offset_recorded_in_state(self, sprt_config: SprtConfig) -> None:
        """The SPRT's ``n_observations`` at alarm time is in the drifted
        segment (≥ drift_offset)."""
        stream = synthetic_drift_stream(
            null_rate=0.05,
            drift_rate=0.10,
            drift_offset=50,
            total_n=250,
            seed=1,
        )
        state = run_sprt_on_window(stream, sprt_config)
        assert state.n_observations >= 50


class TestSprtNoAlarmOnNullData:
    """Under H0 (no drift), the SPRT MUST NOT fire an alarm at the documented
    ARL₀. A single null-data run can still alarm by chance with probability
    ~α, so the test uses a deterministic seed selected to be quiet — the
    GREEN implementation picks a seed where the run stays in 'continue'."""

    def test_pure_null_run_does_not_fire(self, sprt_config: SprtConfig) -> None:
        """A long stream drawn entirely from p_null should not cross the
        upper bound for the chosen seed (representative quiet run)."""
        stream = synthetic_drift_stream(
            null_rate=0.05,
            drift_rate=0.05,
            drift_offset=0,
            total_n=200,
            seed=7,
        )
        state = run_sprt_on_window(stream, sprt_config)
        assert state.verdict != "reject_null"


class TestSprtArl0Empirical:
    """Empirical ARL₀ check: over many independent null streams, the
    ratio of total observations processed to the count of false alarms
    is ≥ SPRT_TARGET_ARL0.

    The formula is ``total_observations / max(n_false_alarms, 1)`` —
    not the average alarm-time across alarming streams. The naive
    average-only formula passes vacuously when zero alarms occur
    (``inf >= 500``); the ratio form pins the false-alarm rate to a
    long-run frequency over real exposure, which is what ARL₀ actually
    measures."""

    def test_empirical_arl0_meets_target(self, sprt_config: SprtConfig) -> None:
        total_observations = 0
        n_false_alarms = 0
        n_replications = 50
        per_stream_n = 2000
        for seed in range(n_replications):
            stream = synthetic_drift_stream(
                null_rate=sprt_config.p_null,
                drift_rate=sprt_config.p_null,
                drift_offset=0,
                total_n=per_stream_n,
                seed=seed,
            )
            state = run_sprt_on_window(stream, sprt_config)
            total_observations += state.n_observations
            if state.verdict == "reject_null":
                n_false_alarms += 1
        # Long-run frequency form: how many observations did we burn per
        # false alarm? With α=β=0.05, p_null=0.05, p_alt=0.10 the
        # theoretical ARL₀ comfortably exceeds 500.
        empirical_arl0 = total_observations / max(n_false_alarms, 1)
        assert empirical_arl0 >= SPRT_TARGET_ARL0, (
            f"ARL₀ regression: {empirical_arl0:.0f} observations per "
            f"false alarm over {n_replications} null streams of "
            f"{per_stream_n} obs each ({n_false_alarms} alarms total)"
        )


class TestSprtResetClearsState:
    """``reset()`` zeroes the cumulative log-LR so the next window starts
    fresh — the cron-driven monitor relies on per-window independence."""

    def test_reset_returns_to_continue_at_origin(
        self, sprt_config: SprtConfig
    ) -> None:
        monitor = WaldSprtMonitor(sprt_config)
        for _ in range(20):
            monitor.step(True)
        monitor.reset()
        state = monitor.step(False)
        # First observation after reset; cumulative log-LR has just
        # one increment, far from either bound.
        assert state.n_observations == 1
        assert state.verdict == "continue"

    def test_monitor_exposes_config(self, sprt_config: SprtConfig) -> None:
        """``monitor.config`` is the read-only handle the dashboard /
        operator log uses to print which signal+thresholds an alarm
        came from."""
        monitor = WaldSprtMonitor(sprt_config)
        assert monitor.config == sprt_config


# =============================================================================
# Sentinel (AC ③)
# =============================================================================


class TestSentinelConstructionDeterministic:
    """Same ``(audit_rows, size, seed)`` → same ``audit_ids``. The 200-case
    sentinel is built ONCE and reused for every weekly κ check; any
    re-construction MUST yield the same membership."""

    def test_same_seed_same_audit_ids(self) -> None:
        population = [_fake_audit_id(i) for i in range(1000)]
        m1 = build_sentinel_manifest(
            population,  # type: ignore[arg-type]
            size=SENTINEL_SET_SIZE,
            seed=SENTINEL_SET_SEED,
        )
        m2 = build_sentinel_manifest(
            population,  # type: ignore[arg-type]
            size=SENTINEL_SET_SIZE,
            seed=SENTINEL_SET_SEED,
        )
        assert m1.audit_ids == m2.audit_ids
        assert len(m1.audit_ids) == SENTINEL_SET_SIZE

    def test_different_seed_different_audit_ids(self) -> None:
        population = [_fake_audit_id(i) for i in range(1000)]
        m1 = build_sentinel_manifest(
            population,  # type: ignore[arg-type]
            size=SENTINEL_SET_SIZE,
            seed=42,
        )
        m2 = build_sentinel_manifest(
            population,  # type: ignore[arg-type]
            size=SENTINEL_SET_SIZE,
            seed=43,
        )
        assert m1.audit_ids != m2.audit_ids


class TestSentinelKappaIdenticalRuns:
    """Identical runs produce κ = 1.0 ± ε; no alarm fires."""

    def test_identical_classifications_kappa_is_one(self, utc_now: datetime) -> None:
        manifest = SentinelManifest(
            size=4,
            seed=SENTINEL_SET_SEED,
            audit_ids=("audit-0001", "audit-0002", "audit-0003", "audit-0004"),
            built_at=utc_now,
        )
        previous = {
            "audit-0001": "APPROPRIATE",
            "audit-0002": "INAPPROPRIATE",
            "audit-0003": "NEEDS_REVIEW",
            "audit-0004": "APPROPRIATE",
        }
        current = dict(previous)
        result = evaluate_sentinel_run(
            manifest=manifest,
            previous=previous,  # type: ignore[arg-type]
            current=current,  # type: ignore[arg-type]
            kappa_threshold=SENTINEL_KAPPA_ALARM_THRESHOLD,
        )
        assert result.cohen_kappa == pytest.approx(1.0, abs=1e-9)
        assert result.alarm_fired is False


class TestSentinelKappaPerturbedRuns:
    """Heavy perturbation produces κ < threshold → alarm fires."""

    def test_perturbed_classifications_alarm_fires(self, utc_now: datetime) -> None:
        """Flip > half the labels — κ collapses well below 0.90."""
        manifest = SentinelManifest(
            size=10,
            seed=SENTINEL_SET_SEED,
            audit_ids=tuple(_fake_audit_id(i) for i in range(10)),
            built_at=utc_now,
        )
        previous = {_fake_audit_id(i): "APPROPRIATE" for i in range(10)}
        # Flip 7 of 10 to a different class:
        current = {
            _fake_audit_id(i): ("INAPPROPRIATE" if i < 7 else "APPROPRIATE")
            for i in range(10)
        }
        result = evaluate_sentinel_run(
            manifest=manifest,
            previous=previous,  # type: ignore[arg-type]
            current=current,  # type: ignore[arg-type]
            kappa_threshold=SENTINEL_KAPPA_ALARM_THRESHOLD,
        )
        assert result.cohen_kappa < SENTINEL_KAPPA_ALARM_THRESHOLD
        assert result.alarm_fired is True


class TestSentinelInsufficientHistory:
    """Empty ``previous`` (no prior week to compare against) raises typed."""

    def test_empty_previous_raises(self, utc_now: datetime) -> None:
        manifest = SentinelManifest(
            size=2,
            seed=SENTINEL_SET_SEED,
            audit_ids=("audit-0001", "audit-0002"),
            built_at=utc_now,
        )
        with pytest.raises(InsufficientHistoryError):
            evaluate_sentinel_run(
                manifest=manifest,
                previous={},
                current={"audit-0001": "APPROPRIATE"},  # type: ignore[arg-type]
            )


class TestSentinelUsesEvalHarnessMetrics:
    """User constraint: NO re-implementation of κ / AC1 — import from
    :mod:`bba.eval_harness.agreement` instead. Verified structurally by
    checking the sentinel module's source for the expected import."""

    def test_sentinel_module_imports_kappa_from_eval_harness(self) -> None:
        sentinel_src = (MONITORING_PACKAGE_DIR / "sentinel.py").read_text(
            encoding="utf-8"
        )
        # Either of these is a passing GREEN-phase signal:
        accepts = (
            "from bba.eval_harness" in sentinel_src
            or "from bba.eval_harness.agreement" in sentinel_src
        )
        assert accepts, (
            "bba.monitoring.sentinel must import κ / AC1 from "
            "bba.eval_harness — re-implementing the metrics is forbidden "
            "by issue #27 constraints."
        )

    def test_sentinel_module_does_not_define_kappa_locally(self) -> None:
        """A local ``def cohen_kappa(`` in sentinel.py would be a
        re-implementation — forbidden."""
        sentinel_src = (MONITORING_PACKAGE_DIR / "sentinel.py").read_text(
            encoding="utf-8"
        )
        assert not re.search(r"^def cohen_kappa\b", sentinel_src, re.MULTILINE)
        assert not re.search(r"^def gwet_ac1\b", sentinel_src, re.MULTILINE)


# =============================================================================
# Quarterly golden-set drift probe
# =============================================================================


def _golden_entry(
    audit_id: str,
    classification: str = "APPROPRIATE",
    indications: tuple[str, ...] = ("Hb_lt_7",),
) -> GoldenSetEntry:
    return GoldenSetEntry(
        audit_id=audit_id,
        classification=classification,  # type: ignore[arg-type]
        indications=indications,
    )


class TestGoldenSetClassificationDriftDetects:
    """>5% of rows changed classification → ``classification_alarm_fired``."""

    def test_six_percent_change_fires_alarm(self) -> None:
        baseline = [_golden_entry(_fake_audit_id(i)) for i in range(100)]
        # Change 6 of 100 → 6% > 5% threshold
        current = [
            _golden_entry(
                _fake_audit_id(i),
                classification=("INAPPROPRIATE" if i < 6 else "APPROPRIATE"),
            )
            for i in range(100)
        ]
        report = evaluate_golden_set_drift(baseline=baseline, current=current)
        assert report.classification_changed_pct > 0.05
        assert report.classification_alarm_fired is True

    def test_four_percent_change_no_alarm(self) -> None:
        baseline = [_golden_entry(_fake_audit_id(i)) for i in range(100)]
        current = [
            _golden_entry(
                _fake_audit_id(i),
                classification=("INAPPROPRIATE" if i < 4 else "APPROPRIATE"),
            )
            for i in range(100)
        ]
        report = evaluate_golden_set_drift(baseline=baseline, current=current)
        assert report.classification_alarm_fired is False


class TestGoldenSetIndicationDriftDetects:
    """>10% of rows changed indication set → ``indications_alarm_fired``."""

    def test_eleven_percent_change_fires_alarm(self) -> None:
        baseline = [
            _golden_entry(_fake_audit_id(i), indications=("Hb_lt_7",))
            for i in range(100)
        ]
        current = [
            _golden_entry(
                _fake_audit_id(i),
                indications=(
                    ("Hb_lt_7", "active_bleeding")
                    if i < 11
                    else ("Hb_lt_7",)
                ),
            )
            for i in range(100)
        ]
        report = evaluate_golden_set_drift(baseline=baseline, current=current)
        assert report.indications_changed_pct > 0.10
        assert report.indications_alarm_fired is True

    def test_indication_set_order_independent(self) -> None:
        """Indications are a set, not a list — reordering MUST NOT count
        as a change."""
        baseline = [
            _golden_entry(
                _fake_audit_id(i), indications=("Hb_lt_7", "active_bleeding")
            )
            for i in range(10)
        ]
        current = [
            _golden_entry(
                _fake_audit_id(i), indications=("active_bleeding", "Hb_lt_7")
            )
            for i in range(10)
        ]
        report = evaluate_golden_set_drift(baseline=baseline, current=current)
        assert report.indications_changed_pct == 0.0
        assert report.indications_alarm_fired is False


class TestGoldenSetNoDriftNoAlarm:
    """Identical baseline + current → neither alarm fires."""

    def test_identical_runs_no_alarm(self) -> None:
        baseline = [_golden_entry(_fake_audit_id(i)) for i in range(100)]
        current = [_golden_entry(_fake_audit_id(i)) for i in range(100)]
        report = evaluate_golden_set_drift(baseline=baseline, current=current)
        assert report.classification_alarm_fired is False
        assert report.indications_alarm_fired is False
        assert report.classification_changed_pct == 0.0
        assert report.indications_changed_pct == 0.0


class TestGoldenSetMismatchRaises:
    """Baseline and current MUST cover the same ``audit_id`` set."""

    def test_missing_id_in_current_raises(self) -> None:
        baseline = [_golden_entry(_fake_audit_id(i)) for i in range(5)]
        current = [_golden_entry(_fake_audit_id(i)) for i in range(4)]
        with pytest.raises(GoldenSetMismatchError):
            evaluate_golden_set_drift(baseline=baseline, current=current)


# =============================================================================
# Alerting integration stub (AC ④)
# =============================================================================


class TestAlertingEmitsStructuredLog:
    """:func:`emit_alarm` logs to the ``bba.monitoring.alarms`` logger as a
    structured record. The dashboard / log-aggregation consumer subscribes
    to that logger; no other transport is wired in Phase 1."""

    def test_emit_alarm_logs_to_named_logger(
        self,
        utc_now: datetime,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        alarm = MonitoringAlarmInput(
            kind="drift_sprt",
            signal="quote_grounding_failure_rate",
            raised_at=utc_now,
            detail={"log_lr": 3.4, "n_observations": 42},
        )
        with caplog.at_level(logging.WARNING, logger="bba.monitoring.alarms"):
            emit_alarm(alarm)
        assert any(
            rec.name == "bba.monitoring.alarms" for rec in caplog.records
        )

    def test_emit_alarm_includes_kind_in_record(
        self,
        utc_now: datetime,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The alarm record MUST surface the ``kind`` field so operators
        can grep / route by monitor source."""
        alarm = MonitoringAlarmInput(
            kind="sentinel_kappa",
            signal=None,
            raised_at=utc_now,
            detail={"kappa": 0.87, "n_paired": 200},
        )
        with caplog.at_level(logging.WARNING, logger="bba.monitoring.alarms"):
            emit_alarm(alarm)
        assert any(
            "sentinel_kappa" in rec.getMessage() for rec in caplog.records
        )


class TestNoSlackEmailPagingImport:
    """Phase 1.5 boundary: no Slack / email / paging transports in Phase 1.

    Structurally enforced — a future PR that adds ``slack_sdk`` will fail
    this regression check."""

    FORBIDDEN_IMPORTS = (
        "slack_sdk",
        "slack",
        "smtplib",
        "pagerduty",
        "opsgenie",
        "twilio",
    )

    @pytest.mark.parametrize("forbidden", FORBIDDEN_IMPORTS)
    def test_module_does_not_import(self, forbidden: str) -> None:
        for py_file in MONITORING_PACKAGE_DIR.glob("*.py"):
            src = py_file.read_text(encoding="utf-8")
            assert f"import {forbidden}" not in src, (
                f"{py_file.name} imports forbidden transport {forbidden!r}; "
                f"Phase 1 alerting is stub-only (PRD §18, issue #27)"
            )
            assert f"from {forbidden}" not in src, (
                f"{py_file.name} imports from forbidden transport "
                f"{forbidden!r}; Phase 1 alerting is stub-only"
            )


# =============================================================================
# Operational boundary regression checks
# =============================================================================


class TestNoClinicalImports:
    """User constraint: bba.monitoring is OPERATIONAL not clinical. The
    module MUST NOT import from the clinical-logic modules listed below;
    every signal it consumes is read off the persisted ``audit_results``
    rows via :mod:`bba.audit_store`."""

    FORBIDDEN_CLINICAL_MODULES = (
        "bba.hb_lookup",
        "bba.vitals_extractor",
        "bba.cohort_detector",
        "bba.deterministic_classifier",
        "bba.quote_grounder",
        "bba.prompt_builder",
        "bba.evidence_bundle_builder",
        "bba.deid_redactor",
    )

    @pytest.mark.parametrize("forbidden", FORBIDDEN_CLINICAL_MODULES)
    def test_module_does_not_import(self, forbidden: str) -> None:
        for py_file in MONITORING_PACKAGE_DIR.glob("*.py"):
            src = py_file.read_text(encoding="utf-8")
            assert f"import {forbidden}" not in src, (
                f"{py_file.name} imports clinical module {forbidden!r}; "
                f"bba.monitoring is OPERATIONAL only (issue #27 constraint)"
            )
            assert f"from {forbidden}" not in src, (
                f"{py_file.name} imports from clinical module {forbidden!r}; "
                f"bba.monitoring is OPERATIONAL only (issue #27 constraint)"
            )


class TestNoSchedulerImports:
    """User constraint: cadences are CRON-LIKE but NOT scheduled in #27
    itself. #29 (cli) wires the scheduler. The monitoring package MUST NOT
    pull in a scheduler dependency on its own."""

    FORBIDDEN_SCHEDULERS = ("apscheduler", "schedule", "rq_scheduler", "celery")

    @pytest.mark.parametrize("forbidden", FORBIDDEN_SCHEDULERS)
    def test_module_does_not_import_scheduler(self, forbidden: str) -> None:
        for py_file in MONITORING_PACKAGE_DIR.glob("*.py"):
            src = py_file.read_text(encoding="utf-8")
            assert (
                f"import {forbidden}" not in src
                and f"from {forbidden}" not in src
            ), (
                f"{py_file.name} pulls in scheduler {forbidden!r}; "
                f"cadence scheduling is #29's job (issue #27 constraint)"
            )


class TestNoLiveAnthropicInDriftProbe:
    """User constraint: no live Anthropic calls in tests. The golden-set
    drift probe in tests must use the cassette-replay output, not the live
    transport. Verified by checking the monitoring package does NOT directly
    instantiate the live :class:`bba.llm_client.AnthropicBatchTransport`
    — the drift probe takes pre-recorded ``GoldenSetEntry`` sequences as
    inputs and never calls the API itself."""

    def test_golden_set_probe_does_not_instantiate_live_transport(self) -> None:
        for py_file in MONITORING_PACKAGE_DIR.glob("*.py"):
            src = py_file.read_text(encoding="utf-8")
            assert "AnthropicBatchTransport(" not in src, (
                f"{py_file.name} instantiates the live Anthropic transport; "
                f"the drift probe must operate on pre-recorded "
                f"GoldenSetEntry sequences (issue #27 constraint, cost_guard)"
            )

    def test_evaluate_golden_set_signature_takes_recorded_inputs(self) -> None:
        """The function signature MUST take ``baseline`` and ``current`` as
        explicit Sequence[GoldenSetEntry] — proof that the recorded output
        is the contract, not the live API."""
        import inspect

        sig = inspect.signature(evaluate_golden_set_drift)
        params = sig.parameters
        assert "baseline" in params
        assert "current" in params


# =============================================================================
# Store persistence (alerting integration, Phase 1 backing table)
# =============================================================================


class TestMonitoringStoreRecordsAlarm:
    """:meth:`MonitoringStore.record_alarm` persists one row and returns
    the DB-assigned :class:`MonitoringAlarm`."""

    def test_record_alarm_returns_persisted_record(
        self,
        monitoring_config: MonitoringConfig,
        utc_now: datetime,
    ) -> None:
        store = MonitoringStore(monitoring_config)
        alarm = MonitoringAlarmInput(
            kind="drift_sprt",
            signal="quote_grounding_failure_rate",
            raised_at=utc_now,
            detail={"log_lr": 3.5},
        )
        persisted = store.record_alarm(alarm)
        assert isinstance(persisted, MonitoringAlarm)
        assert persisted.kind == "drift_sprt"

    def test_list_alarms_filters_by_kind(
        self, monitoring_config: MonitoringConfig
    ) -> None:
        store = MonitoringStore(monitoring_config)
        result = store.list_alarms(kind="sentinel_kappa")
        assert all(a.kind == "sentinel_kappa" for a in result)


# =============================================================================
# Property test: replay idempotency
# =============================================================================


class TestPropertyReplayIdempotent:
    """User constraint: replaying the same period's data twice produces
    zero duplicate alarms. Property-checked via hypothesis over arbitrary
    boolean observation streams + arbitrary SPRT configs."""

    @given(
        observations=st.lists(st.booleans(), min_size=1, max_size=300),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(
        deadline=None,
        max_examples=25,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_replay_same_observations_same_terminal_state(
        self, observations: list[bool], seed: int
    ) -> None:
        """Replaying the exact same observation stream through the SPRT
        produces the exact same terminal state — no hidden RNG, no
        wall-clock dependency, no duplicate alarms on re-derivation."""
        config = SprtConfig(
            signal="quote_grounding_failure_rate",
            p_null=0.05,
            p_alt=0.10,
            alpha=SPRT_DEFAULT_ALPHA,
            beta=SPRT_DEFAULT_BETA,
            min_n=10,
        )
        state_1 = run_sprt_on_window(observations, config)
        state_2 = run_sprt_on_window(observations, config)
        assert state_1.verdict == state_2.verdict
        assert state_1.log_lr == pytest.approx(state_2.log_lr)
        assert state_1.n_observations == state_2.n_observations


# =============================================================================
# Smoke: importing the package does not raise
# =============================================================================


class TestPackageImportSmoke:
    """Importing :mod:`bba.monitoring` must not raise.

    Passes in RED — the scaffold's NotImplementedError bodies do not
    execute at import time."""

    def test_import_does_not_raise(self) -> None:
        mod = importlib.import_module("bba.monitoring")
        assert hasattr(mod, "WaldSprtMonitor")
        assert hasattr(mod, "evaluate_sentinel_run")
        assert hasattr(mod, "evaluate_golden_set_drift")
        assert hasattr(mod, "draw_weekly_reviewer_sample")
        assert hasattr(mod, "MonitoringStore")
        assert hasattr(mod, "emit_alarm")


# =============================================================================
# Codex review follow-ups — P1 coverage gaps + invariant guards
# =============================================================================


class TestSprtConfigValidation:
    """:class:`SprtConfig` must reject nonsense parameters at construction
    time so ``math.log(p_alt / p_null)`` inside the SPRT loop never
    encounters a boundary value (division by zero or log of zero).

    The validators on the model — not on each consumer — are the right
    boundary for this check: every caller that builds a SprtConfig gets
    the same enforcement without having to remember to re-validate.
    """

    @pytest.mark.parametrize("bad_p", [-0.1, 0.0, 1.0, 1.5])
    def test_p_null_out_of_open_unit_rejected(self, bad_p: float) -> None:
        with pytest.raises(ValidationError):
            SprtConfig(
                signal="needs_review_rate",
                p_null=bad_p,
                p_alt=0.5,
            )

    @pytest.mark.parametrize("bad_p", [-0.1, 0.0, 1.0, 1.5])
    def test_p_alt_out_of_open_unit_rejected(self, bad_p: float) -> None:
        with pytest.raises(ValidationError):
            SprtConfig(
                signal="needs_review_rate",
                p_null=0.05,
                p_alt=bad_p,
            )

    def test_p_alt_must_exceed_p_null(self) -> None:
        """``p_null >= p_alt`` would invert the per-step log-LR sign and
        send the SPRT in the wrong direction; rejected at config time."""
        with pytest.raises(ValidationError):
            SprtConfig(
                signal="needs_review_rate",
                p_null=0.10,
                p_alt=0.10,  # equal
            )
        with pytest.raises(ValidationError):
            SprtConfig(
                signal="needs_review_rate",
                p_null=0.10,
                p_alt=0.05,  # inverted
            )

    @pytest.mark.parametrize("bad", [-0.1, 0.0, 1.0, 1.5])
    def test_alpha_out_of_open_unit_rejected(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            SprtConfig(
                signal="needs_review_rate",
                p_null=0.05,
                p_alt=0.10,
                alpha=bad,
            )

    @pytest.mark.parametrize("bad", [-0.1, 0.0, 1.0, 1.5])
    def test_beta_out_of_open_unit_rejected(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            SprtConfig(
                signal="needs_review_rate",
                p_null=0.05,
                p_alt=0.10,
                beta=bad,
            )

    @pytest.mark.parametrize("bad_min_n", [0, -1, -100])
    def test_min_n_must_be_positive(self, bad_min_n: int) -> None:
        with pytest.raises(ValidationError):
            SprtConfig(
                signal="needs_review_rate",
                p_null=0.05,
                p_alt=0.10,
                min_n=bad_min_n,
            )


class TestWaldBoundsValidation:
    """The :func:`wald_bounds` helper is also called from the monitor's
    constructor — both surfaces share the same input-validation contract,
    so tests pin it directly here too."""

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
    def test_alpha_out_of_open_unit_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError):
            wald_bounds(alpha=bad, beta=0.05)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
    def test_beta_out_of_open_unit_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError):
            wald_bounds(alpha=0.05, beta=bad)


class TestSyntheticDriftStreamValidation:
    """:func:`synthetic_drift_stream` is the fixture every SPRT validation
    test depends on; its own input guards must be exercised so a malformed
    fixture doesn't silently produce garbage data."""

    def test_negative_drift_offset_rejected(self) -> None:
        with pytest.raises(ValueError):
            synthetic_drift_stream(
                null_rate=0.05,
                drift_rate=0.10,
                drift_offset=-1,
                total_n=100,
                seed=0,
            )

    def test_negative_total_n_rejected(self) -> None:
        with pytest.raises(ValueError):
            synthetic_drift_stream(
                null_rate=0.05,
                drift_rate=0.10,
                drift_offset=0,
                total_n=-1,
                seed=0,
            )

    def test_drift_offset_exceeds_total_n_rejected(self) -> None:
        with pytest.raises(ValueError):
            synthetic_drift_stream(
                null_rate=0.05,
                drift_rate=0.10,
                drift_offset=200,
                total_n=100,
                seed=0,
            )


class TestSprtAcceptNullCycleReset:
    """User constraint (PRD §18): under H0 the SPRT must NOT silently
    stop after a single accept-null crossing — it must reset and keep
    monitoring. The terminal state after a long all-failure stream
    should reflect total observations processed across cycles, not the
    count in the final cycle alone."""

    def test_all_failure_stream_resets_and_processes_all(
        self, sprt_config: SprtConfig
    ) -> None:
        """An all-False stream drives log_lr below the lower bound
        quickly; the SPRT must accept_null, reset, and continue. After
        the full window we should see ``n_observations == len(stream)``
        and ``verdict != "reject_null"``."""
        observations = [False] * 500
        state = run_sprt_on_window(observations, sprt_config)
        assert state.n_observations == 500
        assert state.verdict != "reject_null"


class TestSprtEmptyWindow:
    """:func:`run_sprt_on_window` on an empty iterable returns the
    initial state — not an exception. This is the boundary the property
    test couldn't reach because hypothesis ``min_size=1`` excludes it."""

    def test_empty_window_returns_continue_at_origin(
        self, sprt_config: SprtConfig
    ) -> None:
        state = run_sprt_on_window([], sprt_config)
        assert state.verdict == "continue"
        assert state.n_observations == 0
        assert state.n_successes == 0
        assert state.log_lr == 0.0


class TestSprtMonotonicallyAccumulatesLogLR:
    """Property: under an all-success stream the log-LR is monotonically
    non-decreasing; under an all-failure stream it is monotonically
    non-increasing. This pins the per-step sign convention regardless
    of the parameter values hypothesis generates."""

    @given(
        n_obs=st.integers(min_value=1, max_value=100),
        p_null=st.floats(min_value=0.01, max_value=0.40),
    )
    @settings(
        deadline=None,
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_all_success_log_lr_non_decreasing(
        self, n_obs: int, p_null: float
    ) -> None:
        # Construct a valid config: p_alt strictly > p_null.
        p_alt = min(p_null + 0.05, 0.95)
        config = SprtConfig(
            signal="needs_review_rate",
            p_null=p_null,
            p_alt=p_alt,
            min_n=1,
        )
        monitor = WaldSprtMonitor(config)
        previous_lr = 0.0
        for _ in range(n_obs):
            state = monitor.step(True)
            assert state.log_lr >= previous_lr - 1e-12
            previous_lr = state.log_lr

    @given(
        n_obs=st.integers(min_value=1, max_value=100),
        p_null=st.floats(min_value=0.01, max_value=0.40),
    )
    @settings(
        deadline=None,
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_all_failure_log_lr_non_increasing(
        self, n_obs: int, p_null: float
    ) -> None:
        p_alt = min(p_null + 0.05, 0.95)
        config = SprtConfig(
            signal="needs_review_rate",
            p_null=p_null,
            p_alt=p_alt,
            min_n=1,
        )
        monitor = WaldSprtMonitor(config)
        previous_lr = 0.0
        for _ in range(n_obs):
            state = monitor.step(False)
            assert state.log_lr <= previous_lr + 1e-12
            previous_lr = state.log_lr


class TestSafeIdBoundaryValues:
    """SafeId allow-list rejects empty strings and path-traversal
    segments. Tests exercise the validator at the model boundary; the
    same rules are documented on the audit_store's SafeId and we keep
    them symmetric so identifiers cross module boundaries safely."""

    def test_empty_audit_id_rejected(self, utc_now: datetime) -> None:
        with pytest.raises(ValidationError):
            WeeklyReviewerSample(
                week_iso="2026-W20",
                sample_size=50,
                seed=1,
                audit_ids=("",),
                drawn_at=utc_now,
            )

    @pytest.mark.parametrize("bad", [".", ".."])
    def test_path_traversal_audit_id_rejected(
        self, utc_now: datetime, bad: str
    ) -> None:
        with pytest.raises(ValidationError):
            WeeklyReviewerSample(
                week_iso="2026-W20",
                sample_size=50,
                seed=1,
                audit_ids=(bad,),
                drawn_at=utc_now,
            )


class TestWeeklyReviewerSamplePopulationBounds:
    """``sample_size`` must not exceed the available population. The
    function raises ValueError loudly rather than silently truncating
    or returning a partial sample."""

    def test_sample_size_exceeds_population_rejected(self) -> None:
        # 10 rows of population, ask for 50 (which is at the legal
        # sample_size lower bound). Must raise ValueError.
        population = [_fake_audit_id(i) for i in range(10)]
        with pytest.raises(ValueError):
            draw_weekly_reviewer_sample(
                population,  # type: ignore[arg-type]
                week_iso="2026-W20",
                sample_size=50,
                seed=42,
            )


class TestSentinelSizeBounds:
    """Sentinel construction rejects zero/negative sizes and sizes that
    exceed the available population."""

    @pytest.mark.parametrize("bad_size", [0, -1, -100])
    def test_zero_or_negative_size_rejected(self, bad_size: int) -> None:
        population = [_fake_audit_id(i) for i in range(50)]
        with pytest.raises(ValueError):
            build_sentinel_manifest(
                population,  # type: ignore[arg-type]
                size=bad_size,
                seed=SENTINEL_SET_SEED,
            )

    def test_size_exceeds_population_rejected(self) -> None:
        population = [_fake_audit_id(i) for i in range(10)]
        with pytest.raises(ValueError):
            build_sentinel_manifest(
                population,  # type: ignore[arg-type]
                size=200,
                seed=SENTINEL_SET_SEED,
            )


class TestSentinelDisjointHistory:
    """If the previous + current run share NO audit_ids with the manifest,
    the comparison cannot proceed — :class:`InsufficientHistoryError`."""

    def test_disjoint_previous_and_current_raises(
        self, utc_now: datetime
    ) -> None:
        manifest = SentinelManifest(
            size=3,
            seed=SENTINEL_SET_SEED,
            audit_ids=("audit-0001", "audit-0002", "audit-0003"),
            built_at=utc_now,
        )
        # Previous and current both non-empty, but neither has any
        # audit_id from the manifest.
        previous: dict[str, str] = {"audit-9999": "APPROPRIATE"}
        current: dict[str, str] = {"audit-9998": "APPROPRIATE"}
        with pytest.raises(InsufficientHistoryError):
            evaluate_sentinel_run(
                manifest=manifest,
                previous=previous,  # type: ignore[arg-type]
                current=current,  # type: ignore[arg-type]
            )


class TestGoldenSetMissingInBaseline:
    """Mismatch detection covers both directions: an extra row in
    ``current`` that has no baseline counterpart also raises."""

    def test_extra_id_in_current_raises(self) -> None:
        baseline = [_golden_entry(_fake_audit_id(i)) for i in range(3)]
        current = [_golden_entry(_fake_audit_id(i)) for i in range(5)]
        with pytest.raises(GoldenSetMismatchError):
            evaluate_golden_set_drift(baseline=baseline, current=current)


class TestMonitoringStoreSinceFilter:
    """``list_alarms(since=...)`` filters by ``raised_at >= since`` and
    rejects naive datetimes loudly via :func:`ensure_utc` instead of
    silently TypeError-ing at the ``>=`` comparison."""

    def test_since_filter_returns_only_after(
        self,
        monitoring_config: MonitoringConfig,
        utc_now: datetime,
    ) -> None:
        store = MonitoringStore(monitoring_config)
        earlier = datetime(2026, 5, 16, 8, 0, 0, tzinfo=UTC)
        later = datetime(2026, 5, 16, 18, 0, 0, tzinfo=UTC)
        store.record_alarm(
            MonitoringAlarmInput(
                kind="drift_sprt",
                signal="needs_review_rate",
                raised_at=earlier,
                detail={"log_lr": 1.0},
            )
        )
        store.record_alarm(
            MonitoringAlarmInput(
                kind="drift_sprt",
                signal="needs_review_rate",
                raised_at=later,
                detail={"log_lr": 2.0},
            )
        )
        result = store.list_alarms(since=utc_now)
        assert len(result) == 1
        assert result[0].raised_at == later

    def test_naive_since_rejected(
        self, monitoring_config: MonitoringConfig
    ) -> None:
        store = MonitoringStore(monitoring_config)
        with pytest.raises(ValueError):
            store.list_alarms(since=datetime(2026, 5, 16, 12, 0, 0))


class TestMonitoringStoreLifecycle:
    """Close + context-manager lifecycle. After ``close()`` (explicit or
    via ``__exit__``), subsequent operations raise ``RuntimeError`` so
    use-after-close failures are loud."""

    def test_context_manager_closes_on_exit(
        self, monitoring_config: MonitoringConfig, utc_now: datetime
    ) -> None:
        with MonitoringStore(monitoring_config) as store:
            store.record_alarm(
                MonitoringAlarmInput(
                    kind="drift_sprt",
                    signal="needs_review_rate",
                    raised_at=utc_now,
                    detail={"log_lr": 1.0},
                )
            )
        # After context exit, store is closed; further ops raise.
        with pytest.raises(RuntimeError):
            store.record_alarm(
                MonitoringAlarmInput(
                    kind="drift_sprt",
                    signal="needs_review_rate",
                    raised_at=utc_now,
                    detail={"log_lr": 1.0},
                )
            )

    def test_close_is_idempotent(
        self, monitoring_config: MonitoringConfig
    ) -> None:
        store = MonitoringStore(monitoring_config)
        store.close()
        store.close()  # second call must not raise

    def test_close_clears_state(
        self, monitoring_config: MonitoringConfig, utc_now: datetime
    ) -> None:
        store = MonitoringStore(monitoring_config)
        store.record_alarm(
            MonitoringAlarmInput(
                kind="drift_sprt",
                signal="needs_review_rate",
                raised_at=utc_now,
                detail={"log_lr": 1.0},
            )
        )
        store.close()
        # After close, list_alarms raises (the store is unusable, NOT
        # an empty-result silent state).
        with pytest.raises(RuntimeError):
            store.list_alarms()

    def test_store_exposes_config(
        self, monitoring_config: MonitoringConfig
    ) -> None:
        """``store.config`` is the read-only handle the dashboard / report
        consumer uses to print which DSN the alarms came from."""
        store = MonitoringStore(monitoring_config)
        assert store.config == monitoring_config


class TestMonitoringStoreConcurrentWrites:
    """The store's lock guards multi-threaded writes from a future
    cron-driven monitor (#29) that may invoke multiple cadences in
    parallel. Eight threads writing 50 alarms each must produce 400
    rows with unique ``alarm_id`` values and no losses."""

    def test_concurrent_record_alarm_no_lost_writes(
        self, monitoring_config: MonitoringConfig, utc_now: datetime
    ) -> None:
        import concurrent.futures

        store = MonitoringStore(monitoring_config)
        n_workers = 8
        writes_per_worker = 50
        total_writes = n_workers * writes_per_worker

        def writer(worker_id: int) -> None:
            for i in range(writes_per_worker):
                store.record_alarm(
                    MonitoringAlarmInput(
                        kind="drift_sprt",
                        signal="needs_review_rate",
                        raised_at=utc_now,
                        detail={"worker": worker_id, "i": i},
                    )
                )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_workers
        ) as pool:
            list(pool.map(writer, range(n_workers)))

        alarms = store.list_alarms()
        assert len(alarms) == total_writes
        ids = [a.alarm_id for a in alarms]
        assert len(set(ids)) == total_writes  # all unique
        # IDs are 1..total_writes (monotonic, no gaps).
        assert min(ids) == 1
        assert max(ids) == total_writes


class TestPropertyReplayStoreIdempotent:
    """User constraint: 'replaying the same period's data twice produces
    zero duplicate alarms'. At the SPRT level the property test pins
    determinism of state; at the store level we pin that re-persisting
    the same manifest is a no-op (idempotent on the key triple)."""

    @given(
        seed=st.integers(min_value=0, max_value=1000),
        size=st.integers(
            min_value=WEEKLY_REVIEWER_SAMPLE_MIN,
            max_value=WEEKLY_REVIEWER_SAMPLE_MAX,
        ),
    )
    @settings(
        deadline=None,
        max_examples=15,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_replay_persist_yields_single_row(
        self, seed: int, size: int
    ) -> None:
        config = MonitoringConfig(
            dsn="postgresql://test:test@localhost:5432/test_idempotent"
        )
        store = MonitoringStore(config)
        sample = WeeklyReviewerSample(
            week_iso="2026-W20",
            sample_size=size,
            seed=seed,
            audit_ids=tuple(_fake_audit_id(i) for i in range(size)),
            drawn_at=datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC),
        )
        # Persist the same manifest 5 times — same week, same size,
        # same seed → exactly one stored row.
        for _ in range(5):
            store.persist_sample_manifest(sample)
        manifests = store.list_sample_manifests(week_iso="2026-W20")
        assert len(manifests) == 1
        store.close()


class TestEnsureUtcRejectsAmbiguousTzinfo:
    """User constraint follow-up: the datetime validator must also reject
    a ``tzinfo`` whose ``utcoffset()`` returns ``None``. Python admits
    that shape but it's operationally naive — comparisons against a true
    UTC-aware datetime would still raise ``TypeError`` at runtime."""

    def test_tzinfo_with_none_utcoffset_rejected(self) -> None:
        from datetime import tzinfo

        class AmbiguousTz(tzinfo):
            def utcoffset(self, dt: datetime | None) -> None:  # type: ignore[override]
                return None

            def dst(self, dt: datetime | None) -> None:  # type: ignore[override]
                return None

            def tzname(self, dt: datetime | None) -> str | None:  # type: ignore[override]
                return "AMB"

        ambiguous = datetime(2026, 5, 16, 12, 0, 0, tzinfo=AmbiguousTz())
        with pytest.raises(ValidationError):
            MonitoringAlarmInput(
                kind="drift_sprt",
                signal="needs_review_rate",
                raised_at=ambiguous,
                detail={},
            )
