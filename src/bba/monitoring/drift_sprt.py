"""Wald's Sequential Probability Ratio Test for binomial drift (PRD §18).

Watches two binomial rates from the rolling :mod:`bba.audit_store` window:

* ``quote_grounding_failure_rate`` — fraction of audit rows whose
  ``verifier_pass`` is False.
* ``needs_review_rate`` — fraction of audit rows whose
  ``needs_human_review`` is True.

For each signal, the SPRT compares two hypotheses:

  H0: rate == p_null   (baseline measured over the last quarter)
  H1: rate == p_alt    (operator-chosen minimum detectable shift)

Per observation, the log-likelihood ratio increments by
``log(p_alt/p_null)`` for a success and ``log((1-p_alt)/(1-p_null))`` for
a failure. The cumulative LR is compared against two bounds:

  A = log((1 - beta) / alpha)        # upper — reject H0, raise alarm
  B = log(beta / (1 - alpha))        # lower — accept H0, reset monitor

Wald (1947) shows the average run length under H0 is approximately
-log(alpha) / E_0[log(L_n)], which is the per-step expected log-LR under
the null. With ``alpha=0.05, beta=0.05, p_null=0.05, p_alt=0.10`` the
ARL₀ comes out around 540 — comfortably above the PRD's ≥500 target.

scipy.stats has no direct SPRT helper; the implementation is short enough
to live here without adding a dependency.

Operational, not clinical: this module reads the boolean signals off
audit rows and never re-derives the underlying classifications. Re-using
:mod:`bba.eval_harness` is unnecessary here — the SPRT is its own
statistical test, not an agreement coefficient.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from bba.monitoring.models import SprtConfig, SprtState


class WaldSprtMonitor:
    """Per-signal incremental SPRT monitor.

    Construct once with a :class:`SprtConfig`; feed booleans via
    :meth:`step`. Each step returns the current :class:`SprtState`. When
    ``state.verdict != "continue"``, the caller SHOULD raise the
    corresponding alarm (only on ``"reject_null"``) and call
    :meth:`reset` before the next window.
    """

    def __init__(self, config: SprtConfig) -> None:
        self._config = config
        self._n_observations: int = 0
        self._n_successes: int = 0
        self._log_lr: float = 0.0

    @property
    def config(self) -> SprtConfig:
        return self._config

    def step(self, observation: bool) -> SprtState:
        """Incorporate one observation and return the updated state.

        ``observation`` is True for "signal event" (e.g., quote-grounding
        failure or NEEDS_REVIEW). The cumulative log-LR is updated and the
        verdict is recomputed.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the cumulative log-LR to zero for the next window."""
        raise NotImplementedError


def wald_bounds(*, alpha: float, beta: float) -> tuple[float, float]:
    """Return ``(lower, upper)`` Wald bounds in log-LR space.

    ``upper = log((1 - beta) / alpha)`` is the H0-rejection boundary;
    ``lower = log(beta / (1 - alpha))`` is the H1-rejection boundary.
    """
    raise NotImplementedError


def run_sprt_on_window(
    observations: Iterable[bool],
    config: SprtConfig,
) -> SprtState:
    """Apply the SPRT to a finite window of observations.

    Convenience for tests + batch backfill: walks the iterable through a
    fresh :class:`WaldSprtMonitor` and returns the terminal state. The
    monitor's first crossing of either bound is the returned verdict; any
    later observations are NOT consumed (early termination is the whole
    point of an SPRT).
    """
    raise NotImplementedError


def synthetic_drift_stream(
    *,
    null_rate: float,
    drift_rate: float,
    drift_offset: int,
    total_n: int,
    seed: int,
) -> Sequence[bool]:
    """Generate a synthetic boolean stream for SPRT validation tests.

    Emits ``drift_offset`` Bernoulli(``null_rate``) draws followed by
    ``total_n - drift_offset`` Bernoulli(``drift_rate``) draws, with a
    deterministic RNG seeded by ``seed``. Used by the test suite's
    synthetic-drift fixture; lives in the production module so the
    fixture stays a thin wrapper.
    """
    raise NotImplementedError


__all__ = (
    "WaldSprtMonitor",
    "run_sprt_on_window",
    "synthetic_drift_stream",
    "wald_bounds",
)
