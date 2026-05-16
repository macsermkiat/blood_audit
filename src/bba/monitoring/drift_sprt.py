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
``E[cycle length under H0] / alpha``. With ``alpha=0.05, beta=0.05,
p_null=0.05, p_alt=0.10`` and the reset-on-accept_null convention the
ARL₀ comes out well above the PRD's ≥500 target.

scipy.stats has no direct SPRT helper; the implementation is short enough
to live here without adding a dependency.

Operational, not clinical: this module reads boolean signals off
audit rows and never re-derives the underlying classifications. Re-using
:mod:`bba.eval_harness` is unnecessary here — the SPRT is its own
statistical test, not an agreement coefficient.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable

from bba.monitoring.models import SprtConfig, SprtState, SprtVerdict


class WaldSprtMonitor:
    """Per-signal incremental SPRT monitor.

    Construct once with a :class:`SprtConfig`; feed booleans via
    :meth:`step`. Each step returns the current :class:`SprtState`. When
    ``state.verdict != "continue"``, the caller SHOULD raise the
    corresponding alarm (only on ``"reject_null"``) and call
    :meth:`reset` before the next window.

    Constructed with a frozen config; the per-step log-LR increments are
    pre-computed once so the hot loop is one add, one compare, one tuple
    allocation per observation.
    """

    def __init__(self, config: SprtConfig) -> None:
        self._config = config
        self._lower, self._upper = wald_bounds(
            alpha=config.alpha, beta=config.beta
        )
        # Pre-compute per-observation log-LR increments so :meth:`step`
        # avoids a `math.log` call on every observation in the hot loop.
        self._step_success = math.log(config.p_alt / config.p_null)
        self._step_failure = math.log(
            (1.0 - config.p_alt) / (1.0 - config.p_null)
        )
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
        verdict is recomputed. The verdict stays at ``"continue"`` until
        ``n_observations >= config.min_n`` — early observations are too
        noisy for a stable verdict and the min-N gate prevents
        single-observation alarms.
        """
        self._n_observations += 1
        if observation:
            self._n_successes += 1
            self._log_lr += self._step_success
        else:
            self._log_lr += self._step_failure

        verdict: SprtVerdict
        if self._n_observations < self._config.min_n:
            verdict = "continue"
        elif self._log_lr >= self._upper:
            verdict = "reject_null"
        elif self._log_lr <= self._lower:
            verdict = "accept_null"
        else:
            verdict = "continue"

        return SprtState(
            signal=self._config.signal,
            n_observations=self._n_observations,
            n_successes=self._n_successes,
            log_lr=self._log_lr,
            upper_bound=self._upper,
            lower_bound=self._lower,
            verdict=verdict,
        )

    def reset(self) -> None:
        """Reset the cumulative log-LR + observation counters to zero.

        After ``reset()``, the next :meth:`step` produces a state with
        ``n_observations == 1`` and ``log_lr`` equal to one step's
        increment. The Wald bounds are unchanged (they depend only on
        the config's α / β).
        """
        self._n_observations = 0
        self._n_successes = 0
        self._log_lr = 0.0


def wald_bounds(*, alpha: float, beta: float) -> tuple[float, float]:
    """Return ``(lower, upper)`` Wald bounds in log-LR space.

    ``upper = log((1 - beta) / alpha)`` is the H0-rejection boundary;
    ``lower = log(beta / (1 - alpha))`` is the H1-rejection boundary.
    Symmetric in α/β: when ``alpha == beta``, ``upper == -lower``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta!r}")
    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))
    return lower, upper


def run_sprt_on_window(
    observations: Iterable[bool],
    config: SprtConfig,
) -> SprtState:
    """Apply the SPRT to a finite window of observations.

    Cycle management: on ``accept_null`` the internal monitor is reset
    and observation processing continues (a single accept-null crossing
    under H0 is expected; it is not an alarm). On ``reject_null`` the
    function returns immediately with the alarm-time state. At end of
    window, the terminal state is returned with whatever verdict
    happened to be current.

    The returned ``SprtState.n_observations`` is the total number of
    observations consumed (across resets), NOT the count in the current
    cycle — that is the metric the ARL₀ check measures against.
    """
    monitor = WaldSprtMonitor(config)
    total_n = 0
    total_successes = 0
    last_state: SprtState | None = None
    for obs in observations:
        total_n += 1
        if obs:
            total_successes += 1
        state = monitor.step(obs)
        if state.verdict == "reject_null":
            return SprtState(
                signal=config.signal,
                n_observations=total_n,
                n_successes=total_successes,
                log_lr=state.log_lr,
                upper_bound=state.upper_bound,
                lower_bound=state.lower_bound,
                verdict="reject_null",
            )
        if state.verdict == "accept_null":
            monitor.reset()
        last_state = state

    if last_state is None:
        lower, upper = wald_bounds(alpha=config.alpha, beta=config.beta)
        return SprtState(
            signal=config.signal,
            n_observations=0,
            n_successes=0,
            log_lr=0.0,
            upper_bound=upper,
            lower_bound=lower,
            verdict="continue",
        )

    return SprtState(
        signal=config.signal,
        n_observations=total_n,
        n_successes=total_successes,
        log_lr=last_state.log_lr,
        upper_bound=last_state.upper_bound,
        lower_bound=last_state.lower_bound,
        verdict=last_state.verdict,
    )


def synthetic_drift_stream(
    *,
    null_rate: float,
    drift_rate: float,
    drift_offset: int,
    total_n: int,
    seed: int,
) -> tuple[bool, ...]:
    """Generate a synthetic boolean stream for SPRT validation tests.

    Emits ``drift_offset`` Bernoulli(``null_rate``) draws followed by
    ``total_n - drift_offset`` Bernoulli(``drift_rate``) draws, with a
    deterministic RNG seeded by ``seed``. Used by the test suite's
    synthetic-drift fixture; lives in the production module so the
    fixture stays a thin wrapper.
    """
    if drift_offset < 0:
        raise ValueError(f"drift_offset must be >= 0, got {drift_offset!r}")
    if total_n < 0:
        raise ValueError(f"total_n must be >= 0, got {total_n!r}")
    if drift_offset > total_n:
        raise ValueError(
            f"drift_offset ({drift_offset}) must be <= total_n ({total_n})"
        )
    rng = random.Random(seed)
    out: list[bool] = []
    for i in range(total_n):
        rate = null_rate if i < drift_offset else drift_rate
        out.append(rng.random() < rate)
    return tuple(out)


__all__ = (
    "WaldSprtMonitor",
    "run_sprt_on_window",
    "synthetic_drift_stream",
    "wald_bounds",
)
