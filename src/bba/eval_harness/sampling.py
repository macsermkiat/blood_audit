"""Stratified-with-enrichment sampling (PRD §11).

For each stratum, draw the target sample size with the constraint that
INAPPROPRIATE predictions are oversampled to ~138 per stratum so per-stratum
sensitivity can be tested at ±0.05 Wilson CI. The remaining slots are filled
by simple random sampling without replacement from the non-positive pool of
the same stratum. Inclusion probabilities are recorded per stratum so a
downstream Horvitz-Thompson reweighting can undo the enrichment when the
metric the report cites is the *population* prevalence.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.eval_harness.models import (
    AuditCase,
    SamplingTargets,
    StratifiedSample,
)


def stratified_with_enrichment(
    population: Sequence[AuditCase],
    targets: SamplingTargets,
    *,
    rng_seed: int,
) -> StratifiedSample:
    """Draw a stratified-with-enrichment sample from ``population``.

    Each stratum is sampled independently to its per-stratum target. The
    enrichment step oversamples ``pred_classification == "INAPPROPRIATE"``
    cases to ``targets.inappropriate_enrichment_per_stratum``; if the
    stratum has fewer positives than the target, ALL positives are drawn
    and the remainder of the stratum's target is filled from negatives.

    The sampler is deterministic given ``rng_seed`` (PRD §"Reproducibility
    = 'we have the original answer'") — re-running with the same seed and
    the same input yields the same draw and the same inclusion probabilities.
    """
    raise NotImplementedError("eval_harness.sampling: RED phase, see issue #20")
