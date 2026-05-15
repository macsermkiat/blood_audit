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

import random
from collections import defaultdict
from collections.abc import Sequence

from bba.eval_harness.exceptions import InsufficientStratumError
from bba.eval_harness.models import (
    AuditCase,
    SamplingTargets,
    StratifiedSample,
    Stratum,
    StratumDraw,
)


def _partition_by_stratum(
    population: Sequence[AuditCase],
) -> dict[Stratum, list[AuditCase]]:
    by_stratum: dict[Stratum, list[AuditCase]] = defaultdict(list)
    for case in population:
        by_stratum[case.stratum].append(case)
    return by_stratum


def _is_inappropriate(case: AuditCase) -> bool:
    return case.pred_classification == "INAPPROPRIATE"


def _draw_stratum(
    stratum: Stratum,
    cases: list[AuditCase],
    *,
    target: int,
    enrichment_target: int,
    rng: random.Random,
) -> StratumDraw:
    population_size = len(cases)
    positives = [c for c in cases if _is_inappropriate(c)]
    negatives = [c for c in cases if not _is_inappropriate(c)]
    population_positives = len(positives)
    population_negatives = len(negatives)

    if population_size <= target:
        # Whole population fits in the target — take everything. The HT
        # weight degenerates to 1.0 (or treated as the trivial case by the
        # reweighter).
        drawn_positives = population_positives
        chosen = list(positives) + list(negatives)
        rng.shuffle(chosen)
        pos_pi = 1.0 if population_positives > 0 else 0.0
        neg_pi = 1.0 if population_negatives > 0 else 0.0
        return StratumDraw(
            stratum=stratum,
            cases=tuple(chosen),
            population_size=population_size,
            population_positives=population_positives,
            drawn_positives=drawn_positives,
            base_inclusion_probability=neg_pi,
            positive_inclusion_probability=pos_pi,
        )

    # Enrichment target may exceed available positives OR the stratum target
    # itself (e.g., adversarial stratum with target=80 but enrichment=138 —
    # impossible to fit 138 positives into 80 slots). The contract is:
    # drawn_positives = min(enrichment_target, target, available_positives).
    # The remainder of the stratum target is filled from negatives. PRD §11's
    # "~138" wording acknowledges the cap; adversarial is the only sub-138
    # stratum in DEFAULT_STRATUM_TARGETS.
    desired_positives = min(enrichment_target, target, population_positives)
    desired_negatives = target - desired_positives
    if desired_negatives > population_negatives:
        # Not enough negatives even after fully consuming the positive pool.
        # The stratum cannot meet its target — fail loud.
        raise InsufficientStratumError(
            f"stratum {stratum} cannot meet target {target}: "
            f"pop={population_size}, pos={population_positives}, "
            f"neg={population_negatives}, enrichment={enrichment_target}"
        )

    drawn_pos = rng.sample(positives, desired_positives) if desired_positives else []
    drawn_neg = rng.sample(negatives, desired_negatives) if desired_negatives else []
    chosen = drawn_pos + drawn_neg
    rng.shuffle(chosen)

    positive_pi = (
        desired_positives / population_positives if population_positives else 0.0
    )
    base_pi = (
        desired_negatives / population_negatives if population_negatives else 0.0
    )

    return StratumDraw(
        stratum=stratum,
        cases=tuple(chosen),
        population_size=population_size,
        population_positives=population_positives,
        drawn_positives=desired_positives,
        base_inclusion_probability=base_pi,
        positive_inclusion_probability=positive_pi,
    )


def stratified_with_enrichment(
    population: Sequence[AuditCase],
    targets: SamplingTargets,
    *,
    rng_seed: int,
) -> StratifiedSample:
    """Draw a stratified-with-enrichment sample from ``population``."""
    by_stratum = _partition_by_stratum(population)
    # Per-stratum RNG fork so adding a new stratum does not perturb existing
    # strata's draws — the report writer relies on cross-run determinism
    # within a stratum even as the configured target list grows.
    root_rng = random.Random(rng_seed)
    draws: list[StratumDraw] = []
    for stratum, target in targets.per_stratum.items():
        cases = by_stratum.get(stratum, [])
        sub_seed = root_rng.randrange(2**32)
        sub_rng = random.Random(sub_seed)
        draws.append(
            _draw_stratum(
                stratum,
                cases,
                target=target,
                enrichment_target=targets.inappropriate_enrichment_per_stratum,
                rng=sub_rng,
            )
        )
    return StratifiedSample(draws=tuple(draws))
