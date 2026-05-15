"""Horvitz-Thompson prevalence reweighting (PRD §11, User Story #28).

Population-level prevalence cannot be read off a stratified-with-enrichment
sample directly: the enrichment step makes the sample non-representative on
purpose. The HT estimator weights each sampled case by the inverse of its
inclusion probability, summed across strata, divided by the known population
size. The variance estimator is the standard HT formula for independent
strata under with-replacement sampling — a slight over-estimate vs the
without-replacement truth, conservative for the publication report.
"""

from __future__ import annotations

from collections.abc import Callable

from bba.eval_harness.models import (
    AuditCase,
    HorvitzThompsonEstimate,
    StratifiedSample,
)


def horvitz_thompson_prevalence(
    sample: StratifiedSample,
    indicator: Callable[[AuditCase], bool],
) -> HorvitzThompsonEstimate:
    """HT estimate of population prevalence under ``indicator``.

    ``indicator`` is a pure function from one :class:`AuditCase` to ``bool``
    — typically ``lambda c: c.pred_classification == "INAPPROPRIATE"`` or the
    same for ``gold_classification``. The estimator is:

    .. math::

       \\hat{p}_{HT} = \\frac{1}{N} \\sum_{h} \\sum_{i \\in s_h}
                       \\frac{y_i}{\\pi_{hi}}

    where ``N`` is the total population size summed across strata, ``s_h``
    is the sample drawn from stratum ``h``, ``y_i`` is ``1`` if the
    indicator holds for case ``i``, ``π_{hi}`` is the case's inclusion
    probability (the per-stratum positive- or base-rate from the sampler).
    """
    raise NotImplementedError("eval_harness.reweight: RED phase, see issue #20")
