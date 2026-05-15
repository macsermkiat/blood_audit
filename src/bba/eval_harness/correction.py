"""Multiple-comparisons correction: Bonferroni + BH-FDR (PRD §11).

Per the pre-registration policy, the eval harness corrects two families of
hypothesis tests:

* **Primary** (confirmatory): family-wise-error-controlled via Bonferroni.
  These are the headline claims the publication and the regulator submission
  rely on; any one of them must survive strict ``alpha / k`` to be cited.
* **Exploratory** (hypothesis-generating): false-discovery-rate-controlled
  via Benjamini-Hochberg. These are the per-subgroup sub-claims; FDR
  control lets the report cite the *expected proportion* of false positives
  rather than the family-wise error.

The two families are corrected *independently* — they do NOT pool. Mixing
them would inflate the primary family's chance of any-false-positive while
deflating the exploratory family's discovery rate.

References (manual): Bonferroni (1936); Benjamini & Hochberg (1995).
"""

from __future__ import annotations

from collections.abc import Mapping

from bba.eval_harness.models import (
    CorrectedTest,
    HierarchicalCorrectionResult,
)


def bonferroni_correction(
    p_values: Mapping[str, float], *, alpha: float = 0.05
) -> tuple[CorrectedTest, ...]:
    """Bonferroni multiplicity correction over ``p_values``.

    ``adjusted_p = min(1, raw_p * k)`` where ``k = len(p_values)``;
    ``rejected = adjusted_p < alpha``. Output order matches input iteration
    order — a regression on dict-ordering behavior in the report writer
    would surface as a named test failure.
    """
    raise NotImplementedError("eval_harness.correction: RED phase, see issue #20")


def benjamini_hochberg_correction(
    p_values: Mapping[str, float], *, alpha: float = 0.05
) -> tuple[CorrectedTest, ...]:
    """Benjamini-Hochberg step-up FDR correction.

    Adjusted p-values are computed as the standard BH q-values:
    ``q_i = min_{j >= i} ( m * p_(j) / j )``, capped at 1. ``rejected`` is
    the BH decision (``raw_p_(i) <= (i/m) * alpha`` for some prefix of the
    sorted p-values). Output order matches input iteration order.
    """
    raise NotImplementedError("eval_harness.correction: RED phase, see issue #20")


def hierarchical_correction(
    primary: Mapping[str, float],
    exploratory: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> HierarchicalCorrectionResult:
    """Two-family hierarchical correction (Bonferroni + BH).

    ``primary`` p-values are Bonferroni-corrected; ``exploratory`` p-values
    are BH-FDR-corrected. The two families are returned as separate tuples
    on the result; the caller does not need to know about the implementation
    split between :func:`bonferroni_correction` and
    :func:`benjamini_hochberg_correction`.
    """
    raise NotImplementedError("eval_harness.correction: RED phase, see issue #20")
