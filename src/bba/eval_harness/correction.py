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
    k = len(p_values)
    if k == 0:
        return ()
    out: list[CorrectedTest] = []
    for name, p in p_values.items():
        adjusted = min(1.0, p * k)
        out.append(
            CorrectedTest(
                name=name,
                raw_p=p,
                adjusted_p=adjusted,
                rejected=adjusted < alpha,
            )
        )
    return tuple(out)


def benjamini_hochberg_correction(
    p_values: Mapping[str, float], *, alpha: float = 0.05
) -> tuple[CorrectedTest, ...]:
    """Benjamini-Hochberg step-up FDR correction.

    Adjusted p-values are computed as the standard BH q-values:
    ``q_(i) = min_{j >= i} ( m * p_(j) / j )``, capped at 1. ``rejected`` is
    the BH decision (``raw_p_(i) <= (i/m) * alpha`` for some prefix of the
    sorted p-values). Output order matches input iteration order.
    """
    m = len(p_values)
    if m == 0:
        return ()
    names = list(p_values.keys())
    raws = [p_values[name] for name in names]

    # Order indices by ascending raw p; ties broken by original index for
    # determinism. ``order[k]`` is the original index of the k-th smallest p.
    order = sorted(range(m), key=lambda i: (raws[i], i))

    # BH step-up: q_(i) = min over j >= i of m*p_(j)/j; iterate from largest
    # rank downward and carry the running min forward.
    sorted_ps = [raws[i] for i in order]
    sorted_qs: list[float] = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        i = rank + 1  # 1-indexed rank
        candidate = m * sorted_ps[rank] / i
        running_min = min(running_min, candidate)
        sorted_qs[rank] = min(1.0, running_min)

    # Compute the BH decision: find the largest rank where p_(i) <= (i/m)*alpha;
    # reject every test at rank <= that.
    largest_rejected_rank = -1
    for rank in range(m - 1, -1, -1):
        i = rank + 1
        if sorted_ps[rank] <= (i / m) * alpha:
            largest_rejected_rank = rank
            break
    rejected_orig_idx: set[int] = set()
    if largest_rejected_rank >= 0:
        rejected_orig_idx = {order[r] for r in range(largest_rejected_rank + 1)}

    # Map adjusted-p back to original index order.
    adjusted_by_orig: dict[int, float] = {
        order[rank]: sorted_qs[rank] for rank in range(m)
    }

    return tuple(
        CorrectedTest(
            name=names[i],
            raw_p=raws[i],
            adjusted_p=adjusted_by_orig[i],
            rejected=i in rejected_orig_idx,
        )
        for i in range(m)
    )


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
    return HierarchicalCorrectionResult(
        primary=bonferroni_correction(primary, alpha=alpha),
        exploratory=benjamini_hochberg_correction(exploratory, alpha=alpha),
        alpha=alpha,
    )
