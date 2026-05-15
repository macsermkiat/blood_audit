"""Inter-rater agreement: Cohen's κ, Gwet's AC1, PABAK (PRD §11).

The three coefficients are reported *together* on purpose. κ is the
publication-default but suffers from the kappa-prevalence paradox: in a
stratum where one class dominates (PRD §11 names Hb>10 as the high-
prevalence APPROPRIATE-majority case), κ shrinks toward zero even with very
high observed agreement, because the chance-agreement baseline ``p_e``
inflates with the marginal-prevalence product.

AC1 (Gwet 2008) replaces the chance baseline with a prevalence-resistant
formulation ``p_e = (k/(k-1)) * sum_i pi_i*(1-pi_i)`` where ``pi_i`` is
the averaged marginal probability of category ``i``. PABAK (Byrt 1993)
collapses to ``2*p_o - 1`` for binary and to ``(k*p_o - 1)/(k-1)`` for
k-category, the maximally prevalence-resistant linear rescaling of
observed agreement. Together the three give a regulator-grade picture:
when κ and AC1 diverge, the divergence is the prevalence signal, not a
coding bug.

References (manual): Cohen (1960), Gwet (2008), Byrt (1993). Test
validation uses published worked examples; see ``test_eval_harness.py``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from bba.eval_harness.exceptions import EmptyInputError, ShapeMismatchError
from bba.eval_harness.models import AgreementResult


def _check_sequences(
    a: Sequence[str], b: Sequence[str]
) -> tuple[list[str], list[str]]:
    if len(a) != len(b):
        raise ShapeMismatchError(
            f"agreement: rater sequences differ in length: {len(a)} vs {len(b)}"
        )
    if not a:
        raise EmptyInputError("agreement: rater sequences are empty")
    return list(a), list(b)


def _observed_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


def _category_marginals(
    a: Sequence[str], b: Sequence[str]
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    categories = sorted(set(a) | set(b))
    n = len(a)
    counts_a = Counter(a)
    counts_b = Counter(b)
    pi_a = {c: counts_a.get(c, 0) / n for c in categories}
    pi_b = {c: counts_b.get(c, 0) / n for c in categories}
    return categories, pi_a, pi_b


def cohen_kappa(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    """Cohen's kappa for two raters over a categorical label set."""
    a, b = _check_sequences(rater_a, rater_b)
    p_o = _observed_agreement(a, b)
    categories, pi_a, pi_b = _category_marginals(a, b)
    p_e = sum(pi_a[c] * pi_b[c] for c in categories)
    if p_e == 1.0:
        # Perfect chance agreement (all raters always pick one category).
        # κ is undefined; report 1.0 if observed agreement is also perfect,
        # else 0 — the deflated boundary case the report consumer expects.
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def gwet_ac1(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    """Gwet's AC1 prevalence-resistant agreement coefficient."""
    a, b = _check_sequences(rater_a, rater_b)
    p_o = _observed_agreement(a, b)
    categories, pi_a, pi_b = _category_marginals(a, b)
    k = len(categories)
    if k <= 1:
        # All items share the same category — trivial agreement.
        return 1.0 if p_o == 1.0 else 0.0
    # AC1 chance term averages the two rater marginals per category and uses
    # pi*(1-pi)/(k-1) — the prevalence-resistant correction that Gwet (2008)
    # showed is invariant to majority-class inflation.
    p_e = sum(
        ((pi_a[c] + pi_b[c]) / 2.0) * (1.0 - (pi_a[c] + pi_b[c]) / 2.0)
        for c in categories
    ) / (k - 1)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def pabak(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    """Prevalence-and-bias-adjusted kappa (Byrt 1993).

    Binary: ``2 * p_o - 1``. k-category: ``(k * p_o - 1) / (k - 1)``.
    """
    a, b = _check_sequences(rater_a, rater_b)
    p_o = _observed_agreement(a, b)
    categories = set(a) | set(b)
    k = len(categories)
    if k <= 1:
        # All items same category → perfect agreement by construction.
        return 1.0
    return (k * p_o - 1.0) / (k - 1.0)


def agreement_with_metrics(
    rater_a: Sequence[str], rater_b: Sequence[str]
) -> AgreementResult:
    """Compute κ + AC1 + PABAK in one pass; raises on shape mismatch."""
    a, b = _check_sequences(rater_a, rater_b)
    return AgreementResult(
        cohen_kappa=cohen_kappa(a, b),
        gwet_ac1=gwet_ac1(a, b),
        pabak=pabak(a, b),
        observed_agreement=_observed_agreement(a, b),
        n_items=len(a),
    )
