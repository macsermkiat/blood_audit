"""Inter-rater agreement: Cohen's κ, Gwet's AC1, PABAK (PRD §11).

The three coefficients are reported *together* on purpose. κ is the
publication-default but suffers from the kappa-prevalence paradox: in a
stratum where one class dominates (PRD §11 names Hb>10 as the high-
prevalence APPROPRIATE-majority case), κ shrinks toward zero even with very
high observed agreement, because the chance-agreement baseline ``p_e``
inflates with the marginal-prevalence product.

AC1 (Gwet 2008) replaces the chance baseline with a prevalence-resistant
formulation. PABAK (Byrt 1993) collapses to ``2*p_o - 1`` for binary, which
is the maximally prevalence-resistant linear rescaling of observed
agreement. Together the three give a regulator-grade picture: when κ and
AC1 diverge, the divergence is the prevalence signal, not a coding bug.

References (manual): Cohen (1960), Gwet (2008), Byrt (1993). Test
validation uses published worked examples; see ``test_eval_harness.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.eval_harness.models import AgreementResult


def cohen_kappa(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    """Cohen's kappa for two raters over a categorical label set.

    Both sequences must be the same length; otherwise
    :class:`bba.eval_harness.exceptions.ShapeMismatchError` is raised. The
    label set is the union of labels appearing in either sequence; categories
    absent from both raters do not contribute to the chance term.
    """
    raise NotImplementedError("eval_harness.agreement: RED phase, see issue #20")


def gwet_ac1(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    """Gwet's AC1 prevalence-resistant agreement coefficient."""
    raise NotImplementedError("eval_harness.agreement: RED phase, see issue #20")


def pabak(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    """Prevalence-and-bias-adjusted kappa (Byrt 1993).

    For binary classification: PABAK = ``2 * observed_agreement - 1``. For
    k-category classification: ``(k * observed_agreement - 1) / (k - 1)``.
    """
    raise NotImplementedError("eval_harness.agreement: RED phase, see issue #20")


def agreement_with_metrics(
    rater_a: Sequence[str], rater_b: Sequence[str]
) -> AgreementResult:
    """Compute κ + AC1 + PABAK in one pass; raises on shape mismatch.

    Returning the bundle (rather than three free functions) is the public
    contract — the eval-harness report writes all three numbers together,
    and decoupling them would invite report-side divergence (e.g., κ on the
    full set, AC1 on a sliced subset).
    """
    raise NotImplementedError("eval_harness.agreement: RED phase, see issue #20")
