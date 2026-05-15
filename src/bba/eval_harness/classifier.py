"""Verifier-as-classifier metrics from a 2x2 confusion matrix (PRD §11).

The quote_grounder (issue #18) produces a
:class:`bba.quote_grounder.ConfusionMatrix` over the 200-row hand-labeled
verifier-evaluation set. This module is the eval-harness consumer: it
turns the 2x2 contingency table into accuracy / sensitivity / specificity /
PPV / NPV / F1, each carrying its Wilson CI at the requested confidence.

The split between "matrix construction" (in quote_grounder) and "metric
computation" (here) is by design: the grounder owns the producer contract
(which counts go in which cell), and the harness owns the inference
contract (what claim the cells support). Re-deriving the matrix here would
duplicate logic and risk drift between the two modules.
"""

from __future__ import annotations

from bba.eval_harness.intervals import wilson_ci
from bba.eval_harness.models import ClassifierMetrics, WilsonInterval
from bba.quote_grounder.models import ConfusionMatrix


def _safe_wilson(successes: int, trials: int, *, confidence: float) -> WilsonInterval:
    """Wilson CI that collapses to a [0, 0] degenerate when trials == 0.

    The all-zero confusion matrix and the no-positive matrix produce
    rates with no denominator. Rather than raising, the harness returns a
    degenerate interval so a fresh-pipeline run against an empty labeled
    set still produces a valid (if uninformative) report row.
    """
    if trials == 0:
        return WilsonInterval(
            point=0.0, lower=0.0, upper=0.0, confidence=confidence
        )
    return wilson_ci(successes, trials, confidence=confidence)


def evaluate_confusion_matrix(
    cm: ConfusionMatrix, *, confidence: float = 0.95
) -> ClassifierMetrics:
    """Convert a 2x2 ConfusionMatrix into classifier metrics with Wilson CIs."""
    tp = cm.true_positive
    tn = cm.true_negative
    fp = cm.false_positive
    fn = cm.false_negative

    pos = tp + fn  # actual positives
    neg = tn + fp  # actual negatives
    pred_pos = tp + fp  # predicted positives
    pred_neg = tn + fn  # predicted negatives
    total = pos + neg

    sensitivity = _safe_wilson(tp, pos, confidence=confidence)
    specificity = _safe_wilson(tn, neg, confidence=confidence)
    ppv = _safe_wilson(tp, pred_pos, confidence=confidence)
    npv = _safe_wilson(tn, pred_neg, confidence=confidence)
    accuracy = _safe_wilson(tp + tn, total, confidence=confidence)

    sens_p = sensitivity.point
    ppv_p = ppv.point
    f1 = (
        (2.0 * ppv_p * sens_p) / (ppv_p + sens_p) if (ppv_p + sens_p) > 0 else 0.0
    )

    return ClassifierMetrics(
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        ppv=ppv,
        npv=npv,
        f1=f1,
    )
