"""Verifier-as-classifier evaluation metric.

The 200-row hand-labeled verdict set (PRD §"Acceptance criteria" + issue #18
AC) is consumed by :mod:`bba.eval_harness` to produce a confusion matrix on
the verifier's own decisions: how often does the grounder accept a
genuinely-grounded citation, how often does it falsely accept a hallucinated
one. This module supplies the pure math; the I/O lives in the eval harness.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.quote_grounder.models import ConfusionMatrix, VerdictSequence


def confusion_matrix(
    verdicts: VerdictSequence,
    gold_labels: Sequence[bool],
) -> ConfusionMatrix:
    """Build a 2x2 confusion matrix over (Verdict.passed, gold_labels[i]).

    ``gold_labels[i]`` is ``True`` when the citation is genuinely grounded
    in the source (the human label). ``verdicts[i].passed`` is the
    grounder's prediction. The matrix counts:

    * TP — gold=True, predicted=True (correctly accepted)
    * TN — gold=False, predicted=False (correctly rejected)
    * FP — gold=False, predicted=True (hallucination slipped through)
    * FN — gold=True, predicted=False (over-rejection of valid citation)

    Raises :class:`ValueError` when the two sequences have different
    lengths; silently truncating would corrupt the metric the acceptance
    target ("verifier-as-classifier ≥ ...") is graded against.
    """
    if len(verdicts) != len(gold_labels):
        raise ValueError(
            f"confusion_matrix: verdicts and gold_labels must have equal "
            f"length (got {len(verdicts)} vs {len(gold_labels)})"
        )
    tp = tn = fp = fn = 0
    for verdict, gold in zip(verdicts, gold_labels, strict=True):
        if gold and verdict.passed:
            tp += 1
        elif gold and not verdict.passed:
            fn += 1
        elif not gold and verdict.passed:
            fp += 1
        else:
            tn += 1
    return ConfusionMatrix(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
    )
