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

from bba.eval_harness.models import ClassifierMetrics
from bba.quote_grounder.models import ConfusionMatrix


def evaluate_confusion_matrix(
    cm: ConfusionMatrix, *, confidence: float = 0.95
) -> ClassifierMetrics:
    """Convert a 2x2 ConfusionMatrix into classifier metrics with Wilson CIs.

    Edge cases the contract pins:

    * ``cm.true_positive + cm.false_negative == 0`` (no positive examples):
      ``sensitivity`` and ``f1`` are reported as point ``0.0`` with a
      Wilson CI degenerated at ``[0, 0]``. No ZeroDivisionError.
    * Similarly ``cm.true_negative + cm.false_positive == 0`` collapses
      specificity to ``[0, 0]``; ``cm.true_positive + cm.false_positive == 0``
      collapses PPV; ``cm.true_negative + cm.false_negative == 0`` collapses
      NPV.
    * All-zero matrix: every rate is ``[0, 0]``, ``f1 = 0.0``. No exception.

    Loud failure is reserved for *structural* problems (e.g., a negative
    count). The graceful zero-rate handling makes a fresh-pipeline run
    against an empty labeled set produce a valid (if uninformative) report.
    """
    raise NotImplementedError("eval_harness.classifier: RED phase, see issue #20")
