"""Outcome-anchored falsification on the labeled subset (PRD §11).

The falsification analysis answers a single regulator-visible question:
when the pipeline predicts ``INAPPROPRIATE`` for an order, did the patient's
downstream outcome support that label or contradict it? Support means the
patient did not require further transfusion within 24 hours; contradiction
means a further transfusion or death from anemia within the follow-up
window. Orders where the model did not predict ``INAPPROPRIATE`` are
ignored — the falsification target is the pipeline's positive call, not
its negative call.

The labels are hand-coded by chart-reviewing clinicians on a small subset
(PRD §11). This module is the harness consumer for that subset; it does
NOT extract outcomes from the EMR itself (that is upstream chart review).
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.eval_harness.models import (
    Classification,
    FalsificationOutcome,
    FalsificationResult,
)


def outcome_anchored_falsification(
    predictions: Sequence[Classification],
    outcomes: Sequence[FalsificationOutcome],
) -> FalsificationResult:
    """Compute the support / contradiction rates for INAPPROPRIATE predictions.

    Both sequences must be the same length; otherwise raises
    :class:`bba.eval_harness.exceptions.ShapeMismatchError`. Empty input
    raises :class:`bba.eval_harness.exceptions.EmptyInputError` — a zero
    contradiction rate on zero cases is not a positive finding.

    Returned rates are Wilson CIs at 95% confidence on the
    INAPPROPRIATE-predicted subset; if zero predictions were
    ``INAPPROPRIATE``, the support and contradiction CIs collapse to
    ``[0, 0]`` (graceful — same handling as
    :func:`bba.eval_harness.classifier.evaluate_confusion_matrix`).
    """
    raise NotImplementedError(
        "eval_harness.falsification: RED phase, see issue #20"
    )
