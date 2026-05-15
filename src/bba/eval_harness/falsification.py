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

from bba.eval_harness.classifier import _safe_wilson
from bba.eval_harness.exceptions import EmptyInputError, ShapeMismatchError
from bba.eval_harness.models import (
    Classification,
    FalsificationOutcome,
    FalsificationResult,
)

_CONTRADICTION = {
    FalsificationOutcome.FURTHER_TRANSFUSION_24H,
    FalsificationOutcome.DEATH_FROM_ANEMIA_30D,
}


def outcome_anchored_falsification(
    predictions: Sequence[Classification],
    outcomes: Sequence[FalsificationOutcome],
) -> FalsificationResult:
    """Compute the support / contradiction rates for INAPPROPRIATE predictions."""
    if len(predictions) != len(outcomes):
        raise ShapeMismatchError(
            f"outcome_anchored_falsification: predictions ({len(predictions)}) "
            f"and outcomes ({len(outcomes)}) disagree"
        )
    if not predictions:
        raise EmptyInputError(
            "outcome_anchored_falsification: predictions must be non-empty"
        )

    inappropriate_outcomes = [
        out
        for pred, out in zip(predictions, outcomes, strict=True)
        if pred == "INAPPROPRIATE"
    ]
    n_inappropriate = len(inappropriate_outcomes)
    n_contradicted = sum(1 for o in inappropriate_outcomes if o in _CONTRADICTION)
    n_supported = n_inappropriate - n_contradicted

    support_rate = _safe_wilson(n_supported, n_inappropriate, confidence=0.95)
    contradiction_rate = _safe_wilson(
        n_contradicted, n_inappropriate, confidence=0.95
    )

    return FalsificationResult(
        n_labeled=len(predictions),
        n_inappropriate_pred=n_inappropriate,
        n_supported=n_supported,
        n_contradicted=n_contradicted,
        support_rate=support_rate,
        contradiction_rate=contradiction_rate,
    )
