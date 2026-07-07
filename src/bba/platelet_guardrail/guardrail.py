"""Platelet LLM over-clear guardrail (docs plan §5.3 stage 5; user ruling 2026-07-08).

Pure, side-effect-free mirror of
:func:`bba.audit_pipeline.replay.llm_overclear_suspect`, adapted to platelet
hard signals. Fires when the LLM upgraded a withholding deterministic verdict
(``NEEDS_REVIEW`` / ``INSUFFICIENT_EVIDENCE``) to ``APPROPRIATE`` and NO
grounded platelet hard signal (:class:`bba.platelet_guardrail.PlateletHardSignals`)
justifies the clear — in which case the pipeline floors the verdict to human
review (never to ``INAPPROPRIATE``; a verdict the committee cannot audit is
sent to review, not condemned).

This is the mechanism that makes the "ADD platelet hard signals" ruling safe:
the LLM CAN clear a sub-ceiling count, but only with a grounded positive
indication; a bare low count (the TTP/HIT/dengue exclusion trap) can never
clear.
"""

from __future__ import annotations

from bba.audit_store import Classification
from bba.platelet_guardrail.models import PlateletHardSignals

PLATELET_OVERCLEAR_REVIEW_REASON = "platelet_llm_overclear_suspect"
"""Typed ``review_reason`` stamped on platelet rows floored by this guardrail,
so the dashboard can triage LLM over-clear suspects separately."""

_PLATELET_OVERCLEAR_DET_VERDICTS: frozenset[Classification] = frozenset(
    {"NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE"}
)
"""Deterministic platelet verdicts that withheld a clear. Only an LLM
APPROPRIATE upgrading one of these is an over-clear candidate; a deterministic
POTENTIALLY_INAPPROPRIATE (plt >= ceiling) is handled by its own prompt path."""


def platelet_overclear_suspect(
    final_classification: Classification,
    rule_classification: Classification,
    hard_signals: PlateletHardSignals,
) -> bool:
    """True iff the LLM over-cleared a withholding deterministic platelet verdict.

    Fires only when the LLM returned ``APPROPRIATE``, the deterministic gate had
    withheld the clear, and no grounded platelet hard signal justifies it.
    Deterministic and side-effect-free so the pipeline override is trivially
    testable and replay-stable.
    """
    if final_classification != "APPROPRIATE":
        return False
    if rule_classification not in _PLATELET_OVERCLEAR_DET_VERDICTS:
        return False
    return not hard_signals.any_signal()


__all__ = (
    "PLATELET_OVERCLEAR_REVIEW_REASON",
    "platelet_overclear_suspect",
)
