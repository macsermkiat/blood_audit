"""Platelet LLM over-clear guardrail (docs plan §5.3 stage 5; user ruling 2026-07-08).

Pure, side-effect-free mirror of
:func:`bba.audit_pipeline.replay.llm_overclear_suspect`, adapted to platelet
hard signals. Fires when the LLM upgraded a withholding deterministic verdict
to ``APPROPRIATE`` and NO grounded platelet hard signal
(:class:`bba.platelet_guardrail.PlateletHardSignals`) justifies the clear —
in which case the pipeline floors the verdict to human review (never to
``INAPPROPRIATE``; a verdict the committee cannot audit is sent to review,
not condemned).

Covered verdict classes:

* ``NEEDS_REVIEW`` / ``INSUFFICIENT_EVIDENCE`` — sub-ceiling counts (plt < 100k)
  where the LLM must ground a positive indication; a bare low count (the
  TTP/HIT/dengue exclusion trap) can never clear alone.
* ``POTENTIALLY_INAPPROPRIATE`` — above-ceiling counts (plt >= 100k) where
  transfusion is almost never appropriate.  There is no separate high-count
  prompt path to act as a backstop; this guardrail is the sole safety mechanism.

This is the mechanism that makes the "ADD platelet hard signals" ruling safe:
the LLM CAN clear any platelet count, but only with a grounded positive
indication; an ungrounded clear at any level floors to human review.
"""

from __future__ import annotations

from bba.audit_store import Classification
from bba.platelet_guardrail.models import PlateletHardSignals

PLATELET_OVERCLEAR_REVIEW_REASON = "platelet_llm_overclear_suspect"
"""Typed ``review_reason`` stamped on platelet rows floored by this guardrail,
so the dashboard can triage LLM over-clear suspects separately."""

_PLATELET_OVERCLEAR_DET_VERDICTS: frozenset[Classification] = frozenset(
    {"NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE", "POTENTIALLY_INAPPROPRIATE"}
)
"""Deterministic platelet verdicts that withheld a clear.

An LLM APPROPRIATE upgrading any of these without a grounded
:class:`bba.platelet_guardrail.PlateletHardSignals` indication is an
over-clear candidate.  Includes POTENTIALLY_INAPPROPRIATE (plt >= ceiling)
because transfusing at a normal or high count is almost never appropriate
and there is no separate high-count prompt path backstop; an ungrounded
clear at high count must be reviewed just as at low count."""


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
