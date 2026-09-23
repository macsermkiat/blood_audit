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
from bba.platelet_classifier import PLATELET_PROPHYLAXIS_THRESHOLD
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


PLATELET_TREND_REVIEW_REASON = "platelet_trend_unsupported"
"""Typed ``review_reason`` stamped on platelet rows floored by
:func:`platelet_trend_unsupported` (issue #237)."""


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


def platelet_trend_unsupported(
    final_classification: Classification,
    hard_signals: PlateletHardSignals,
    trigger_count_k_ul: float | None,
    projected_24h_k_ul: float | None,
) -> bool:
    """True iff an LLM clear rests only on an "expected to fall" claim the count
    history does not support (issue #237, clinician ruling 2026-09-21).

    At a trigger count of 10,000 /uL or more, the only route to
    ``prophylactic_marrow_failure`` is indication 5's second arm, "expected to
    drop below 10,000 /uL within 24 hours". That clause is a straight-line
    projection through the last two pre-order counts
    (:func:`bba.platelet_lookup.project_24h`), so code decides it: the model
    sets the hard signal itself, which is why
    :func:`platelet_overclear_suspect` cannot catch these.

    Fires only when ``prophylactic_marrow_failure`` is the ONLY true signal; a
    grounded bleeding / procedure / intracranial indication carries its own
    higher threshold. A ``None`` projection (fewer than two usable draws) does
    not support the clause. The pipeline floors a hit to human review, never to
    ``INAPPROPRIATE`` (never-guess convention).
    """
    if final_classification != "APPROPRIATE":
        return False
    if not hard_signals.prophylactic_marrow_failure:
        return False
    if (
        hard_signals.active_bleeding
        or hard_signals.procedure_indication
        or hard_signals.intracranial_bleed_indication
        or hard_signals.aplastic_active_therapy_indication
    ):
        return False
    if (
        trigger_count_k_ul is None
        or trigger_count_k_ul < PLATELET_PROPHYLAXIS_THRESHOLD
    ):
        return False
    return (
        projected_24h_k_ul is None
        or projected_24h_k_ul >= PLATELET_PROPHYLAXIS_THRESHOLD
    )


PLATELET_SPECIALIST_TARGET_REVIEW_REASON = "platelet_specialist_target"
"""Typed ``review_reason`` stamped on platelet rows moved to review by
:func:`platelet_specialist_target_review` (ruling 2026-09-23, REQNO 68012561)."""


def platelet_specialist_target_review(
    final_classification: Classification,
    hard_signals: PlateletHardSignals,
    trigger_count_k_ul: float | None,
) -> bool:
    """True iff an INAPPROPRIATE order followed a consultant's higher target.

    Ruling 2026-09-23 (REQNO 68012561: hemoptysis at 89,000 /uL, chest team
    "keep plt 100,000"): the order fails policy, but the ordering doctor acted
    on a specialist's documented advice, so the audit must not flag them as
    inappropriate. The target counts only when documented before the order
    (user decision 2026-09-23); the evidence ends at the order anchor, so later
    advice never reaches the model. In 68012561 the advice first appears 32 h
    after the order, so that case stays INAPPROPRIATE. The model reports the
    target it read; code does the
    comparison. Fires only when the trigger count sits below that target. A
    count at or above it means the order went beyond the advice, and a missing
    count leaves nothing to compare. The pipeline moves a hit to human review,
    never to ``APPROPRIATE``.
    """
    if final_classification != "INAPPROPRIATE":
        return False
    target = hard_signals.specialist_target_per_ul
    if target is None or trigger_count_k_ul is None:
        return False
    return trigger_count_k_ul * 1000 < target


__all__ = (
    "PLATELET_OVERCLEAR_REVIEW_REASON",
    "PLATELET_SPECIALIST_TARGET_REVIEW_REASON",
    "PLATELET_TREND_REVIEW_REASON",
    "platelet_overclear_suspect",
    "platelet_specialist_target_review",
    "platelet_trend_unsupported",
)
