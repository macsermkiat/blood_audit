"""Platelet LLM over-clear guardrail + structured hard signals (Phase 2).

Public surface: :class:`PlateletHardSignals`, :func:`platelet_overclear_suspect`
and its :data:`PLATELET_OVERCLEAR_REVIEW_REASON` slug, plus the count-trend floor
:func:`platelet_trend_unsupported` and its :data:`PLATELET_TREND_REVIEW_REASON`
slug (issue #237), and the specialist-target review floor
:func:`platelet_specialist_target_review` with its
:data:`PLATELET_SPECIALIST_TARGET_REVIEW_REASON` slug.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.platelet_guardrail.guardrail import (
    PLATELET_OVERCLEAR_REVIEW_REASON,
    PLATELET_SPECIALIST_TARGET_REVIEW_REASON,
    PLATELET_TREND_REVIEW_REASON,
    platelet_overclear_suspect,
    platelet_specialist_target_review,
    platelet_trend_unsupported,
)
from bba.platelet_guardrail.models import PlateletHardSignals

__all__: Sequence[str] = (
    "PLATELET_OVERCLEAR_REVIEW_REASON",
    "PLATELET_SPECIALIST_TARGET_REVIEW_REASON",
    "PLATELET_TREND_REVIEW_REASON",
    "PlateletHardSignals",
    "platelet_overclear_suspect",
    "platelet_specialist_target_review",
    "platelet_trend_unsupported",
)
