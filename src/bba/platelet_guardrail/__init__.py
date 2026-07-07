"""Platelet LLM over-clear guardrail + structured hard signals (Phase 2).

Public surface: :class:`PlateletHardSignals`, :func:`platelet_overclear_suspect`,
and the :data:`PLATELET_OVERCLEAR_REVIEW_REASON` slug.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.platelet_guardrail.guardrail import (
    PLATELET_OVERCLEAR_REVIEW_REASON,
    platelet_overclear_suspect,
)
from bba.platelet_guardrail.models import PlateletHardSignals

__all__: Sequence[str] = (
    "PLATELET_OVERCLEAR_REVIEW_REASON",
    "PlateletHardSignals",
    "platelet_overclear_suspect",
)
