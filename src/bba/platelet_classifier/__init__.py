"""Deterministic platelet appropriateness gate (Phase 2, docs plan §5.1).

Public surface: :func:`classify_platelet`, the :data:`PLATELET_REVIEW_CEILING`
constant, and the frozen :class:`PlateletClassifierInputs` /
:class:`PlateletClassifierResult` models.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.platelet_classifier.classifier import (
    PLATELET_REVIEW_CEILING,
    classify_platelet,
)
from bba.platelet_classifier.models import (
    PlateletClassifierInputs,
    PlateletClassifierResult,
)

__all__: Sequence[str] = (
    "PLATELET_REVIEW_CEILING",
    "PlateletClassifierInputs",
    "PlateletClassifierResult",
    "classify_platelet",
)
