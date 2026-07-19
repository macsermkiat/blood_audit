"""Rationale slugs that carry routing meaning beyond the audit trail.

``ClassifierResult.rationale`` is normally a short debug slug, but three of its
values are load-bearing dispatch inputs downstream — MSBOS reservation
eligibility, the reserve-ahead prompt task-mode, and the declared-exempt overlay.
They live here as a single source of truth so the producer
(:func:`bba.deterministic_classifier.classify`) and every consumer share one
definition: a rename is a single edit and the sets cannot drift between copies.

The slugs stay plain strings (they are persisted verbatim as
``ClassifierResult.rationale`` / audit-row ``rationale``); this module adds no
new stored field. Downstream typed routing derives from them on demand — see
:class:`bba.audit_pipeline.replay.ReservationEligibility`.
"""

from __future__ import annotations

from typing import Final

# A clinician-declared surgical / type-screen order: exempt from the transfusion
# judgment, but still MSBOS-reservation-screen eligible. The classifier emits
# PERIOP_TRANSFUSION_EXEMPT with this rationale.
PREOP_DECLARED_EXEMPT: Final = "preop_declared_exempt"

# Reserve-ahead deferrals to the LLM (classification NEEDS_REVIEW): a crossmatch /
# declared reservation is not itself a transfusion indication, so the order is
# routed to review rather than auto-cleared.
PREOP_DEFER_LLM: Final = "preop_defer_llm"  # structured upcoming operative row
PREOP_DEFER_LLM_DECLARED: Final = "preop_defer_llm_declared"  # declared-only, no op row

# The reserve-ahead dispatch set (both deferral flavours). Consumers test
# ``rationale in RESERVE_AHEAD_RATIONALES``.
RESERVE_AHEAD_RATIONALES: Final[frozenset[str]] = frozenset(
    {PREOP_DEFER_LLM, PREOP_DEFER_LLM_DECLARED}
)

__all__ = [
    "PREOP_DECLARED_EXEMPT",
    "PREOP_DEFER_LLM",
    "PREOP_DEFER_LLM_DECLARED",
    "RESERVE_AHEAD_RATIONALES",
]
