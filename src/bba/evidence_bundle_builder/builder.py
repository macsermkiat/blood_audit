"""Top-level evidence-bundle pipeline (issue #16).

Orchestrates the per-source windowing, ranking, ID assignment, canonical
serialization, and hashing into a single :class:`EvidenceBundle`. The window
constants live here so every threshold the issue body names has exactly one
authoritative declaration in the codebase.
"""

from __future__ import annotations

from datetime import timedelta

from bba.evidence_bundle_builder.models import (
    EvidenceBundle,
    EvidenceInputs,
)


# =============================================================================
# Per-source windows (PRD §7 / issue #16 body)
#
# Every constant here is referenced by a test in
# ``tests/unit/test_evidence_bundle_builder.py``. Changing a value is a
# spec-level change — the windows are not tunable knobs, they encode the
# clinical "what's relevant to a transfusion decision" decision.
# =============================================================================

WINDOW_PROGRESS = timedelta(hours=24)
"""IPDADMPROGRESS window: ``[anchor - 24h, anchor + 24h]``."""

WINDOW_FOCUS = timedelta(hours=24)
"""IPDNRFOCUSDT window: ``[anchor - 24h, anchor + 24h]``, then 5+5 split."""

WINDOW_MED_BEFORE = timedelta(hours=72)
"""MED window backward edge: ``anchor - 72h`` (drug history shaping the order)."""

WINDOW_MED_AFTER = timedelta(hours=24)
"""MED window forward edge: ``anchor + 24h`` (post-order administration)."""

WINDOW_HB_BEFORE = timedelta(days=7)
"""Lab Hb history backward edge: ``anchor - 7d`` (mirrors :mod:`bba.hb_lookup`)."""

WINDOW_VITALS = timedelta(hours=6)
"""Vitals window: ``[anchor - 6h, anchor + 6h]`` (mirrors :mod:`bba.vitals_extractor`)."""

CAP_PROGRESS = 8
"""Maximum IPDADMPROGRESS entries in a bundle (issue #16 body)."""

CAP_FOCUS_BEFORE = 5
"""Maximum IPDNRFOCUSDT entries strictly before the order anchor."""

CAP_FOCUS_AFTER = 5
"""Maximum IPDNRFOCUSDT entries at or after the order anchor."""

DEFAULT_CHAR_CAP = 8000
"""Bundle character-count proxy for the LLM token budget (issue #16 body).

A char-count proxy is used instead of a tiktoken-style estimate because the
bundle goes to Anthropic's tokenizer (not OpenAI's) and the published Claude
tokenization is not stable enough to depend on for a hard cap. 8 K characters
is roughly 2 K tokens of typical Thai medical text, well inside the prompt
budget after the system prompt + few-shot examples."""


def build_evidence_bundle(
    *,
    inputs: EvidenceInputs,
    char_cap: int = DEFAULT_CHAR_CAP,
) -> EvidenceBundle:
    """Build the canonical evidence bundle for one order anchor.

    Pipeline:

    1. Window-filter each source to the windows in this module.
    2. Rank IPDNRFOCUSDT 5-before / 5-after the anchor (closest-first).
    3. Cap IPDADMPROGRESS at :data:`CAP_PROGRESS`; on tie, prefer entries
       closer to the anchor.
    4. Parse IPDADMPROGRESS notes into SOAP sections (priority order).
    5. Emit items in canonical source order (the literal order of
       :data:`bba.evidence_bundle_builder.models.EvidenceSource`); within
       each source, sort by ``(timestamp_utc, payload-key-then-value)`` so
       the assignment of E1, E2, ..., EN is byte-stable across input
       shuffles.
    6. Truncate to ``char_cap`` using
       :func:`bba.evidence_bundle_builder.ranking.truncate_to_char_cap`.
    7. Serialize via :func:`bba.evidence_bundle_builder.canonical.canonical_serialize`
       and hash via :func:`bba.evidence_bundle_builder.canonical.bundle_hash`.

    The returned :class:`EvidenceBundle.bundle_hash` lands on every
    :class:`bba.audit_store.AuditRow.evidence_bundle_hash` and is what
    :mod:`bba.deid_redactor` reads to detect mid-pipeline mutation.
    """
    raise NotImplementedError("build_evidence_bundle: not implemented in RED phase")


__all__ = (
    "CAP_FOCUS_AFTER",
    "CAP_FOCUS_BEFORE",
    "CAP_PROGRESS",
    "DEFAULT_CHAR_CAP",
    "WINDOW_FOCUS",
    "WINDOW_HB_BEFORE",
    "WINDOW_MED_AFTER",
    "WINDOW_MED_BEFORE",
    "WINDOW_PROGRESS",
    "WINDOW_VITALS",
    "build_evidence_bundle",
)
