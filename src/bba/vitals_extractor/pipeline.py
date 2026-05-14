"""Top-level vital-sign extraction pipeline (issue #6).

Selects the single best :class:`VitalsNote` in the +/-6 h window around the
order anchor, runs the regex extractor, optionally invokes the LLM fallback
(only when regex left SBP or HR null), and returns a :class:`VitalsResult`
with provenance and quality flags.

Selection rule:

1. Filter notes to ``[anchor - 6h, anchor + 6h]``.
2. Among pre-anchor notes, pick the most-recent. Tiebreak by source:
   ``IPDADMPROGRESS`` beats ``IPDNRFOCUSDT`` (cleaner SOAP structure).
3. If no pre-anchor note exists, pick the earliest post-anchor note and
   record the :class:`VitalsFlag.POST_ORDER` flag.
4. If no note in the window, return source ``NONE_IN_WINDOW`` and an
   all-None :class:`VitalSigns`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from bba.vitals_extractor.models import (
    LLMFallback,
    VitalsNote,
    VitalsResult,
)


def extract_vitals(
    *,
    anchor: datetime,
    notes: Sequence[VitalsNote],
    llm_fallback: LLMFallback | None = None,
) -> VitalsResult:
    """Run the full vitals-extraction pipeline for one order anchor.

    ``anchor`` MUST be tz-aware UTC (ingest contract). ``notes`` may be empty
    or unsorted; the pipeline performs the window filter and the source-aware
    most-recent-before-anchor selection.

    ``llm_fallback`` is the regex-vs-llm-structured-text boundary: when given
    AND the chosen note's regex extraction left ``sbp`` or ``hr`` as ``None``,
    it is invoked exactly once with the note text; its result fills in
    regex-missing fields. Provenance flips to ``LLM_EXTRACTED`` if the
    fallback actually contributed any non-None field that regex missed.
    """
    del anchor, notes, llm_fallback  # signature pinned; body in the GREEN phase
    raise NotImplementedError("extract_vitals — RED phase (issue #6)")
