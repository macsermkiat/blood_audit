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

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta

from bba.vitals_extractor.bounds import (
    is_bt_valid,
    is_dbp_valid,
    is_hr_valid,
    is_rr_valid,
    is_sbp_valid,
)
from bba.vitals_extractor.extractor import _extract_with_discards
from bba.vitals_extractor.models import (
    LLMFallback,
    SourceProvenance,
    VitalSigns,
    VitalsFlag,
    VitalsNote,
    VitalsResult,
)

_WINDOW = timedelta(hours=6)


def _source_rank(source: str) -> int:
    """Higher rank wins on tiebreak. IPDADMPROGRESS > IPDNRFOCUSDT.

    Implementation Decisions §4: when two notes share a timestamp, the
    cleaner SOAP-formatted source wins because its regex hits are higher-
    quality and the LLM fallback (if engaged) sees less noise.
    """
    return 1 if source == "IPDADMPROGRESS" else 0


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
    in_window = [n for n in notes if abs(n.timestamp - anchor) <= _WINDOW]

    if not in_window:
        return VitalsResult(
            vitals=VitalSigns(),
            source=SourceProvenance.NONE_IN_WINDOW,
            flags=frozenset(),
            note_timestamp=None,
        )

    pre = [n for n in in_window if n.timestamp <= anchor]
    post = [n for n in in_window if n.timestamp > anchor]

    flags: set[VitalsFlag] = set()
    if pre:
        # Most-recent pre-anchor; on tie, higher source rank (IPDADMPROGRESS) wins.
        chosen = max(pre, key=lambda n: (n.timestamp, _source_rank(n.source)))
    else:
        # Earliest post-anchor; on tie, higher source rank wins. We negate the
        # rank so ``min`` prefers IPDADMPROGRESS without an extra sort pass.
        chosen = min(post, key=lambda n: (n.timestamp, -_source_rank(n.source)))
        flags.add(VitalsFlag.POST_ORDER)

    regex_vitals, discards = _extract_with_discards(chosen.text)
    if discards:
        flags.add(VitalsFlag.DATA_ERROR)

    final_vitals = regex_vitals
    source_prov = SourceProvenance(chosen.source)

    needs_fallback = regex_vitals.sbp is None or regex_vitals.hr is None
    if llm_fallback is not None and needs_fallback:
        final_vitals, contributed = _merge_with_llm(regex_vitals, llm_fallback(chosen.text))
        if contributed:
            source_prov = SourceProvenance.LLM_EXTRACTED

    return VitalsResult(
        vitals=final_vitals,
        source=source_prov,
        flags=frozenset(flags),
        note_timestamp=chosen.timestamp,
    )


def _merge_with_llm(
    regex_vitals: VitalSigns,
    llm_vitals: VitalSigns,
) -> tuple[VitalSigns, bool]:
    """Fill regex-missing fields from the LLM result, gated by sanity bounds.

    Regex hits always win — the LLM never overwrites a value the regex found.
    A field is only filled from the LLM when the LLM's value passes its
    sanity bound, so the public output's "any value is in-bounds" invariant
    holds at the pipeline boundary too. Returns ``(merged, contributed)`` so
    the pipeline can decide whether to flip provenance to ``LLM_EXTRACTED``.
    """
    sbp, c_sbp = _fill_int(regex_vitals.sbp, llm_vitals.sbp, is_sbp_valid)
    dbp, c_dbp = _fill_int(regex_vitals.dbp, llm_vitals.dbp, is_dbp_valid)
    hr, c_hr = _fill_int(regex_vitals.hr, llm_vitals.hr, is_hr_valid)
    rr, c_rr = _fill_int(regex_vitals.rr, llm_vitals.rr, is_rr_valid)
    bt, c_bt = _fill_float(regex_vitals.bt, llm_vitals.bt, is_bt_valid)
    contributed = c_sbp or c_dbp or c_hr or c_rr or c_bt
    return VitalSigns(sbp=sbp, dbp=dbp, hr=hr, rr=rr, bt=bt), contributed


def _fill_int(
    regex_val: int | None,
    llm_val: int | None,
    validator: Callable[[int], bool],
) -> tuple[int | None, bool]:
    if regex_val is not None:
        return regex_val, False
    if llm_val is None or not validator(llm_val):
        return None, False
    return llm_val, True


def _fill_float(
    regex_val: float | None,
    llm_val: float | None,
    validator: Callable[[float], bool],
) -> tuple[float | None, bool]:
    if regex_val is not None:
        return regex_val, False
    if llm_val is None or not validator(llm_val):
        return None, False
    return llm_val, True
