"""Regex-first vital-sign extractor from a single free-text note.

Surface contract: :func:`extract_vitals_from_text` returns a :class:`VitalSigns`
in which any field that the regex layer could not confidently populate is
``None``. Values outside their sanity bound (see :mod:`bba.vitals_extractor.bounds`)
are discarded at this layer too — the pipeline records the
:class:`VitalsFlag.DATA_ERROR` flag based on whether any discard happened
versus this layer's clean output.

This module deliberately does NOT call the LLM fallback. The fallback is
orchestrated in :mod:`bba.vitals_extractor.pipeline`, gated on the AC rule
"only invoked when regex returns null SBP or HR" (issue #6).
"""

from __future__ import annotations

import re

from bba.vitals_extractor.bounds import (
    is_bt_valid,
    is_dbp_valid,
    is_hr_valid,
    is_rr_valid,
    is_sbp_valid,
)
from bba.vitals_extractor.models import VitalSigns

# Compiled once at import. Case-insensitive so "Temp" / "temp" / "TEMP" all
# qualify, and word-boundary-anchored so "PR" inside "PRESCRIPTION", "BP"
# inside "BPM", or stray "P" inside "Step" can never satisfy a label.
#
# Each captured numeric group is terminated by a ``(?!\d)`` negative lookahead
# so an OCR-noisy overlong token like "HR 2000" or "RR 500" can NEVER match
# its in-range prefix ("200", "50") and slip past sanity bounds. Truncation
# would silently produce wrong-but-plausible vitals with no DATA_ERROR flag,
# which is exactly the contract this extractor must uphold (codex review,
# 2026-05-15). BP relies on the structural ``/`` separator for the same
# protection; the bare "P" form is bounded by a trailing ``\b`` already.
_BP_RE = re.compile(r"\bBP\s*:?\s*(\d{2,3})\s*/\s*(\d{2,3})", re.IGNORECASE)
_HR_RE = re.compile(r"\b(?:HR|PR)\s*:?\s*(\d{2,3})(?!\d)", re.IGNORECASE)
# RR may carry an observed-variability range ("RR 20-23"); the lower bound is
# the deterministic floor (clinically: at-or-above this value across the
# observation window).
_RR_RE = re.compile(
    r"\bRR\s*:?\s*(\d{1,2})(?!\d)(?:\s*[-–]\s*\d{1,2}(?!\d))?",
    re.IGNORECASE,
)
# "P" alone is RR per the issue grouping; constrain to 1-2 digits + word
# boundary so it cannot consume a pulse-shaped 3-digit value.
_P_RR_RE = re.compile(r"\bP\s*:?\s*(\d{1,2})\b", re.IGNORECASE)
_BT_RE = re.compile(r"\b(?:BT|Temp)\s*:?\s*(\d{2,3}(?:\.\d+)?)(?!\d)", re.IGNORECASE)


def extract_vitals_from_text(text: str) -> VitalSigns:
    """Extract SBP/DBP/HR/RR/BT from ``text`` via regex.

    Recognized patterns (Thai + English; bounds-enforced):

    * BP: ``BP 110/60``, ``BP:118/63``
    * HR: ``PR108``, ``HR 97``
    * RR: ``P 14``, ``RR 20-23`` (range -> lower bound)
    * BT: ``BT 38.4``, ``Temp 37``

    Missing or out-of-bound values yield ``None`` for that field. The function
    is total: it never raises on malformed input.
    """
    vitals, _discards = _extract_with_discards(text)
    return vitals


def _extract_with_discards(text: str) -> tuple[VitalSigns, frozenset[str]]:
    """Like :func:`extract_vitals_from_text` but also report which fields had a
    regex hit that was rejected by sanity bounds.

    The discard set drives the pipeline's :class:`VitalsFlag.DATA_ERROR` flag.
    Module-private because the public surface stays a single ``VitalSigns``;
    only :mod:`bba.vitals_extractor.pipeline` needs the discard signal.
    """
    discards: set[str] = set()

    sbp: int | None = None
    dbp: int | None = None
    if m := _BP_RE.search(text):
        s, d = int(m.group(1)), int(m.group(2))
        if is_sbp_valid(s):
            sbp = s
        else:
            discards.add("sbp")
        if is_dbp_valid(d):
            dbp = d
        else:
            discards.add("dbp")

    hr: int | None = None
    if m := _HR_RE.search(text):
        h = int(m.group(1))
        if is_hr_valid(h):
            hr = h
        else:
            discards.add("hr")

    rr: int | None = None
    # Try the canonical RR label first; the bare "P" form is only a fallback
    # so a note that contains both "P 80" (pulse) and "RR 16" extracts rr=16
    # without spuriously discarding the pulse value as out-of-RR-bounds.
    rr_match = _RR_RE.search(text) or _P_RR_RE.search(text)
    if rr_match:
        r = int(rr_match.group(1))
        if is_rr_valid(r):
            rr = r
        else:
            discards.add("rr")

    bt: float | None = None
    if m := _BT_RE.search(text):
        b = float(m.group(1))
        if is_bt_valid(b):
            bt = b
        else:
            discards.add("bt")

    return VitalSigns(sbp=sbp, dbp=dbp, hr=hr, rr=rr, bt=bt), frozenset(discards)
