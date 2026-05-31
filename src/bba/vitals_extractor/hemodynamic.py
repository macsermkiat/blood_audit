"""Window-wide hemodynamic scan (issue #76).

This is a SEPARATE pass from the single-note :func:`extract_vitals` selection
(issue #6). Where that picks one best note for (SBP, DBP, HR, RR, BT), this scans
EVERY note in the window for the two signals that were starved in Case 2 /
REQNO 68012352:

* the **MAP nadir** — the lowest *measured* mean arterial pressure across the
  whole window (not the most-recent reading), because shock severity is the
  worst point, not the latest;
* **vasopressor mentions** — the agent (and dose, if charted) keeping the
  pressure up.

The result is a fact-only :class:`HemodynamicSummary`. It contains no
appropriateness language and no derived "instability" verdict — hemodynamic
status is a supporting factor for the auditor and the LLM, never a standalone
indication (the deterministic classifier has no hemodynamic gate).

Targets vs. measurements: ``keep MAP >/= 65`` / ``goal MAP 65`` are resuscitation
*orders*, not charted values, and are excluded so the scan never invents a
measurement. Ambiguous abbreviations (``NAD`` = no acute distress, ``Na`` =
sodium, ``N/A`` = not applicable) are never treated as norepinephrine.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence

from bba.vitals_extractor.bounds import is_map_valid
from bba.vitals_extractor.models import (
    HemodynamicSummary,
    VasopressorMention,
    VitalsNote,
)

# A measured MAP: "MAP 56", "MAP: 56", "MAP=70". The 2-3 digit floor rejects a
# single-digit OCR scrap, and the trailing ``(?!\d)`` rejects an overlong token
# like "MAP 560" outright rather than silently truncating it to a plausible
# in-range value (mirrors the extractor's lookahead discipline). Only ``:`` or a
# single ``=`` may sit between the label and the number, so a comparator form
# ("MAP >/= 65", "MAP >= 65") cannot match — those are targets, not readings.
_MAP_RE = re.compile(r"\bMAP\b\s*[:=]?\s*(\d{2,3})(?!\d)", re.IGNORECASE)

# Real KCMH/HOSxP charting rarely labels the MAP; it writes the arterial
# pressure as "ABP 77/44 (56)" / "ABP = 120/58(83)" / "BP 90/60 (70)" -- the
# systolic/diastolic pair with the MAP in parentheses. Capture all three
# numbers (sys, dia, map). A downstream physiological guard
# (:func:`_iter_map_measurements`) keeps the parenthesised value ONLY when it
# lies between diastolic and systolic, so an unrelated bracketed integer (a
# count, a score, a misparse) can never be fabricated into a MAP. The pilot run
# against the encrypted bundle proved the labelled-only regex matched nothing on
# Case 2 / REQNO 68012352, where every reading used this parenthesised form.
_ABP_PAREN_RE = re.compile(
    r"\b(?:A?BP|NIBP)\b\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*\(\s*(\d{2,3})\s*\)",
    re.IGNORECASE,
)

# Belt-and-suspenders for comparator-free target phrasing ("goal MAP 65",
# "keep MAP 65"): if any of these words sit just before the MAP token, the
# number is an order, not a measurement. Thai: รักษา ("maintain"), เป้า ("goal").
_MAP_TARGET_KEYWORDS = ("keep", "goal", "target", "maintain", "titrate", "รักษา", "เป้า")
_MAP_TARGET_LOOKBACK = 16

# A vasopressor/inotrope dose phrase trailing an agent name, e.g.
# "0.1 mcg/kg/min", "4 mg/hr", "2 units/min". Permissive on unit spelling; the
# captured substring is kept verbatim as provenance, never parsed for value.
_DOSE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mcg|µg|ug|mg|units?|u)\s*/\s*(?:kg\s*/\s*)?(?:min|hr|hour|h)\b",
    re.IGNORECASE,
)
_DOSE_LOOKAHEAD = 40

# (compiled pattern, canonical agent). Full drug/brand names are case-insensitive.
# "NE" is the one short alias we accept, and only as a case-SENSITIVE, fully
# word-bounded uppercase token: lowercasing or relaxing the boundary would start
# matching fragments of unrelated words. The genuinely ambiguous abbreviations
# (NA = sodium, NAD = no acute distress) are deliberately ABSENT — mapping them
# to a vasopressor would fabricate hemodynamic support.
_VASOPRESSOR_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bnor[\- ]?adrenalin[e]?\b", re.IGNORECASE), "norepinephrine"),
    (re.compile(r"\bnorepinephrine\b", re.IGNORECASE), "norepinephrine"),
    (re.compile(r"\blevophed\b", re.IGNORECASE), "norepinephrine"),
    (re.compile(r"นอร์อะดรีนาลีน"), "norepinephrine"),
    (re.compile(r"\bNE\b"), "norepinephrine"),
    (re.compile(r"\bepinephrine\b", re.IGNORECASE), "epinephrine"),
    (re.compile(r"\badrenalin[e]?\b", re.IGNORECASE), "epinephrine"),
    (re.compile(r"\bdopamine\b", re.IGNORECASE), "dopamine"),
    (re.compile(r"โดพามีน|โดปามีน"), "dopamine"),
    (re.compile(r"\bdobutamine\b", re.IGNORECASE), "dobutamine"),
    (re.compile(r"\bvasopressin\b", re.IGNORECASE), "vasopressin"),
)


def scan_hemodynamics(notes: Sequence[VitalsNote]) -> HemodynamicSummary:
    """Scan ``notes`` for the MAP nadir and vasopressor mentions.

    ``notes`` are assumed already filtered to the relevant window by the caller
    (the builder/pilot owns windowing); this function is a pure aggregation over
    whatever it is given and is total — empty input yields an empty summary.

    MAP nadir is the minimum in-bounds measured MAP across all notes, tie-broken
    by the earliest timestamp. Vasopressors are deduplicated by canonical agent,
    keeping the earliest mention (so its dose/timestamp/source stay consistent).
    """
    map_candidates: list[tuple[int, VitalsNote]] = []
    earliest_mention: dict[str, VasopressorMention] = {}

    for note in sorted(notes, key=lambda n: n.timestamp):
        for value in _iter_map_measurements(note.text):
            if is_map_valid(value):
                map_candidates.append((value, note))
        for agent, dose in _iter_vasopressors(note.text):
            if agent not in earliest_mention:
                earliest_mention[agent] = VasopressorMention(
                    agent=agent,
                    dose=dose,
                    at=note.timestamp,
                    source=note.source,
                )

    if map_candidates:
        nadir_value, nadir_note = min(
            map_candidates, key=lambda c: (c[0], c[1].timestamp)
        )
        map_nadir: int | None = nadir_value
        map_nadir_at = nadir_note.timestamp
        map_nadir_source: str | None = nadir_note.source
    else:
        map_nadir = None
        map_nadir_at = None
        map_nadir_source = None

    vasopressors = tuple(
        sorted(earliest_mention.values(), key=lambda v: (v.at, v.agent))
    )

    return HemodynamicSummary(
        map_nadir=map_nadir,
        map_nadir_at=map_nadir_at,
        map_nadir_source=map_nadir_source,  # type: ignore[arg-type]  # Literal narrowed by source
        vasopressors=vasopressors,
    )


def _iter_map_measurements(text: str) -> Iterator[int]:
    """Yield each *measured* MAP integer in ``text``, skipping targets/goals.

    Two notations are recognised: the labelled ``MAP 56`` form (target/goal
    phrasing excluded) and the parenthesised ``ABP 77/44 (56)`` form (kept only
    when the bracketed value is physiologically between diastolic and systolic).
    """
    for m in _MAP_RE.finditer(text):
        prefix = text[max(0, m.start() - _MAP_TARGET_LOOKBACK) : m.start()].lower()
        if any(kw in prefix for kw in _MAP_TARGET_KEYWORDS):
            continue
        yield int(m.group(1))
    for m in _ABP_PAREN_RE.finditer(text):
        systolic, diastolic, paren = (int(m.group(i)) for i in (1, 2, 3))
        if diastolic <= paren <= systolic:
            yield paren


def _iter_vasopressors(text: str) -> Iterator[tuple[str, str | None]]:
    """Yield ``(canonical_agent, dose_or_none)`` in text order of first appearance."""
    found: list[tuple[int, str, str | None]] = []
    for pattern, agent in _VASOPRESSOR_PATTERNS:
        for m in pattern.finditer(text):
            found.append((m.start(), agent, _dose_after(text, m.end())))
    found.sort(key=lambda f: f[0])
    for _start, agent, dose in found:
        yield agent, dose


def _dose_after(text: str, pos: int) -> str | None:
    """Return the dose phrase immediately following an agent at ``pos``, if any."""
    window = text[pos : pos + _DOSE_LOOKAHEAD]
    m = _DOSE_RE.search(window)
    return m.group(0).strip() if m else None
