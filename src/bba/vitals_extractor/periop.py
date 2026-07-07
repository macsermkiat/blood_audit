"""Window-wide peri-operative scan (Case 107 / REQNO 68074627).

This is a SEPARATE pass from :func:`scan_hemodynamics`. Where Case 2 starved the
LLM of MAP/vasopressor evidence, Case 107 starved it of the *surgical context*:
an Hb of 12.6 (above threshold) with a 5-hour ORIF and 1500 ml blood loss, but
the structured operative rows were empty and the surgery lived only in a
free-text IPDNRFOCUSDT nursing note. The LLM returned INSUFFICIENT_EVIDENCE and
wrote "no operative procedure documented" — it trusted the structured absence
over the prose that was already in the bundle. This scan lifts that prose into a
structured, high-salience signal the LLM cannot skim past.

Three fact-only signals are recovered:

* **surgical context** — a charted operation (ORIF/CRIF, "post-op", ผ่าตัด,
  "under GA/spinal", craniotomy/laparotomy, "to OR"). Boolean: a surgery named
  anywhere in the window counts, even with no structured procedure code.
* **estimated blood loss** — the EBL volume, normalized to millilitres (litres
  ×1000, "cc" as-is) so a unit-of-measure quirk cannot hide a major haemorrhage.
  The MAX across the window is kept (the worst loss, not the latest charted).
* **intra-operative transfusion** — a SPECIFIC blood component (LPRC, PRBC, FFP,
  platelets, ...) co-located with an intra-op marker. Generic "blood" near
  "intra-op" (e.g. "intra-op blood loss") is deliberately NOT matched.

The result is a fact-only :class:`PeriopSummary` (see its binding guardrail): no
appropriateness language, no verdict. Peri-op context is a supporting factor the
auditor and the LLM weigh; the deterministic classifier's procedure bypass keys
on structured timing, never on this scan.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from bba.vitals_extractor.models import (
    PeriopFinding,
    PeriopSummary,
    VitalsNote,
)

# STRONG surgical cues: any one asserts peri-operative context. Curated to
# favour precision over recall on Thai ortho/cardiac/general notes (lifted from
# the notes_surgical_context prototype). CRIF is added alongside ORIF because
# Case 107 charted both. ``EBL`` is intentionally NOT a surgery cue here — the
# EBL volume is recovered by its own extractor below, so an EBL line already
# makes the summary non-empty without being mislabelled "surgery".
_SURGERY_CUE_RE = re.compile(
    r"(ผ่าตัด|นัดมาทำ|under\s+(?:SAB|GA|GGA|spinal)|optime|post[\s-]?op"
    r"|\bRedo\b|\bORIF\b|\bCRIF\b|\bTKA\b|\bTHA\b|\bPVR\b|craniotomy|laparotomy"
    r"|ห้องผ่าตัด|ไป\s*OR\b)",
    re.IGNORECASE,
)

# EBL / blood-loss volume + unit (lifted verbatim from build_review._EBL_RE so
# the bundle's structured signal and the human-review display agree on what
# counts as a blood-loss line).
_EBL_RE = re.compile(
    r"\b(?:EBL|blood\s*loss)\s*[:=.]?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(mL|ml|cc|L|liter|liters)\b",
    re.IGNORECASE,
)

# Intra-operative transfusion: a specific blood component within 40 chars of an
# intra-op marker (either order). Precision-favouring by design — only named
# components count, never generic "blood"/"เลือด", so "intra-op blood loss" is
# not misread as a transfusion. The marker requirement excludes the ward
# transfusion order being audited (no marker) from begging the question.
_INTRAOP_MARKER = r"(?:intra[\s-]?op(?:erative)?|ในห้องผ่าตัด|ระหว่างผ่าตัด)"
_BLOOD_COMPONENT = r"(?:LPRC|PRBC|PRC|FFP|platelets?|SDP|cryo(?:precipitate)?)"
_INTRAOP_TX_RE = re.compile(
    rf"{_INTRAOP_MARKER}[^\n]{{0,40}}?{_BLOOD_COMPONENT}"
    rf"|{_BLOOD_COMPONENT}[^\n]{{0,40}}?{_INTRAOP_MARKER}",
    re.IGNORECASE,
)

_SNIPPET_RADIUS = 60


def scan_periop(notes: Sequence[VitalsNote]) -> PeriopSummary:
    """Scan ``notes`` for surgical context, EBL, and intra-op transfusion.

    ``notes`` are assumed already filtered to the relevant window by the caller
    (the builder/pilot owns windowing); this function is a pure aggregation over
    whatever it is given and is total — empty input yields an empty summary.

    At most one finding per category is kept: the earliest surgery cue, the
    MAX-volume EBL (tie-broken by earliest note then earliest position), and the
    earliest intra-op transfusion. Iterating notes in timestamp order makes the
    "earliest" selection deterministic across input shuffles.
    """
    surgery: PeriopFinding | None = None
    intraop: PeriopFinding | None = None
    best_loss_ml: int | None = None
    loss_finding: PeriopFinding | None = None

    for note in sorted(notes, key=lambda n: n.timestamp):
        if surgery is None:
            m = _SURGERY_CUE_RE.search(note.text)
            if m is not None:
                surgery = _finding("surgery", note, m.start(), m.end())

        for em in _EBL_RE.finditer(note.text):
            ml = _ebl_to_ml(em.group(1), em.group(2))
            if ml is None or ml <= 0:
                continue
            if best_loss_ml is None or ml > best_loss_ml:
                best_loss_ml = ml
                loss_finding = _finding("blood_loss", note, em.start(), em.end())

        if intraop is None:
            tm = _INTRAOP_TX_RE.search(note.text)
            if tm is not None:
                intraop = _finding("intraop_transfusion", note, tm.start(), tm.end())

    findings = tuple(f for f in (surgery, loss_finding, intraop) if f is not None)
    return PeriopSummary(
        surgical_context=surgery is not None,
        blood_loss_ml=best_loss_ml,
        intraop_transfusion=intraop is not None,
        findings=findings,
    )


def _finding(category: str, note: VitalsNote, start: int, end: int) -> PeriopFinding:
    return PeriopFinding(
        category=category,  # type: ignore[arg-type]  # narrowed by call sites
        snippet=_snippet(note.text, start, end),
        at=note.timestamp,
        source=note.source,
    )


def _ebl_to_ml(amount: str, unit: str) -> int | None:
    """Convert a captured EBL ``amount``/``unit`` to integer millilitres.

    Litres scale ×1000; "cc" and "mL" are millilitres as-is. Returns ``None``
    on an unparseable amount rather than fabricating a volume."""
    try:
        value = float(amount.replace(",", ""))
    except ValueError:
        return None
    if unit.lower() in ("l", "liter", "liters"):
        value *= 1000.0
    return int(round(value))


def _snippet(text: str, start: int, end: int) -> str:
    """A whitespace-collapsed window around ``[start, end)`` with ellipses."""
    lo = max(0, start - _SNIPPET_RADIUS)
    hi = min(len(text), end + _SNIPPET_RADIUS)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return prefix + " ".join(text[lo:hi].split()) + suffix
