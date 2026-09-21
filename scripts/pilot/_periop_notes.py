"""Shared free-text note loader for the pilot scripts.

Single owner of the IPDADMPROGRESS / IPDNRFOCUSDT → :class:`VitalsNote`
build so ``run_pipeline.py`` (deterministic leg) and ``run_llm_leg.py``
(LLM leg) feed :func:`bba.vitals_extractor.periop.scan_periop` from the
SAME note set. The two legs must stay unified — a divergent note loader
would let the deterministic peri-op pre-pass and the LLM bundle disagree
on what surgery/EBL was documented. Extracted verbatim from the previous
``run_llm_leg._vitals_notes_for`` copy.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from bba.ingest.models import ParsedTimeOfDay
from bba.vitals_extractor import VitalsNote

from _hosxp_dt import _combine, _parse_hosxp_date, _parse_time


_END_OF_DAY = ParsedTimeOfDay(hour=23, minute=59, second=59)
_MAX_ENTRY_LAG = timedelta(days=1)


def progress_note_timestamp(progdate_raw: str, firstdate_raw: str) -> datetime | None:
    """When a doctor progress note became part of the record (UTC).

    ``PROGDATE`` is a date with no time; ``FIRSTDATE`` is when the note was first
    saved. The loader used to stamp every note at 00:00 of ``PROGDATE``. The
    evidence bundle keeps progress notes within 24 h of the order, so every
    previous-day note looked more than 24 h old and was dropped (platelet order
    68000711: both doctor notes naming an acute SDH, written at 14:09 and 19:19
    the day before a 10:11 order; 78 of 364 LLM-judged hematology orders).

    * ``FIRSTDATE`` on the clinical date or the day after (96% of real notes,
      the second case being notes typed the next morning): use it.
    * Otherwise (back-filled days later, entered before the clinical date,
      blank or unparseable): the entry time says nothing about the clinical
      day, so the note sits at the END of ``PROGDATE``, which keeps it visible
      to a next-day order.
    """
    clinical_day = _parse_hosxp_date(progdate_raw)
    if clinical_day is None:
        return None
    entered = _entry_datetime(firstdate_raw)
    if entered is not None:
        entry_day, entry_time = entered
        if timedelta(0) <= entry_day - clinical_day <= _MAX_ENTRY_LAG:
            return _combine(entry_day, entry_time)
    return _combine(clinical_day, _END_OF_DAY)


def _entry_datetime(raw: str) -> tuple[date, ParsedTimeOfDay] | None:
    """Split a ``YYYY-MM-DD HH:MM:SS[.mmm]`` cell into its local date and time."""
    day_part, _, time_part = raw.strip().partition(" ")
    day = _parse_hosxp_date(day_part)
    digits = time_part.split(".", 1)[0].replace(":", "")
    time_of_day = _parse_time(digits) if len(digits) == 6 else None
    if day is None or time_of_day is None:
        return None
    return day, time_of_day


def vitals_notes_for(
    progress: list[dict[str, str]],
    focus: list[dict[str, str]],
    an: str,
    anchor: datetime,
) -> tuple[VitalsNote, ...]:
    """Build a :class:`VitalsNote` list from notes for the AN.

    PHI safety is handled by the upstream de-identification gate (issue #76):
    these notes — and the narrative the bundle ships from them — are treated as
    already de-identified, so the full SOAP text is forwarded, not just numbers.

    IPDADMPROGRESS carries four free-text SOAP columns (S/O/A/P); MAP and
    vasopressor evidence can live in any of them, so all four are joined here.
    Restricting to OBJECTIVE would starve both the LLM narrative and the
    hemodynamic scan of assessment/plan-charted pressor support. Each column is
    prefixed with its SOAP label so the builder's ``parse_soap_sections`` can
    re-split them; an unlabelled join would collapse to a single OBJECTIVE block
    and let priority-aware truncation drop the assessment/plan with it.

    ``anchor`` is accepted for signature parity with the per-leg windowing
    helpers; windowing itself is owned by the caller / builder, so this loader
    returns every note for the AN.
    """
    _ = anchor
    out: list[VitalsNote] = []
    for r in progress:
        if r.get("AN") != an:
            continue
        dt = progress_note_timestamp(r.get("PROGDATE") or "", r.get("FIRSTDATE") or "")
        soap = (
            ("Subjective", (r.get("SUBJECTIVE") or "").strip()),
            ("Objective", (r.get("OBJECTIVE") or "").strip()),
            ("Assessment", (r.get("ASSESSMENT") or "").strip()),
            ("Plan", (r.get("PLAN") or "").strip()),
        )
        text = "\n".join(f"{label}: {value}" for label, value in soap if value)
        if dt is None or not text:
            continue
        out.append(
            VitalsNote(
                timestamp=dt,
                text=text,
                source="IPDADMPROGRESS",
            )
        )
    for r in focus:
        if r.get("AN") != an:
            continue
        dt = _combine(
            _parse_hosxp_date(r.get("PROGRESSDATE") or ""),
            _parse_time(r.get("PROGRESSTIME") or ""),
        )
        text = " ".join(
            filter(
                None,
                [
                    (r.get("ACTION") or "").strip(),
                    (r.get("RESPONSE") or "").strip(),
                ],
            )
        )
        if dt is None or not text:
            continue
        out.append(
            VitalsNote(
                timestamp=dt,
                text=text,
                source="IPDNRFOCUSDT",
            )
        )
    return tuple(out)
