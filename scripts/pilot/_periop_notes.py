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

from datetime import datetime

from bba.ingest.models import ParsedTimeOfDay
from bba.vitals_extractor import VitalsNote

from _hosxp_dt import _combine, _parse_hosxp_date, _parse_time


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
        dt = _combine(
            _parse_hosxp_date(r.get("PROGDATE") or ""),
            ParsedTimeOfDay(hour=0, minute=0, second=0),
        )
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
