"""The blood-bank visit fallback for a missing Hb, shared by the pilot scripts.

An issue event is judged at its issue time (``bba.hb_lookup.resolve_event_anchors``).
When no Hb sits in the 7 days before it, ``bba.hb_lookup.resolve_hb_with_fallback``
reads the first Hb after the earliest BDVSTDT BDVST{DATE,TIME} visit instead.
``run_pipeline.py``, ``run_llm_leg.py`` and ``_gold_check_hb.py`` build that
fallback here so all three read the same Hb.
"""

from __future__ import annotations

from datetime import datetime

from bba.hb_lookup import AnchorCandidate

from _hosxp_dt import _combine, _fmt_local_datetime, _parse_hosxp_date, _parse_time


def visit_fallback_candidates(
    bdvstdt_rows: list[dict[str, str]],
) -> dict[str, list[AnchorCandidate]]:
    """Each REQNO's earliest blood-bank visit as its one fallback candidate."""
    bank_anchor: dict[str, datetime] = {}
    bank_display: dict[str, str] = {}
    for r in bdvstdt_rows:
        reqno = r["REQNO"]
        visit_dt = _combine(
            _parse_hosxp_date(r.get("BDVSTDATE") or ""),
            _parse_time(r.get("BDVSTTIME") or ""),
        )
        if visit_dt is None:
            continue
        current = bank_anchor.get(reqno)
        if current is None or visit_dt < current:
            bank_anchor[reqno] = visit_dt
            bank_display[reqno] = _fmt_local_datetime(
                r.get("BDVSTDATE"), r.get("BDVSTTIME")
            )
    return {
        reqno: [
            AnchorCandidate(
                anchor_utc=at,
                display=bank_display[reqno],
                reason="blood_bank_visit_fallback",
            )
        ]
        for reqno, at in bank_anchor.items()
    }
