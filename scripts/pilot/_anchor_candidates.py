"""Shared Hb-anchor candidate builder for the pilot scripts.

Extracts the bundle/HOSxP-row anchor logic that used to live inline in
``run_pipeline.py`` so both ``run_pipeline.py`` and ``run_llm_leg.py``
build the *same* ordered fallback candidates and feed them to the single
resolver (``bba.hb_lookup.resolve_hb_with_fallback``).

The candidate ORDER encodes the clinical fallback ladder and must not
change: the exact issue/collection datetime first, then the blood-bank
visit timestamp. See ``docs/handoff-hb-anchor-unification.md``.
"""

from __future__ import annotations

from datetime import datetime

from bba.hb_lookup import AnchorCandidate

from _hosxp_dt import _combine, _fmt_local_datetime, _parse_hosxp_date, _parse_time


def build_anchor_candidates(
    *,
    bdvstdt_rows: list[dict[str, str]],
    bdvst_by_reqno: dict[str, dict[str, str]],
) -> dict[str, list[AnchorCandidate]]:
    """Build ordered Hb-anchor fallback candidates keyed by REQNO.

    Each list is ordered ``[issue_datetime?, blood_bank_visit_fallback?]``;
    a candidate is omitted when its source datetime is absent. The issue
    anchor prefers BDVST PICK{DATE,TIME}, then BDVSTDT USE{DATE,TIME}, and
    keeps the *earliest* such datetime per REQNO. The blood-bank fallback
    is the earliest BDVSTDT BDVST{DATE,TIME} visit timestamp.
    """
    issue_anchor: dict[str, datetime] = {}
    issue_display: dict[str, str] = {}
    bank_anchor: dict[str, datetime] = {}
    bank_display: dict[str, str] = {}

    for r in bdvstdt_rows:
        reqno = r["REQNO"]

        visit_dt = _combine(
            _parse_hosxp_date(r.get("BDVSTDATE") or ""),
            _parse_time(r.get("BDVSTTIME") or ""),
        )
        if visit_dt is not None:
            current = bank_anchor.get(reqno)
            if current is None or visit_dt < current:
                bank_anchor[reqno] = visit_dt
                bank_display[reqno] = _fmt_local_datetime(
                    r.get("BDVSTDATE"), r.get("BDVSTTIME")
                )

        parent = bdvst_by_reqno.get(reqno, {})
        pick_date = (parent.get("PICKDATE") or "").strip().split(" ")[0]
        pick_time_raw = (parent.get("PICKTIME") or "").strip()
        pick_dt = (
            _combine(_parse_hosxp_date(pick_date), _parse_time(pick_time_raw))
            if pick_date and pick_time_raw
            else None
        )
        if pick_dt is not None:
            if reqno not in issue_anchor or pick_dt < issue_anchor[reqno]:
                issue_anchor[reqno] = pick_dt
                issue_display[reqno] = _fmt_local_datetime(pick_date, pick_time_raw)
            continue

        use_date = (r.get("USEDATE") or "").strip().split(" ")[0]
        use_time_raw = (r.get("USETIME") or "").strip()
        if use_date and use_time_raw:
            use_dt = _combine(_parse_hosxp_date(use_date), _parse_time(use_time_raw))
            if use_dt is not None and (
                reqno not in issue_anchor or use_dt < issue_anchor[reqno]
            ):
                issue_anchor[reqno] = use_dt
                issue_display[reqno] = _fmt_local_datetime(use_date, use_time_raw)

    candidates: dict[str, list[AnchorCandidate]] = {}
    for reqno in {*issue_anchor, *bank_anchor}:
        ordered: list[AnchorCandidate] = []
        if reqno in issue_anchor:
            ordered.append(
                AnchorCandidate(
                    anchor_utc=issue_anchor[reqno],
                    display=issue_display[reqno],
                    reason="issue_datetime",
                )
            )
        if reqno in bank_anchor:
            ordered.append(
                AnchorCandidate(
                    anchor_utc=bank_anchor[reqno],
                    display=bank_display[reqno],
                    reason="blood_bank_visit_fallback",
                )
            )
        candidates[reqno] = ordered
    return candidates
