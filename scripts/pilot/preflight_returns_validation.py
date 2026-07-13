"""Pre-flight validation report for the returns-ledger screen (ticket #125).

This is a READ-ONLY validation report. It PRODUCES the evidence a clinician
owner needs to decide whether to flip ``RETURNS_LEDGER_ENABLED`` on for a real
pilot run; it never enables the feature and never changes pipeline behaviour.
It computes dispositions DIRECTLY on the pilot bundle with the shipped
``summarize_returns`` (#120) and mirrors the deterministic leg's screened-set
decision (the #122 ``RETURNED_NOT_TRANSFUSED`` terminal) — it does not re-derive
disposition logic and does not touch physician attribution.

It answers four questions the spec's completeness note (spec #119) requires
before go-live:

1. **Reissue / partial-coverage prevalence** on the screened set — orders whose
   BDVSTTRANS unit count disagrees with the ordered quantity (a visible reissue;
   a partial export could also be *hiding* a transfused replacement unit that no
   count-based guard can detect — the residual over-screen risk Codex confirmed).
2. **Administration-note recall** — does a screened order carry an affirmative
   ``ให้เลือด`` note the terminal would hide? Reported both admission-wide and
   WINDOWED to the order's dispense->return interval (the windowed count drives
   the gate; admission-wide over-flags separate transfusions in a long stay).
3. **Invariant** — zero screened orders contain a non-returned unit, re-derived
   from the raw ledger rows so it cannot false-pass off ``summarize_returns``.
4. A short **sign-off summary** with a go / narrow / hold recommendation.

The count is reconciled in layers so the "real 55-order count" (deferred from
#122/#123/#124) is auditable: raw all-returned (300) -> among audited orders ->
ledger-complete ``not_transfused`` -> screened (no hard intra-op/EBL guard hit).

Environment variables:

* ``BBA_PILOT_WORK_DIR`` — directory containing ``bundle/`` (default
  ``/tmp/bba_mini``), same as ``run_pipeline.py``.
* ``BBA_PREFLIGHT_BDVSTTRANS`` — path to the returns ledger. Defaults to the
  bundle's ``BDVSTTRANS.csv`` if present, else the raw Bloodbank export.
* ``BBA_PREFLIGHT_OUT`` — path for the machine-readable JSON artifact (default
  ``<work>/preflight_returns_validation.json``).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from bba.audit_orders import (
    AuditOrdersConfig,
    BloodOrderInput,
    build_audit_orders,
)
from bba.deterministic_classifier import PERIOP_MIN_EBL_ML
from bba.returns_ledger import ReturnsSummary, summarize_returns
from bba.vitals_extractor import VitalsNote, scan_periop
from bba.vitals_extractor.administration import scan_administration

from _hosxp_dt import _parse_hosxp_date, _parse_time
from _periop_notes import vitals_notes_for

csv.field_size_limit(sys.maxsize)

WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
BUNDLE = WORK / "bundle"
CODE_VERSION = "pilot-mini"

# "Returned" keys on the returned status code (spec #119, decision 2), the same
# constant summarize_returns uses. Re-stated here so the invariant check reads
# the raw rows independently of the summary's own counters.
_RETURNED_STATUS = "3"

# Windowed-recall padding (days) around an order's dispense->return interval.
# An administration of THIS order's standby units is charted while the units are
# out; ±2 days tolerates late/next-shift charting without reaching the separate
# transfusions that make up the rest of a weeks-long admission.
_RECALL_WINDOW_PAD_DAYS = 2

# The partial returns ledger lives outside the repo, alongside the other raw
# HOSxP exports (mirrors sample_bundle.py's ``../Bloodbank/data`` convention).
_RAW_BDVSTTRANS_DEFAULT = (
    Path(__file__).resolve().parents[2].parent
    / "Bloodbank"
    / "data"
    / "raw"
    / "BDVSTTRANS.csv"
)


# --- pure decision functions (unit-tested) -----------------------------------


def is_reissue(summary: ReturnsSummary) -> bool:
    """Whether a ledger-complete order's unit count disagrees with the order.

    A complete ledger with more physical units than were ordered signals a
    reissued / replacement unit. On the screened set (all complete) this is the
    visible face of the residual risk that a partial export could instead be
    *hiding* a transfused replacement unit (spec #119 Risk 1).
    """
    return (
        summary.ledger_complete
        and summary.ordered_unit_amount is not None
        and summary.units_total != summary.ordered_unit_amount
    )


def is_over_dispense_guard_excluded(summary: ReturnsSummary) -> bool:
    """An all-returned order the disposition guard now excludes from the screen.

    A ledger-complete order whose units are ALL returned but whose count
    exceeds the ordered amount is an over-dispensed reissue: ``summarize_returns``
    derives it ``inconclusive`` (spec #119 NARROW), so it never reaches the
    screened ``not_transfused`` set. Surfaced separately so the clinician
    sign-off still documents which orders the NARROW guard excluded, rather than
    silently folding them into the inconclusive bucket.
    """
    return (
        summary.ledger_complete
        and summary.units_total > 0
        and summary.units_returned == summary.units_total
        and summary.disposition == "inconclusive"
    )


def hard_transfusion_contradiction(
    *, intraop_transfusion: bool, blood_loss_ml: int | None
) -> bool:
    """Mirror the classifier's returned-exit contradiction guard.

    A charted intra-operative transfusion or estimated blood loss at or above
    :data:`PERIOP_MIN_EBL_ML` contradicts an all-returned ledger, so the #122
    terminal leaves the order in the legacy chain (fail loud). Cross-checked
    against the real ``classify`` in the test suite so it cannot drift from
    ``classifier.py``.
    """
    return intraop_transfusion or (
        blood_loss_ml is not None and blood_loss_ml >= PERIOP_MIN_EBL_ML
    )


def is_screened_returned_not_transfused(
    summary: ReturnsSummary,
    *,
    intraop_transfusion: bool,
    blood_loss_ml: int | None,
) -> bool:
    """Whether the deterministic leg would screen this order not-transfused.

    Equals ``classify(...).classification == "RETURNED_NOT_TRANSFUSED"`` for the
    returns exit (pinned by ``test_screened_predicate_matches_real_classifier``).
    """
    return (
        summary.disposition == "not_transfused"
        and not hard_transfusion_contradiction(
            intraop_transfusion=intraop_transfusion, blood_loss_ml=blood_loss_ml
        )
    )


def nonreturned_unit_count(trans_rows: Sequence[Mapping[str, str]]) -> int:
    """Count ledger rows whose ``UNITSTAT`` is not the returned status.

    Re-derived straight from the raw rows (not from ``ReturnsSummary``'s
    counters) so the "no screened order contains a non-returned unit" invariant
    is an independent check, not a tautology over the same aggregation.
    """
    return sum(
        1
        for r in trans_rows
        if str(r.get("UNITSTAT") or "").strip() != _RETURNED_STATUS
    )


@dataclass(frozen=True)
class RecallConflict:
    """A screened order carrying an affirmative administration note."""

    reqno: str
    categories: tuple[str, ...]
    snippets: tuple[str, ...]


def administration_recall_conflicts(
    screened_notes: Mapping[str, Sequence[VitalsNote]],
) -> tuple[RecallConflict, ...]:
    """Scan each screened order's notes for a ให้เลือด administration marker.

    Any affirmative red-cell administration marker on a screened (all-returned)
    order is a CONFLICT: the ``RETURNED_NOT_TRANSFUSED`` terminal would hide a
    charted administration. Delegates recall to the shipped, precision-favoured
    :func:`scan_administration`.
    """
    conflicts: list[RecallConflict] = []
    for reqno in sorted(screened_notes):
        summary = scan_administration(screened_notes[reqno])
        if summary.has_affirmative_marker:
            conflicts.append(
                RecallConflict(
                    reqno=reqno,
                    categories=tuple(f.category for f in summary.findings),
                    snippets=tuple(f.snippet for f in summary.findings),
                )
            )
    return tuple(conflicts)


def parse_ledger_date(raw: str | None) -> date | None:
    """Parse a BDVSTTRANS PAYDATE/RTNDATE (``'March 31, 2025, 3:29 PM'``).

    Returns ``None`` on any unrecognised/blank value so the caller fails SAFE —
    an unwindowable order keeps its full admission-wide notes rather than
    silently dropping a possible administration.
    """
    text = (raw or "").strip()
    if not text:
        return None
    for fmt in ("%B %d, %Y, %I:%M %p", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def recall_window(
    anchor_dates: Sequence[date | None], *, pad_days: int = _RECALL_WINDOW_PAD_DAYS
) -> tuple[date, date] | None:
    """Padded ``[earliest, latest]`` window over an order's dispense/return dates.

    ``None`` when no anchor date is parseable — the caller then falls back to the
    admission-wide notes (fail-safe: never narrow a window we cannot place).
    """
    dates = [d for d in anchor_dates if d is not None]
    if not dates:
        return None
    return (
        min(dates) - timedelta(days=pad_days),
        max(dates) + timedelta(days=pad_days),
    )


def notes_in_window(
    notes: Sequence[VitalsNote], window: tuple[date, date] | None
) -> tuple[VitalsNote, ...]:
    """Notes whose date falls in ``window``; all notes when ``window`` is None."""
    if window is None:
        return tuple(notes)
    lo, hi = window
    return tuple(n for n in notes if lo <= n.timestamp.date() <= hi)


@dataclass(frozen=True)
class SiblingUnit:
    """One ledger unit of a DIFFERENT order in the same admission."""

    reqno: str
    dispense_date: date | None
    is_returned: bool
    status: str


def explaining_sibling(
    window: tuple[date, date] | None, sibling_units: Sequence[SiblingUnit]
) -> SiblingUnit | None:
    """A sibling order's NOT-returned unit dispensed inside ``window``, or None.

    When a screened (all-returned) order carries an in-window administration
    note, a not-returned unit of a *different* order in the same admission,
    dispensed while the screened standby units were out, is the parsimonious
    source of that note — so the note is attributed there rather than to the
    returned order (the automated form of the 68019920 -> 68020779
    adjudication). Conservative: fires only on a concrete dispensed-not-returned
    unit, and every attribution is reported for human audit. Returns the
    earliest-dispensed matching unit (deterministic).
    """
    if window is None:
        return None
    lo, hi = window
    matches = [
        u
        for u in sibling_units
        if u.dispense_date is not None
        and lo <= u.dispense_date <= hi
        and not u.is_returned
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda u: (u.dispense_date, u.reqno))[0]


def recommendation(
    *,
    screened_count: int,
    notes_available: bool,
    reissue_count: int,
    recall_conflicts: int,
    invariant_violations: int,
) -> str:
    """Deterministic go / narrow / hold gate for the clinician sign-off.

    Absence of evidence is NOT clean (Codex critical): an empty/misjoined ledger
    screens nothing, and missing note exports make recall vacuous — neither may
    read as GO. So GO/NARROW require actual coverage first:

    * HOLD when there is nothing to validate (``screened_count == 0``) or recall
      cannot be assessed (``notes_available`` is False), OR when recall is not
      clean / the invariant is violated (a screened order may hide a real
      transfusion — correctness-threatening).
    * NARROW when coverage exists, recall is clean and the invariant holds but a
      reissue signal is present — go live only on the count-agreeing orders.
    * GO only when there is coverage AND every check is clean.
    """
    if screened_count == 0 or not notes_available:
        return "HOLD"
    if recall_conflicts > 0 or invariant_violations > 0:
        return "HOLD"
    if reissue_count > 0:
        return "NARROW"
    return "GO"


# --- findings the report emits -----------------------------------------------


@dataclass(frozen=True)
class ReissueFinding:
    reqno: str
    units_total: int
    ordered_unit_amount: int | None


@dataclass(frozen=True)
class HardFallthrough:
    reqno: str
    intraop_transfusion: bool
    blood_loss_ml: int | None


@dataclass(frozen=True)
class InvariantViolation:
    reqno: str
    statuses: tuple[str, ...]


@dataclass(frozen=True)
class NoteAttribution:
    """A windowed recall hit attributed to a different same-admission order."""

    reqno: str
    attributed_to: str
    dispensed: str
    status: str


@dataclass(frozen=True)
class PreflightResult:
    bdvsttrans_source: str
    orders_total: int
    orders_included: int
    orders_excluded: int
    orders_red_cell: int
    orders_joined_to_ledger: int
    platelet_all_returned: int
    raw_all_returned_total: int
    raw_all_returned_excluded: int
    raw_all_returned_included: int
    not_transfused: int
    hard_fallthroughs: list[HardFallthrough]
    screened: int
    notes_available: bool
    screened_without_notes: int
    disposition_counts: dict[str, int]
    reissue_findings: list[ReissueFinding]
    recall_conflicts: list[RecallConflict]
    recall_conflicts_windowed: list[RecallConflict]
    recall_conflicts_windowed_net: list[RecallConflict]
    note_attributions: list[NoteAttribution]
    invariant_violations: list[InvariantViolation]
    recommendation: str
    screened_reqnos: list[str] = field(default_factory=list)
    # Over-dispensed all-returned orders the NARROW guard excludes from the
    # screen (they derive `inconclusive`, so they never enter `screened`).
    over_dispense_guard_excluded: list[ReissueFinding] = field(default_factory=list)


# --- bundle loading (read-only) ----------------------------------------------


def _read_csv(name: str) -> list[dict[str, str]]:
    with (BUNDLE / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_optional_csv(name: str) -> list[dict[str, str]]:
    path = BUNDLE / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _icd_codes(*raws: str | None) -> tuple[str, ...]:
    """Split HOSxP diagnosis fields on ``,``/``;`` (mirrors run_pipeline)."""
    out: list[str] = []
    for raw in raws:
        if not raw:
            continue
        for chunk in raw.replace(";", ",").split(","):
            code = chunk.strip()
            if code:
                out.append(code)
    return tuple(out)


def _resolve_bdvsttrans_path() -> Path:
    override = os.environ.get("BBA_PREFLIGHT_BDVSTTRANS", "").strip()
    if override:
        return Path(override)
    bundle_copy = BUNDLE / "BDVSTTRANS.csv"
    if bundle_copy.exists():
        return bundle_copy
    return _RAW_BDVSTTRANS_DEFAULT


def _load_bdvsttrans_by_reqno(path: Path) -> dict[str, list[dict[str, str]]]:
    """Index the returns ledger by REQNO with uppercased keys (spec #119)."""
    by_reqno: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            row = {k.upper(): v for k, v in raw.items()}
            by_reqno.setdefault(row.get("REQNO") or "", []).append(row)
    return by_reqno


def _build_inputs(
    bdvst: list[dict[str, str]],
    products_by_reqno: Mapping[str, list[str]],
    diag_by_an: Mapping[str, list[str]],
) -> list[BloodOrderInput]:
    inputs: list[BloodOrderInput] = []
    for r in bdvst:
        an = (r.get("AN") or "").strip() or None
        reqno = r["REQNO"]
        inputs.append(
            BloodOrderInput(
                hn=r["HN"],
                an=an,
                reqno=reqno,
                bdvstst=(r.get("BDVSTST") or "").strip(),
                reqtype=(r.get("REQTYPE") or "").strip(),
                canceldate=(r.get("CANCELDATE") or "").strip() or None,
                req_date=_parse_hosxp_date(r.get("REQDATE") or ""),
                req_time=_parse_time(r.get("REQTIME") or ""),
                bdvst_date=_parse_hosxp_date(r.get("BDVSTDATE") or ""),
                bdvst_time=_parse_time(r.get("BDVSTTIME") or ""),
                products=tuple(products_by_reqno.get(reqno, [])),
                diagnosis_codes=_icd_codes(
                    *(diag_by_an.get(an or "", []) + [r.get("ICD10")])
                ),
            )
        )
    return inputs


# --- validation ---------------------------------------------------------------


def _all_returned(trans_rows: Sequence[Mapping[str, str]]) -> bool:
    return bool(trans_rows) and nonreturned_unit_count(trans_rows) == 0


def _sibling_units_for(
    an: str,
    own_reqno: str,
    reqnos_by_an: Mapping[str, set[str]],
    trans_by_reqno: Mapping[str, list[dict[str, str]]],
) -> list[SiblingUnit]:
    """Ledger units of every OTHER order in ``an`` (same admission)."""
    units: list[SiblingUnit] = []
    for reqno in reqnos_by_an.get(an, set()):
        if reqno == own_reqno:
            continue
        for u in trans_by_reqno.get(reqno, []):
            status = str(u.get("UNITSTAT") or "").strip()
            units.append(
                SiblingUnit(
                    reqno=reqno,
                    dispense_date=parse_ledger_date(u.get("PAYDATE")),
                    is_returned=status == _RETURNED_STATUS,
                    status=status,
                )
            )
    return units


def run_preflight() -> PreflightResult:
    if not BUNDLE.exists():
        sys.exit(f"bundle not found: {BUNDLE} (run sample_bundle.py first)")

    bdvst = _read_csv("BDVST.csv")
    bdvstdt = _read_csv("BDVSTDT.csv")
    diag = _read_csv("Diagnosis.csv")
    progress = _read_optional_csv("IPDADMPROGRESS.csv")
    focus = _read_optional_csv("IPDNRFOCUSDT.csv")
    # Same-admission related orders (sidecar) power windowed-recall attribution:
    # an in-window note is attributed to a neighbouring dispensed-not-returned
    # order rather than the returned one. Optional -> attribution just no-ops.
    related = _read_optional_csv("BDVST_RELATED.csv")
    reqnos_by_an: dict[str, set[str]] = {}
    for r in [*bdvst, *related]:
        an_v, rq = (r.get("AN") or "").strip(), (r.get("REQNO") or "").strip()
        if an_v and rq:
            reqnos_by_an.setdefault(an_v, set()).add(rq)

    bdvsttrans_path = _resolve_bdvsttrans_path()
    if not bdvsttrans_path.exists():
        sys.exit(f"BDVSTTRANS ledger not found: {bdvsttrans_path}")
    trans_by_reqno = _load_bdvsttrans_by_reqno(bdvsttrans_path)

    products_by_reqno: dict[str, list[str]] = {}
    unitamt_lines_by_reqno: dict[str, list[str]] = {}
    for r in bdvstdt:
        reqno = r["REQNO"]
        products_by_reqno.setdefault(reqno, []).append((r.get("BDTYPE") or "").strip())
        unitamt_lines_by_reqno.setdefault(reqno, []).append(
            (r.get("UNITAMT") or "").strip()
        )

    diag_by_an: dict[str, list[str]] = {}
    for r in diag:
        diag_by_an.setdefault(r.get("AN", ""), []).append(
            (r.get("ICD10") or "").strip()
        )

    filter_result = build_audit_orders(
        _build_inputs(bdvst, products_by_reqno, diag_by_an),
        AuditOrdersConfig(code_version=CODE_VERSION),
    )
    included_reqnos = {o.reqno for o in filter_result.included}

    # Reconcile the raw all-returned count across every layer (spec #125 "real
    # 55-order count"). The audited count keys on build_audit_orders inclusion.
    raw_all_returned_total = sum(
        1 for r in bdvst if _all_returned(trans_by_reqno.get(r["REQNO"], []))
    )
    raw_all_returned_included = sum(
        1 for reqno in included_reqnos if _all_returned(trans_by_reqno.get(reqno, []))
    )
    raw_all_returned_excluded = raw_all_returned_total - raw_all_returned_included

    # A note source must actually be present or the recall check is vacuous
    # (Codex high): missing exports would make every screened order read "clean".
    notes_available = bool(progress) or bool(focus)
    orders_joined_to_ledger = sum(
        1 for o in filter_result.included if trans_by_reqno.get(o.reqno)
    )

    disposition_counts: Counter[str] = Counter()
    orders_red_cell = 0
    not_transfused = 0
    platelet_all_returned = 0
    screened_reqnos: list[str] = []
    screened_notes: dict[str, Sequence[VitalsNote]] = {}
    screened_notes_windowed: dict[str, Sequence[VitalsNote]] = {}
    window_by_reqno: dict[str, tuple[date, date] | None] = {}
    sibling_units_by_reqno: dict[str, list[SiblingUnit]] = {}
    screened_without_notes = 0
    hard_fallthroughs: list[HardFallthrough] = []
    reissue_findings: list[ReissueFinding] = []
    over_dispense_guard_excluded: list[ReissueFinding] = []
    invariant_violations: list[InvariantViolation] = []

    for order in filter_result.included:
        trans_rows = trans_by_reqno.get(order.reqno, [])
        summary = summarize_returns(
            trans_rows, unitamt_lines_by_reqno.get(order.reqno, [])
        )
        # The pilot deterministic leg exits the platelet path BEFORE the returns
        # wiring (run_pipeline.py), so only red-cell orders are ever screened
        # there (Codex high). Count platelet all-returned separately for the
        # tracked platelet-returns follow-up; never screen them here.
        if order.component != "red_cell":
            if summary.disposition == "not_transfused":
                platelet_all_returned += 1
            continue
        orders_red_cell += 1
        disposition_counts[summary.disposition] += 1
        if summary.disposition != "not_transfused":
            # Surface the over-dispensed all-returned orders the NARROW guard
            # excludes from the screen, so the sign-off documents the excluded
            # reissues instead of silently folding them into `inconclusive`.
            if is_over_dispense_guard_excluded(summary):
                over_dispense_guard_excluded.append(
                    ReissueFinding(
                        reqno=order.reqno,
                        units_total=summary.units_total,
                        ordered_unit_amount=summary.ordered_unit_amount,
                    )
                )
            continue
        not_transfused += 1

        # Mirror the pilot deterministic leg: admission-wide notes -> scan_periop
        # -> the classifier's hard-transfusion contradiction guard.
        notes = vitals_notes_for(progress, focus, order.an, order.order_datetime)
        periop = scan_periop(notes)
        if not is_screened_returned_not_transfused(
            summary,
            intraop_transfusion=periop.intraop_transfusion,
            blood_loss_ml=periop.blood_loss_ml,
        ):
            hard_fallthroughs.append(
                HardFallthrough(
                    reqno=order.reqno,
                    intraop_transfusion=periop.intraop_transfusion,
                    blood_loss_ml=periop.blood_loss_ml,
                )
            )
            continue

        screened_reqnos.append(order.reqno)
        screened_notes[order.reqno] = notes
        if not notes:
            screened_without_notes += 1
        # Windowed recall: restrict notes to this order's dispense->return
        # interval so a later, separate transfusion elsewhere in a weeks-long
        # admission is not misattributed to the returned standby units.
        window = recall_window(
            [parse_ledger_date(r.get("PAYDATE")) for r in trans_rows]
            + [parse_ledger_date(r.get("RTNDATE")) for r in trans_rows]
        )
        screened_notes_windowed[order.reqno] = notes_in_window(notes, window)
        window_by_reqno[order.reqno] = window
        sibling_units_by_reqno[order.reqno] = _sibling_units_for(
            order.an or "", order.reqno, reqnos_by_an, trans_by_reqno
        )

        if is_reissue(summary):
            reissue_findings.append(
                ReissueFinding(
                    reqno=order.reqno,
                    units_total=summary.units_total,
                    ordered_unit_amount=summary.ordered_unit_amount,
                )
            )
        # Invariant, re-derived straight from the raw ledger rows (NOT from the
        # summary counters): a screened order must contain zero non-returned
        # units. It guards against summarize_returns disposition drift; a unit
        # ABSENT from the partial export is out of its reach and is instead
        # covered by the administration-note recall check below.
        if nonreturned_unit_count(trans_rows) > 0:
            invariant_violations.append(
                InvariantViolation(
                    reqno=order.reqno,
                    statuses=tuple(
                        str(r.get("UNITSTAT") or "").strip() for r in trans_rows
                    ),
                )
            )

    recall_conflicts = list(administration_recall_conflicts(screened_notes))
    recall_conflicts_windowed = list(
        administration_recall_conflicts(screened_notes_windowed)
    )
    # Auto-attribution: demote a windowed conflict when a NOT-returned unit of a
    # different order in the same admission was dispensed inside the window (the
    # generalised 68019920 -> 68020779 adjudication). The net count drives the
    # gate; each attribution is reported for human audit.
    recall_conflicts_windowed_net: list[RecallConflict] = []
    note_attributions: list[NoteAttribution] = []
    for conflict in recall_conflicts_windowed:
        sibling = explaining_sibling(
            window_by_reqno.get(conflict.reqno),
            sibling_units_by_reqno.get(conflict.reqno, ()),
        )
        if sibling is None:
            recall_conflicts_windowed_net.append(conflict)
        else:
            note_attributions.append(
                NoteAttribution(
                    reqno=conflict.reqno,
                    attributed_to=sibling.reqno,
                    dispensed=str(sibling.dispense_date),
                    status=sibling.status,
                )
            )

    return PreflightResult(
        bdvsttrans_source=str(bdvsttrans_path),
        orders_total=len(bdvst),
        orders_included=len(filter_result.included),
        orders_excluded=len(filter_result.excluded),
        orders_red_cell=orders_red_cell,
        orders_joined_to_ledger=orders_joined_to_ledger,
        platelet_all_returned=platelet_all_returned,
        raw_all_returned_total=raw_all_returned_total,
        raw_all_returned_excluded=raw_all_returned_excluded,
        raw_all_returned_included=raw_all_returned_included,
        not_transfused=not_transfused,
        hard_fallthroughs=hard_fallthroughs,
        screened=len(screened_reqnos),
        notes_available=notes_available,
        screened_without_notes=screened_without_notes,
        disposition_counts=dict(disposition_counts),
        reissue_findings=reissue_findings,
        recall_conflicts=recall_conflicts,
        recall_conflicts_windowed=recall_conflicts_windowed,
        recall_conflicts_windowed_net=recall_conflicts_windowed_net,
        note_attributions=note_attributions,
        invariant_violations=invariant_violations,
        # NET windowed recall (after same-admission attribution) is the accurate
        # "does THIS order hide a transfusion" measure and drives the gate; the
        # windowed-pre-attribution and admission-wide counts are reported for
        # transparency (both over-flag separate transfusions on a shared AN).
        recommendation=recommendation(
            screened_count=len(screened_reqnos),
            notes_available=notes_available,
            reissue_count=len(reissue_findings),
            recall_conflicts=len(recall_conflicts_windowed_net),
            invariant_violations=len(invariant_violations),
        ),
        screened_reqnos=screened_reqnos,
        over_dispense_guard_excluded=over_dispense_guard_excluded,
    )


# --- reporting ----------------------------------------------------------------


def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def print_report(result: PreflightResult) -> None:
    line = "=" * 78
    print(line)
    print("RETURNS-LEDGER PRE-FLIGHT VALIDATION (ticket #125, spec #119)")
    print("READ-ONLY. Produces the go/no-go evidence; does NOT enable the feature.")
    print(line)
    print(f"bundle          : {BUNDLE}")
    print(f"returns ledger  : {result.bdvsttrans_source}")
    if result.bdvsttrans_source == str(_RAW_BDVSTTRANS_DEFAULT):
        print(
            "  (note: bundle carries no BDVSTTRANS.csv; joined the raw partial "
            "export by REQNO)"
        )

    print("\n-- Coverage (a validation gate must have something to validate) --")
    print(
        f"  audited orders joined to ledger by REQNO : "
        f"{result.orders_joined_to_ledger} / {result.orders_included} "
        f"({_pct(result.orders_joined_to_ledger, result.orders_included)})"
    )
    print(
        f"  note sources present (recall assessable) : {result.notes_available}"
        f"  (screened orders with no notes: {result.screened_without_notes})"
    )
    if result.platelet_all_returned:
        print(
            f"  platelet all-returned (NOT screened here) : "
            f"{result.platelet_all_returned}  (pilot leg skips platelet returns; "
            "tracked follow-up)"
        )

    print("\n-- Disposition reconciliation ('real 55-order count') --")
    print(f"  orders in bundle                         : {result.orders_total}")
    print(
        f"  audited (build_audit_orders included)    : {result.orders_included}"
        f"  (excluded {result.orders_excluded}; red-cell {result.orders_red_cell})"
    )
    print(
        f"  raw all-returned, all orders             : {result.raw_all_returned_total}"
    )
    print(
        f"    of which excluded from the audit       : {result.raw_all_returned_excluded}"
    )
    print(
        f"    among audited orders                   : {result.raw_all_returned_included}"
    )
    print(
        f"  ledger-complete not_transfused (red-cell): {result.not_transfused}"
        f"  (all-returned minus incomplete/inconclusive)"
    )
    print(
        f"  hard-contradiction fall-throughs         : {len(result.hard_fallthroughs)}"
        "  (intra-op transfusion or EBL >= "
        f"{PERIOP_MIN_EBL_ML} ml -> NOT screened)"
    )
    print(f"  ==> SCREENED as RETURNED_NOT_TRANSFUSED   : {result.screened}")
    print(f"  disposition counts (red-cell audited)    : {result.disposition_counts}")
    print(
        f"  over-dispense guard excluded (NARROW)    : "
        f"{len(result.over_dispense_guard_excluded)}"
        "  (all-returned but ledger count != ordered -> inconclusive, NOT screened)"
    )
    for ex in result.over_dispense_guard_excluded:
        print(
            f"      excluded reqno={ex.reqno} ledger_units={ex.units_total} "
            f"ordered={ex.ordered_unit_amount}"
        )
    for hf in result.hard_fallthroughs:
        print(
            f"      fall-through reqno={hf.reqno} "
            f"intraop={hf.intraop_transfusion} ebl={hf.blood_loss_ml}"
        )

    print("\n-- 1. Reissue / partial-coverage prevalence (screened set) --")
    print(
        f"  offenders: {len(result.reissue_findings)} / {result.screened} "
        f"({_pct(len(result.reissue_findings), result.screened)})"
    )
    for rf in result.reissue_findings:
        print(
            f"      reqno={rf.reqno} ledger_units={rf.units_total} "
            f"ordered={rf.ordered_unit_amount}"
        )

    print("\n-- 2. Administration-note recall (screened set) --")
    print(
        f"  Notes windowed to the order's dispense->return interval "
        f"+/-{_RECALL_WINDOW_PAD_DAYS}d; an in-window hit is then attributed to a "
        "same-admission neighbour that was dispensed-not-returned in the window."
    )
    net = result.recall_conflicts_windowed_net
    print(
        f"    NET windowed conflicts (drives gate): {len(net)} / "
        f"{result.screened} ({_pct(len(net), result.screened)})"
    )
    for rc in net:
        print(f"      CONFLICT reqno={rc.reqno} categories={list(rc.categories)}")
        for snip in rc.snippets:
            print(f"          {snip}")
    print(
        f"    auto-attributed to a sibling order  : {len(result.note_attributions)}"
        "  (in-window note explained by another order -> NOT a true conflict)"
    )
    for na in result.note_attributions:
        print(
            f"      reqno={na.reqno} -> sibling {na.attributed_to} "
            f"(dispensed {na.dispensed}, status {na.status} not-returned)"
        )
    print(
        f"    windowed (pre-attribution): {len(result.recall_conflicts_windowed)}"
        f"   | admission-wide (context): {len(result.recall_conflicts)} "
        f"[{', '.join(rc.reqno for rc in result.recall_conflicts) or 'none'}]"
    )

    print("\n-- 3. Invariant: no screened order contains a non-returned unit --")
    if result.invariant_violations:
        print(f"  VIOLATED: {len(result.invariant_violations)} order(s)")
        for iv in result.invariant_violations:
            print(f"      reqno={iv.reqno} statuses={list(iv.statuses)}")
    else:
        print("  HOLDS (0 violations)")

    print("\n" + line)
    print("SIGN-OFF SUMMARY (for the clinician owner)")
    print(line)
    print(_signoff_text(result))
    print(line)


def _signoff_text(result: PreflightResult) -> str:
    reissue_n = len(result.reissue_findings)
    recall_net = len(result.recall_conflicts_windowed_net)
    recall_w = len(result.recall_conflicts_windowed)
    recall_a = len(result.recall_conflicts)
    attributed_n = len(result.note_attributions)
    invariant_ok = not result.invariant_violations
    lines = [
        f"Real all-returned count : {result.raw_all_returned_total} in the "
        f"{result.orders_total}-order pilot; {result.raw_all_returned_included} "
        f"among the {result.orders_included} audited orders; "
        f"{result.not_transfused} ledger-complete; {result.screened} screened as "
        "RETURNED_NOT_TRANSFUSED after the hard intra-op/EBL guard.",
        f"NARROW guard excluded   : {len(result.over_dispense_guard_excluded)} "
        "all-returned order(s) whose ledger unit count != ordered (over-dispensed "
        "reissue) -> inconclusive, NOT screened"
        + (
            f" (reqnos {', '.join(ex.reqno for ex in result.over_dispense_guard_excluded)})."
            if result.over_dispense_guard_excluded
            else "."
        ),
        f"Reissue prevalence      : {reissue_n} / {result.screened} screened orders "
        f"({_pct(reissue_n, result.screened)}) have a ledger unit count that "
        "disagrees with the ordered quantity"
        + (
            f" (reqnos {', '.join(rf.reqno for rf in result.reissue_findings)})."
            if result.reissue_findings
            else "."
        ),
        f"Administration recall   : {recall_net} / {result.screened} screened orders "
        f"({_pct(recall_net, result.screened)}) carry an affirmative ให้เลือด note "
        f"in the dispense->return window (+/-{_RECALL_WINDOW_PAD_DAYS}d) NOT explained "
        "by another same-admission order"
        + (
            f" (reqnos {', '.join(rc.reqno for rc in result.recall_conflicts_windowed_net)})."
            if result.recall_conflicts_windowed_net
            else " — net recall is clean."
        )
        + f" [{attributed_n} auto-attributed to a sibling order; windowed pre-"
        f"attribution {recall_w}; admission-wide {recall_a}]",
        f"Invariant               : {'HOLDS' if invariant_ok else 'VIOLATED'} — "
        f"{'zero' if invariant_ok else str(len(result.invariant_violations))} "
        "screened order(s) contain a non-returned unit.",
        "",
        f"RECOMMENDATION: {result.recommendation}",
        _recommendation_rationale(result),
    ]
    return "\n".join(lines)


def _recommendation_rationale(result: PreflightResult) -> str:
    if result.recommendation == "HOLD":
        reasons = []
        if result.screened == 0:
            reasons.append(
                "zero orders are screened (nothing to validate — an empty or "
                "misjoined ledger cannot be blessed as clean)"
            )
        if not result.notes_available:
            reasons.append(
                "no note sources were loaded, so administration recall is "
                "unassessable (a hidden transfusion would go undetected)"
            )
        if result.recall_conflicts_windowed_net:
            reasons.append(
                f"{len(result.recall_conflicts_windowed_net)} screened order(s) carry "
                "an affirmative administration note within the dispense->return "
                "window not explained by another same-admission order (possible "
                "hidden transfusion)"
            )
        if result.invariant_violations:
            reasons.append(f"{len(result.invariant_violations)} invariant violation(s)")
        return (
            "  Do NOT flip RETURNS_LEDGER_ENABLED on yet. "
            + "; ".join(reasons)
            + ". Adjudicate the flagged orders with the blood bank and request a "
            "guaranteed-complete ledger before go-live, or NARROW the screen to "
            "exclude the flagged reqnos."
        )
    if result.recommendation == "NARROW":
        return (
            "  Recall is clean and the invariant holds, but "
            f"{len(result.reissue_findings)} screened order(s) show a reissue "
            "signal. Consider going live only on the count-agreeing screened "
            "orders (exclude the reissue reqnos) pending a complete export."
        )
    return (
        "  All checks clean on this bundle. Safe to flip the flag for a FRESH "
        "pilot run id (per spec §G) after clinician sign-off; re-baseline "
        "attribution scorecards on the enabled run."
    )


def _write_artifact(result: PreflightResult) -> Path:
    out = Path(
        os.environ.get(
            "BBA_PREFLIGHT_OUT", str(WORK / "preflight_returns_validation.json")
        )
    )
    payload = {
        **asdict(result),
        "signoff_summary": _signoff_text(result),
        "periop_min_ebl_ml": PERIOP_MIN_EBL_ML,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    result = run_preflight()
    print_report(result)
    artifact = _write_artifact(result)
    print(f"\nMachine-readable artifact written to {artifact}")


if __name__ == "__main__":
    main()
