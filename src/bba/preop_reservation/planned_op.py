"""MSBOS-local planned-op picker v2 (ticket #198, spec #196).

Pure re-derivation of the planned-operation pick from the existing
``Sequence[OperativeEvent]`` plus the OPRTACT->ICD9 bridge, WITHOUT touching
the shared operative-event builder (the frozen preflight / name-study / timing
consumers all share it). Inert library module: deliberately NOT exported from
``bba.preop_reservation.__init__`` — the wiring ticket consumes it directly
(the ``name_match`` precedent).

Design (advisor-frozen):

* Horizon is classified on the RAW future events BEFORE any exclusion, so a
  denied in-window candidate yields ``all_candidates_excluded`` — never
  ``outside_window``.
* Denylist + resolved-code exclusions are skip-and-continue: ranking proceeds
  over the survivors, never converting a denial into "no plan".
* Ambiguity uses the ``(or_flag, operative_datetime)`` equivalence key ONLY;
  ``resolved_icd9`` picks a deterministic presentation candidate strictly
  AFTER ambiguity is decided.
* The near-simultaneous cluster rule only ever triggers review
  (``ambiguous_top_rank``); it never re-ranks the winner. The fuller
  operation-cluster ranking rule is deferred by user ruling (2026-07-18).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, TypeVar

from pydantic import BaseModel

from bba.cohort_detector.models import OperativeEvent
from bba.preop_reservation.bridge import OprtactBridge
from bba.preop_reservation.models import PlannedOpProvenance
from bba.preop_reservation.reference import MsbosReference

PLANNED_OP_WINDOW_HOURS = 72
"""Inclusive pre-op horizon, hours (parity with PRE_OP_CROSSMATCH_WINDOW_HOURS)."""

PLANNED_OP_CLUSTER_WINDOW_SECONDS = 60
"""Near-simultaneous charge cluster window around the rank-1 candidate,
seconds, absolute and inclusive. Ancillary billing charges land 1-3 s from the
real operation, so distinct MSBOS-eligible codes inside this window route to
review instead of auto-picking a possibly-ancillary charge."""

PLANNED_OP_SOURCE_CODE_DENYLIST: frozenset[str] = frozenset(
    {
        "AS056",
        "AS058",
        "MD529",
        "MD530",
        "SU062",
        "P0001",
        "P0067",
        "EY619",
        "P0093",
        "P1395",
    }
)
"""Reviewed pilot non-operative billing codes (anesthesia, dialysis,
fluoroscopy, ETT, intralesional injection, pupil dilation, spinal taps).
Exact codes only — never prefixes. Additions require clinician review."""

_SENTINEL_PREFIX = "INCPT:"

AMBIGUOUS_PLANNED_OP_SENTINEL = "\x00AMBIG"
"""Legacy ambiguity sentinel: both reservation evaluators map it to
``ambiguous_planned_op`` (the platelet evaluator directly; the RBC callers via
their existing sentinel branch)."""

BRIDGE_HARD_VERDICT_MIN_SCORE = 0.95
"""Minimum First-Choice score for a bridge-sourced over to keep a hard
verdict (clinician-tunable, spec #196 §2d). Below it — or without human
agreement — the over routes to NEEDS_REVIEW instead."""

PickStatus = Literal[
    "selected",
    "selected_blank_code",
    "no_future_event",
    "outside_window",
    "all_candidates_excluded",
    "ambiguous_top_rank",
]

PickSource = Literal["icd9", "incpt_bridge"]


@dataclass(frozen=True, slots=True)
class PlannedOpPick:
    """Fully-populated planned-op selection result with bridge provenance."""

    resolved_icd9: str
    source_code: str
    source: PickSource | None
    bridge_score: float | None
    human_index: str | None
    human_agreed: bool | None
    human_icd9: str | None
    or_flag: bool | None
    matched_datetime: datetime | None
    pick_status: PickStatus
    candidate_count: int
    tie_count: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    resolved_icd9: str
    source_code: str
    source: PickSource | None
    bridge_score: float | None
    human_index: str | None
    human_agreed: bool | None
    human_icd9: str | None
    event: OperativeEvent


def _failure(status: PickStatus) -> PlannedOpPick:
    return PlannedOpPick(
        resolved_icd9="",
        source_code="",
        source=None,
        bridge_score=None,
        human_index=None,
        human_agreed=None,
        human_icd9=None,
        or_flag=None,
        matched_datetime=None,
        pick_status=status,
        candidate_count=0,
        tie_count=0,
    )


def _nodot(code: str) -> str:
    """Strip decimal points ONLY — never add or remove leading zeros."""
    return code.replace(".", "")


def _derive(event: OperativeEvent, bridge: OprtactBridge) -> _Candidate:
    code = event.icd9.strip()
    if code.startswith(_SENTINEL_PREFIX):
        source_code = code[len(_SENTINEL_PREFIX) :]
        entry = bridge.get(source_code)
        if entry is None:
            return _Candidate(
                resolved_icd9=code,
                source_code=source_code,
                source=None,
                bridge_score=None,
                human_index=None,
                human_agreed=None,
                human_icd9=None,
                event=event,
            )
        return _Candidate(
            resolved_icd9=entry.icd9,
            source_code=source_code,
            source="incpt_bridge",
            bridge_score=entry.score,
            human_index=entry.human_index,
            human_agreed=entry.human_agreed,
            human_icd9=entry.human_icd9,
            event=event,
        )
    return _Candidate(
        resolved_icd9=code,
        source_code=code,
        source="icd9",
        bridge_score=None,
        human_index=None,
        human_agreed=None,
        human_icd9=None,
        event=event,
    )


def _rank_key(candidate: _Candidate) -> tuple[bool, datetime]:
    return (not candidate.event.or_flag, candidate.event.operative_datetime)


def _presentation_key(candidate: _Candidate) -> tuple[str, str, str]:
    return (
        candidate.resolved_icd9,
        candidate.source_code,
        candidate.source or "",
    )


def _result(
    candidate: _Candidate,
    *,
    status: PickStatus,
    candidate_count: int,
    tie_count: int,
) -> PlannedOpPick:
    return PlannedOpPick(
        resolved_icd9=candidate.resolved_icd9,
        source_code=candidate.source_code,
        source=candidate.source,
        bridge_score=candidate.bridge_score,
        human_index=candidate.human_index,
        human_agreed=candidate.human_agreed,
        human_icd9=candidate.human_icd9,
        or_flag=candidate.event.or_flag,
        matched_datetime=candidate.event.operative_datetime,
        pick_status=status,
        candidate_count=candidate_count,
        tie_count=tie_count,
    )


def select_planned_op_v2(
    events: Sequence[OperativeEvent],
    order_datetime: datetime,
    bridge: OprtactBridge,
    *,
    denylist: frozenset[str] = PLANNED_OP_SOURCE_CODE_DENYLIST,
    msbos_codes: Collection[str],
    approved_non_blood_codes: Collection[str],
    cluster_window_seconds: int = PLANNED_OP_CLUSTER_WINDOW_SECONDS,
) -> PlannedOpPick:
    """Select the planned operation for one order (pure; tz-aware UTC inputs).

    ``msbos_codes`` is the dotless-code universe of the MSBOS reference;
    ``approved_non_blood_codes`` is the clinician-approved non-blood procedure
    set (callers pass ``NON_BLOOD_PROCEDURE_ICD9``). Both are supplied by the
    caller to keep this function pure.
    """
    future = [e for e in events if e.operative_datetime >= order_datetime]
    if not future:
        return _failure("no_future_event")
    window = timedelta(hours=PLANNED_OP_WINDOW_HOURS)
    in_window = [e for e in future if e.operative_datetime - order_datetime <= window]
    if not in_window:
        return _failure("outside_window")

    survivors = [
        candidate
        for candidate in (_derive(event, bridge) for event in in_window)
        if candidate.source_code not in denylist
        and _nodot(candidate.resolved_icd9) not in approved_non_blood_codes
    ]
    if not survivors:
        return _failure("all_candidates_excluded")

    candidate_count = len(survivors)
    rank1_key = min(_rank_key(candidate) for candidate in survivors)
    rank1 = [c for c in survivors if _rank_key(c) == rank1_key]
    tie_count = len(rank1)
    rank1_dt = rank1[0].event.operative_datetime

    ambiguous = len({c.resolved_icd9 for c in rank1}) > 1
    if not ambiguous:
        cluster_window = timedelta(seconds=cluster_window_seconds)
        cluster_codes = {
            _nodot(c.resolved_icd9)
            for c in survivors
            if abs(c.event.operative_datetime - rank1_dt) <= cluster_window
            and _nodot(c.resolved_icd9) in msbos_codes
        }
        ambiguous = len(cluster_codes) > 1

    winner = min(rank1, key=_presentation_key)
    if ambiguous:
        return _result(
            winner,
            status="ambiguous_top_rank",
            candidate_count=candidate_count,
            tie_count=tie_count,
        )
    status: PickStatus = (
        "selected_blank_code" if winner.resolved_icd9 == "" else "selected"
    )
    return _result(
        winner,
        status=status,
        candidate_count=candidate_count,
        tie_count=tie_count,
    )


def planned_op_v2_for_events(
    events: Sequence[OperativeEvent],
    order_datetime: datetime,
    *,
    bridge: OprtactBridge,
    msbos_codes: Collection[str],
    approved_non_blood_codes: Collection[str],
) -> tuple[str, PlannedOpPick]:
    """Shared leg seam: pick, plus the evaluator-facing planned-code string.

    Maps the pick onto the exact string contract both reservation evaluators
    already speak, so a flag-ON leg swaps only the picker, never the
    evaluation flow: ambiguity -> the legacy ambiguity sentinel; failure
    statuses -> ``""`` (no plan); otherwise the resolved code (which may be an
    unresolvable ``INCPT:`` sentinel, yielding ``unresolved_code`` exactly as
    the legacy picker's sentinel did).
    """
    pick = select_planned_op_v2(
        events,
        order_datetime,
        bridge,
        msbos_codes=msbos_codes,
        approved_non_blood_codes=approved_non_blood_codes,
    )
    if pick.pick_status == "ambiguous_top_rank":
        return AMBIGUOUS_PLANNED_OP_SENTINEL, pick
    return pick.resolved_icd9, pick


_DecisionT = TypeVar("_DecisionT", bound=BaseModel)


def _bridge_gate(
    pick: PlannedOpPick, *, is_over: bool, reference: MsbosReference
) -> Literal["", "bridge_disagreement", "bridge_over_unconfirmed"]:
    """Verdict-gate ruling for one pick (spec #196 §2d).

    Only bridge-sourced, non-ambiguous picks are ever gated (an ambiguous
    pick already routes to review on its status, so its provenance must not
    also assert a gate). The disagreement guard fires BEFORE the score gate.
    """
    if pick.source != "incpt_bridge" or pick.pick_status == "ambiguous_top_rank":
        return ""
    human_icd9 = pick.human_icd9 or ""
    if human_icd9 and human_icd9 != pick.resolved_icd9:
        first_hits = reference.resolve(_nodot(pick.resolved_icd9)) is not None
        human_hits = reference.resolve(_nodot(human_icd9)) is not None
        if first_hits or human_hits:
            return "bridge_disagreement"
    if is_over:
        score = pick.bridge_score if pick.bridge_score is not None else 0.0
        confirmed = score >= BRIDGE_HARD_VERDICT_MIN_SCORE and bool(pick.human_agreed)
        if not confirmed:
            return "bridge_over_unconfirmed"
    return ""


def attach_planned_op(
    decision: _DecisionT,
    pick: PlannedOpPick,
    *,
    reference: MsbosReference,
    bridge_hash: str,
) -> _DecisionT:
    """Attach picker-v2 provenance (including the gate ruling) to a decision.

    Works for both the RBC ``ReservationDecision`` and the platelet
    ``PlateletReservationDecision`` — both carry the optional ``planned_op``
    field and expose ``is_over``. The raw ``is_over`` judgment is preserved
    for audit; the verdict overlay consults ``planned_op.gate`` instead.
    """
    is_over = bool(getattr(decision, "is_over", False))
    provenance = PlannedOpProvenance(
        source_code=pick.source_code,
        source=pick.source or "",
        bridge_icd9=pick.resolved_icd9 if pick.source == "incpt_bridge" else "",
        bridge_score=pick.bridge_score,
        human_index=pick.human_index or "",
        human_agreed=pick.human_agreed,
        human_icd9=pick.human_icd9 or "",
        pick_status=pick.pick_status,
        candidate_count=pick.candidate_count,
        tie_count=pick.tie_count,
        bridge_hash=bridge_hash,
        gate=_bridge_gate(pick, is_over=is_over, reference=reference),
    )
    return decision.model_copy(update={"planned_op": provenance})


__all__ = [
    "AMBIGUOUS_PLANNED_OP_SENTINEL",
    "BRIDGE_HARD_VERDICT_MIN_SCORE",
    "PLANNED_OP_CLUSTER_WINDOW_SECONDS",
    "PLANNED_OP_SOURCE_CODE_DENYLIST",
    "PLANNED_OP_WINDOW_HOURS",
    "PickSource",
    "PickStatus",
    "PlannedOpPick",
    "attach_planned_op",
    "planned_op_v2_for_events",
    "select_planned_op_v2",
]
