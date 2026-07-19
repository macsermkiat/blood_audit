"""MSBOS-local planned-op picker v2 tests for ticket #198.

All fixtures are synthetic: events, bridge rows, and code sets are invented
here. The picker is pure, so every branch is driven directly through
``select_planned_op_v2`` with UTC-aware datetimes.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from bba.cohort_detector.models import OperativeEvent
from bba.preop_reservation.bridge import OprtactBridge, _bridge_from_rows
from bba.preop_reservation.planned_op import (
    PLANNED_OP_CLUSTER_WINDOW_SECONDS,
    PLANNED_OP_SOURCE_CODE_DENYLIST,
    PLANNED_OP_SOURCE_CODE_DENYLIST_PREFIXES,
    PLANNED_OP_WINDOW_HOURS,
    PlannedOpPick,
    select_planned_op_v2,
)

_ORDER_DT = datetime(2026, 3, 1, 8, 0, 0, tzinfo=UTC)


def _event(
    icd9: str,
    *,
    hours: float = 1.0,
    seconds: float = 0.0,
    or_flag: bool = False,
    name: str | None = None,
) -> OperativeEvent:
    return OperativeEvent(
        icd9=icd9,
        or_flag=or_flag,
        operative_datetime=_ORDER_DT + timedelta(hours=hours, seconds=seconds),
        name=name,
    )


def _bridge_row(
    oprtact: str,
    *,
    icd9: str = "3725",
    score: str = "0.90",
    human_index: str = "1",
    human_icd9: str = "",
) -> dict[str, str]:
    return {
        "oprtact": oprtact,
        "icd9": icd9,
        "icd9_nodot": icd9.replace(".", ""),
        "score": score,
        "human_index": human_index,
        "human_agreed": "true" if human_index == "0" else "false",
        "human_icd9": human_icd9,
        "name": f"Synthetic op {oprtact}",
    }


def _bridge(rows: Sequence[dict[str, str]] = ()) -> OprtactBridge:
    return _bridge_from_rows(rows, content_hash="b" * 64)


def _pick(
    events: Sequence[OperativeEvent],
    *,
    bridge: OprtactBridge | None = None,
    msbos_codes: frozenset[str] = frozenset(),
    approved: frozenset[str] = frozenset(),
    denylist: frozenset[str] = PLANNED_OP_SOURCE_CODE_DENYLIST,
    denylist_prefixes: frozenset[str] = PLANNED_OP_SOURCE_CODE_DENYLIST_PREFIXES,
    cluster_window_seconds: int = PLANNED_OP_CLUSTER_WINDOW_SECONDS,
) -> PlannedOpPick:
    return select_planned_op_v2(
        events,
        _ORDER_DT,
        bridge if bridge is not None else _bridge(),
        denylist=denylist,
        denylist_prefixes=denylist_prefixes,
        msbos_codes=msbos_codes,
        approved_non_blood_codes=approved,
        cluster_window_seconds=cluster_window_seconds,
    )


def _assert_failure(pick: PlannedOpPick, status: str) -> None:
    assert pick.pick_status == status
    assert pick.resolved_icd9 == ""
    assert pick.source_code == ""
    assert pick.source is None
    assert pick.bridge_score is None
    assert pick.human_index is None
    assert pick.human_agreed is None
    assert pick.or_flag is None
    assert pick.matched_datetime is None
    assert pick.candidate_count == 0
    assert pick.tie_count == 0


# --- candidate derivation ----------------------------------------------------


def test_sentinel_bridge_hit_resolves_with_provenance() -> None:
    bridge = _bridge([_bridge_row("P0937", icd9="554", score="0.97", human_index="0")])

    pick = _pick([_event("INCPT:P0937")], bridge=bridge)

    assert pick.pick_status == "selected"
    assert pick.resolved_icd9 == "554"
    assert pick.source_code == "P0937"
    assert pick.source == "incpt_bridge"
    assert pick.bridge_score == 0.97
    assert pick.human_index == "0"
    assert pick.human_agreed is True
    assert pick.matched_datetime == _ORDER_DT + timedelta(hours=1)
    assert pick.candidate_count == 1
    assert pick.tie_count == 1


def test_sentinel_bridge_miss_stays_unresolvable() -> None:
    pick = _pick([_event("INCPT:SSC109")], bridge=_bridge())

    assert pick.pick_status == "selected"
    assert pick.resolved_icd9 == "INCPT:SSC109"
    assert pick.source_code == "SSC109"
    assert pick.source is None
    assert pick.bridge_score is None
    assert pick.human_agreed is None


def test_non_sentinel_event_passes_through() -> None:
    pick = _pick([_event("8151")])

    assert pick.pick_status == "selected"
    assert pick.resolved_icd9 == "8151"
    assert pick.source_code == "8151"
    assert pick.source == "icd9"
    assert pick.bridge_score is None


# --- denylist + resolved-code exclusion (skip-and-continue) ------------------


def test_denylisted_source_code_skips_to_later_real_op() -> None:
    bridge = _bridge([_bridge_row("P1247", icd9="8151", score="0.99", human_index="0")])
    events = [
        _event("INCPT:MD529", hours=1.0),
        _event("INCPT:P1247", hours=2.0),
    ]

    pick = _pick(events, bridge=bridge)

    assert pick.pick_status == "selected"
    assert pick.resolved_icd9 == "8151"
    assert pick.source_code == "P1247"
    assert pick.candidate_count == 1


def test_resolved_code_in_approved_non_blood_set_is_skipped() -> None:
    bridge = _bridge(
        [
            _bridge_row("MDX01", icd9="3895", score="0.99", human_index="0"),
            _bridge_row("P1247", icd9="8151", score="0.99", human_index="0"),
        ]
    )
    events = [
        _event("INCPT:MDX01", hours=1.0),
        _event("INCPT:P1247", hours=2.0),
    ]

    pick = _pick(events, bridge=bridge, approved=frozenset({"3895"}))

    assert pick.pick_status == "selected"
    assert pick.resolved_icd9 == "8151"


def test_leading_zero_codes_are_not_conflated() -> None:
    # Bridge resolves to "331" (lobectomy family); the approved non-blood set
    # holds "0331" (lumbar puncture). Dot-stripping only must NOT exclude it.
    bridge = _bridge([_bridge_row("PX331", icd9="331", score="0.99", human_index="0")])

    pick = _pick([_event("INCPT:PX331")], bridge=bridge, approved=frozenset({"0331"}))

    assert pick.pick_status == "selected"
    assert pick.resolved_icd9 == "331"


# --- horizon (bound-first attribution) ---------------------------------------


def test_event_exactly_at_order_time_is_kept() -> None:
    pick = _pick([_event("8151", hours=0.0)])

    assert pick.pick_status == "selected"
    assert pick.matched_datetime == _ORDER_DT


def test_event_exactly_at_72h_is_kept() -> None:
    pick = _pick([_event("8151", hours=float(PLANNED_OP_WINDOW_HOURS))])

    assert pick.pick_status == "selected"


def test_event_just_past_72h_is_outside_window() -> None:
    pick = _pick([_event("8151", hours=float(PLANNED_OP_WINDOW_HOURS), seconds=1.0)])

    _assert_failure(pick, "outside_window")


def test_past_only_events_are_no_future_event() -> None:
    _assert_failure(_pick([_event("8151", hours=-2.0)]), "no_future_event")


def test_empty_events_are_no_future_event() -> None:
    _assert_failure(_pick([]), "no_future_event")


def test_in_window_denied_plus_outside_window_clean_is_all_excluded() -> None:
    # The 68016696 discriminator: horizon classification precedes exclusion,
    # so a denied in-window candidate yields all_candidates_excluded, never
    # outside_window.
    events = [
        _event("INCPT:MD530", hours=1.0),
        _event("8151", hours=100.0),
    ]

    pick = _pick(events)

    _assert_failure(pick, "all_candidates_excluded")


def test_all_in_window_candidates_denylisted_is_all_excluded() -> None:
    _assert_failure(_pick([_event("INCPT:AS058")]), "all_candidates_excluded")


# --- ranking + exact-tie ambiguity -------------------------------------------


def test_or_flagged_event_outranks_earlier_non_or_event() -> None:
    events = [
        _event("1234", hours=1.0, or_flag=False),
        _event("8151", hours=48.0, or_flag=True),
    ]

    pick = _pick(events)

    assert pick.pick_status == "selected"
    assert pick.resolved_icd9 == "8151"
    assert pick.or_flag is True
    assert pick.candidate_count == 2
    assert pick.tie_count == 1


def test_exact_tie_distinct_codes_is_ambiguous_with_stable_presentation() -> None:
    events = [
        _event("6561", hours=1.0),
        _event("544", hours=1.0),
    ]

    forward = _pick(events)
    reversed_pick = _pick(list(reversed(events)))

    assert forward.pick_status == "ambiguous_top_rank"
    assert forward.tie_count == 2
    assert forward.candidate_count == 2
    # Deterministic presentation candidate regardless of input order.
    assert forward.resolved_icd9 == "544"
    assert reversed_pick == forward


def test_exact_tie_same_code_selects_with_tie_count() -> None:
    events = [
        _event("8151", hours=1.0),
        _event("8151", hours=1.0),
    ]

    pick = _pick(events)

    assert pick.pick_status == "selected"
    assert pick.resolved_icd9 == "8151"
    assert pick.tie_count == 2
    assert pick.candidate_count == 2


# --- near-simultaneous cluster review ----------------------------------------


def test_cluster_with_two_distinct_msbos_codes_is_ambiguous() -> None:
    events = [
        _event("7915", hours=1.0),
        _event("8151", hours=1.0, seconds=30.0),
    ]

    pick = _pick(events, msbos_codes=frozenset({"7915", "8151"}))

    assert pick.pick_status == "ambiguous_top_rank"
    assert pick.tie_count == 1
    assert pick.candidate_count == 2


def test_cluster_with_single_msbos_code_selects_rank_one() -> None:
    # Only one distinct MSBOS-eligible code in the cluster: no review trigger,
    # and NO winner change (cluster-ranking is deferred by user ruling) — the
    # rank-1 candidate wins even though the MSBOS code sits second.
    events = [
        _event("598", hours=1.0),
        _event("8151", hours=1.0, seconds=30.0),
    ]

    pick = _pick(events, msbos_codes=frozenset({"8151"}))

    assert pick.pick_status == "selected"
    assert pick.resolved_icd9 == "598"


def test_cluster_earlier_than_or_winner_also_triggers() -> None:
    # The rank-1 candidate is an OR event that occurs LATER than a nearby
    # non-OR event; the absolute cluster window still captures the earlier one.
    events = [
        _event("8151", hours=1.0, seconds=30.0, or_flag=True),
        _event("7915", hours=1.0, or_flag=False),
    ]

    pick = _pick(events, msbos_codes=frozenset({"7915", "8151"}))

    assert pick.pick_status == "ambiguous_top_rank"


def test_cluster_boundary_inclusive_at_window_exclusive_past() -> None:
    at_window = [
        _event("7915", hours=1.0),
        _event("8151", hours=1.0, seconds=float(PLANNED_OP_CLUSTER_WINDOW_SECONDS)),
    ]
    past_window = [
        _event("7915", hours=1.0),
        _event(
            "8151", hours=1.0, seconds=float(PLANNED_OP_CLUSTER_WINDOW_SECONDS) + 1.0
        ),
    ]
    codes = frozenset({"7915", "8151"})

    assert _pick(at_window, msbos_codes=codes).pick_status == "ambiguous_top_rank"
    assert _pick(past_window, msbos_codes=codes).pick_status == "selected"


def test_cluster_ignores_unresolvable_sentinels() -> None:
    events = [
        _event("8151", hours=1.0),
        _event("INCPT:SSC109", hours=1.0, seconds=10.0),
    ]

    pick = _pick(events, msbos_codes=frozenset({"8151"}))

    assert pick.pick_status == "selected"
    assert pick.resolved_icd9 == "8151"


# --- blank codes -------------------------------------------------------------


def test_blank_code_winner_is_selected_blank_code() -> None:
    pick = _pick([_event("")])

    assert pick.pick_status == "selected_blank_code"
    assert pick.resolved_icd9 == ""
    assert pick.source_code == ""
    assert pick.source == "icd9"
    assert pick.candidate_count == 1
    assert pick.tie_count == 1


# --- purity ------------------------------------------------------------------


def test_events_input_is_not_mutated_and_result_is_deterministic() -> None:
    events = [
        _event("8151", hours=2.0),
        _event("INCPT:AS056", hours=1.0),
        _event("7915", hours=50.0),
    ]
    snapshot = list(events)

    first = _pick(events, msbos_codes=frozenset({"8151"}))
    shuffled = _pick([events[2], events[0], events[1]], msbos_codes=frozenset({"8151"}))

    assert events == snapshot
    assert first == shuffled
    assert first.pick_status == "selected"
    assert first.resolved_icd9 == "8151"
    # AS056 is denylisted; 8151 (+2h) and 7915 (+50h) both survive in-window.
    assert first.candidate_count == 2


# --- guarded-prefix denylist (families AS/ML/X + exact SU030, #211) ----------


def test_prefix_denied_non_msbos_code_is_skipped() -> None:
    # An AS* anesthesia charge bridging to a non-MSBOS code is excluded by the
    # prefix family (AS999 is NOT in the exact denylist, so this exercises the
    # PREFIX path); a later real operation is selected instead.
    bridge = _bridge(
        [
            _bridge_row("AS999", icd9="3895", score="0.99", human_index="0"),
            _bridge_row("P1247", icd9="8151", score="0.99", human_index="0"),
        ]
    )
    events = [
        _event("INCPT:AS999", hours=1.0),
        _event("INCPT:P1247", hours=2.0),
    ]

    pick = _pick(events, bridge=bridge, msbos_codes=frozenset({"8151"}))

    assert pick.pick_status == "selected"
    assert pick.resolved_icd9 == "8151"
    assert pick.source_code == "P1247"
    assert pick.candidate_count == 1


def test_prefix_denied_but_msbos_mapped_code_survives_crossover_guard() -> None:
    # Zero-MSBOS-crossover invariant: a prefixed source code whose RESOLVED code
    # IS in the MSBOS universe must NEVER be suppressed by the prefix rule — a
    # prefix can never hide a verdict-capable operation.
    bridge = _bridge([_bridge_row("AS999", icd9="8151", score="0.99", human_index="0")])

    pick = _pick(
        [_event("INCPT:AS999")], bridge=bridge, msbos_codes=frozenset({"8151"})
    )

    assert pick.pick_status == "selected"
    assert pick.resolved_icd9 == "8151"
    assert pick.source_code == "AS999"


def test_ml_and_x_prefix_families_are_denied_when_non_msbos() -> None:
    for oprtact in ("ML001", "X042"):
        bridge = _bridge(
            [
                _bridge_row(oprtact, icd9="3895", score="0.99", human_index="0"),
                _bridge_row("P1247", icd9="8151", score="0.99", human_index="0"),
            ]
        )
        events = [
            _event(f"INCPT:{oprtact}", hours=1.0),
            _event("INCPT:P1247", hours=2.0),
        ]

        pick = _pick(events, bridge=bridge, msbos_codes=frozenset({"8151"}))

        assert pick.pick_status == "selected", oprtact
        assert pick.resolved_icd9 == "8151", oprtact


def test_su030_exact_code_is_denied_even_when_msbos_mapped() -> None:
    # SU030 is an EXACT denial (SU is NOT a safe prefix family): it is suppressed
    # unconditionally, even though it bridges to an MSBOS-mapped code here.
    bridge = _bridge(
        [
            _bridge_row("SU030", icd9="8151", score="0.99", human_index="0"),
            _bridge_row("P1247", icd9="5122", score="0.99", human_index="0"),
        ]
    )
    events = [
        _event("INCPT:SU030", hours=1.0),
        _event("INCPT:P1247", hours=2.0),
    ]

    pick = _pick(events, bridge=bridge, msbos_codes=frozenset({"8151", "5122"}))

    assert pick.pick_status == "selected"
    assert pick.source_code == "P1247"
    assert pick.resolved_icd9 == "5122"


def test_prefixes_do_not_affect_non_prefixed_source_codes() -> None:
    # A plain sentinel code that does not start with a denied prefix is untouched.
    bridge = _bridge([_bridge_row("P1247", icd9="8151", score="0.99", human_index="0")])

    pick = _pick(
        [_event("INCPT:P1247")], bridge=bridge, msbos_codes=frozenset({"8151"})
    )

    assert pick.pick_status == "selected"
    assert pick.resolved_icd9 == "8151"
    assert pick.source_code == "P1247"


def test_all_in_window_prefix_denied_is_all_excluded() -> None:
    # Only a prefix-denied non-MSBOS candidate in-window -> all_candidates_excluded.
    bridge = _bridge([_bridge_row("AS999", icd9="3895", score="0.99", human_index="0")])

    pick = _pick(
        [_event("INCPT:AS999")], bridge=bridge, msbos_codes=frozenset({"8151"})
    )

    _assert_failure(pick, "all_candidates_excluded")
