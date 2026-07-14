"""Unit tests for the pure returns-ledger aggregator (spec #119, ticket #120).

``summarize_returns`` is the highest-value test seam: it encodes the disposition
rules that later tickets route on, and the fail-closed completeness contract that
keeps a partial ledger export from silently marking a transfused order as
not-transfused. Each test states WHY the disposition matters, not just the count.
"""

from __future__ import annotations

from bba.returns_ledger import ReturnsSummary, summarize_returns


def _unit(status: str, **extra: str) -> dict[str, str]:
    """One BDVSTTRANS unit row with an UPPERCASE ``UNITSTAT`` (as normalized).

    Rows carry no ``DNRNO`` by default, so each row is its own physical unit
    (identifier-free fallback). Tests that exercise lifecycle-row collapsing
    pass an explicit ``DNRNO``/``SEQNO``.
    """
    return {"UNITSTAT": status, **extra}


def test_all_returned_is_not_transfused() -> None:
    # Every dispensed unit came back to the blood bank -> no blood entered the
    # patient, so the order must resolve to not_transfused (the core false
    # attribution this feature fixes).
    summary = summarize_returns([_unit("3"), _unit("3")], ["2"])

    assert summary.ledger_complete is True
    assert summary.units_total == 2
    assert summary.units_returned == 2
    assert summary.units_transfused == 0
    assert summary.disposition == "not_transfused"


def test_all_returned_over_dispensed_is_not_transfused() -> None:
    # An over-dispensed all-returned order (more distinct units returned than
    # ordered — a reissue) genuinely had ALL its units returned, so with a
    # guaranteed-complete ledger the patient received nothing: it screens
    # not_transfused. This RELAXES the earlier NARROW exact-count guard, which
    # only existed to hedge against a hidden transfused replacement unit on a
    # PARTIAL export (spec #119 complete-ledger go-live). Each row carries a
    # distinct DNRNO so they are three physical units, not one collapsed unit.
    summary = summarize_returns(
        [
            _unit("3", DNRNO="A", SEQNO="0"),
            _unit("3", DNRNO="B", SEQNO="0"),
            _unit("3", DNRNO="C", SEQNO="0"),
        ],
        ["2"],
    )

    assert summary.units_total == 3
    assert summary.ordered_unit_amount == 2
    assert summary.ledger_complete is True
    assert summary.disposition == "not_transfused"


def test_transfused_over_dispensed_stays_transfused() -> None:
    # The exact-count requirement is SCOPED to the not-transfused screen. An
    # over-dispensed order with a non-returned unit already confirms a
    # transfusion; over-dispense there is benign clinical top-up, not a hidden-
    # transfusion risk. It stays transfused (preserving its peri-op exemption /
    # normal judgment), NOT demoted to inconclusive (spec #119 NARROW: the guard
    # must not strip PERIOP_TRANSFUSION_EXEMPT from peri-op transfusions).
    summary = summarize_returns([_unit("3"), _unit("2"), _unit("2")], ["2"])

    assert summary.units_total == 3
    assert summary.ordered_unit_amount == 2
    assert summary.ledger_complete is True
    assert summary.disposition == "transfused"


def test_one_non_returned_unit_is_transfused() -> None:
    # A single unit that was not returned (here dispensed, Unitstat=2) means
    # blood was given: not-returned == transfused (reverses #105 conservatism).
    summary = summarize_returns([_unit("3"), _unit("2")], ["2"])

    assert summary.disposition == "transfused"
    assert summary.units_returned == 1


def test_any_transfused_status_is_transfused() -> None:
    # Unitstat=5 is the strongest positive signal; its presence confirms a
    # transfusion regardless of the other units.
    summary = summarize_returns([_unit("5"), _unit("3")], ["2"])

    assert summary.units_transfused == 1
    assert summary.disposition == "transfused"


def test_returned_keys_on_status_not_return_date() -> None:
    # A transfused unit can also carry a stray return timestamp. "Returned" must
    # key on Unitstat=3, never on RTNDATE presence, or a real transfusion would
    # be miscounted as returned.
    summary = summarize_returns([_unit("5", RTNDATE="January 1, 2026")], ["1"])

    assert summary.units_returned == 0
    assert summary.units_transfused == 1
    assert summary.disposition == "transfused"


def test_returned_status_without_return_date_still_counts_returned() -> None:
    summary = summarize_returns([_unit("3")], ["1"])

    assert summary.units_returned == 1
    assert summary.disposition == "not_transfused"


def test_mixed_counts_are_exact() -> None:
    summary = summarize_returns([_unit("3"), _unit("3"), _unit("5"), _unit("2")], ["4"])

    assert summary.units_total == 4
    assert summary.units_returned == 2
    assert summary.units_transfused == 1
    assert summary.ordered_unit_amount == 4
    assert summary.disposition == "transfused"


def test_empty_ledger_is_inconclusive() -> None:
    # No ledger rows -> we cannot assert a disposition; fall through to the
    # existing pipeline rather than guess.
    summary = summarize_returns([], ["1"])

    assert summary.units_total == 0
    assert summary.ledger_complete is False
    assert summary.disposition == "inconclusive"


def test_ledger_below_ordered_quantity_is_inconclusive() -> None:
    # The export does not fully account for the ordered units, so it must not be
    # trusted to mark the order not-transfused.
    summary = summarize_returns([_unit("3")], ["2"])

    assert summary.ledger_complete is False
    assert summary.disposition == "inconclusive"


def test_blank_unitamt_line_fails_closed() -> None:
    # One unparseable ordered-quantity line makes the WHOLE order inconclusive;
    # the ordered amount must not be computed from only the parseable lines.
    summary = summarize_returns([_unit("3"), _unit("3")], ["1", ""])

    assert summary.ordered_unit_amount is None
    assert summary.ledger_complete is False
    assert summary.disposition == "inconclusive"


def test_non_numeric_unitamt_fails_closed() -> None:
    summary = summarize_returns([_unit("3")], ["x"])

    assert summary.ordered_unit_amount is None
    assert summary.disposition == "inconclusive"


def test_zero_unitamt_fails_closed() -> None:
    summary = summarize_returns([_unit("3")], ["0"])

    assert summary.ordered_unit_amount is None
    assert summary.disposition == "inconclusive"


def test_negative_unitamt_fails_closed() -> None:
    summary = summarize_returns([_unit("3")], ["-1"])

    assert summary.ordered_unit_amount is None
    assert summary.disposition == "inconclusive"


def test_missing_unitamt_lines_fails_closed() -> None:
    # No ordered-quantity lines at all -> ordered amount unknown -> inconclusive.
    summary = summarize_returns([_unit("3")], [])

    assert summary.ordered_unit_amount is None
    assert summary.ledger_complete is False
    assert summary.disposition == "inconclusive"


def test_dispense_then_return_of_one_unit_collapses_to_returned() -> None:
    # The complete export records a unit's lifecycle as two rows sharing
    # (DNRNO, SEQNO): a dispense (Unitstat=2) and a later return (Unitstat=3).
    # These are ONE physical unit, not two; collapsed to its terminal status it
    # is returned, so a one-unit order reads not_transfused. Without dedup the
    # dispense row would inflate units_total to 2 and mislabel the order
    # transfused (the safe-direction bug the complete-ledger ingest fixes).
    summary = summarize_returns(
        [
            _unit("2", DNRNO="10068045545", SEQNO="0"),
            _unit("3", DNRNO="10068045545", SEQNO="0"),
        ],
        ["1"],
    )

    assert summary.units_total == 1
    assert summary.units_returned == 1
    assert summary.disposition == "not_transfused"


def test_aliquots_with_different_seqno_are_distinct_units() -> None:
    # Split aliquots of one donor bag share DNRNO but differ by SEQNO; they are
    # DISTINCT physical units and must not be collapsed. Two returned aliquots of
    # a 2-unit order read not_transfused.
    summary = summarize_returns(
        [
            _unit("3", DNRNO="10068045545", SEQNO="0"),
            _unit("3", DNRNO="10068045545", SEQNO="1"),
        ],
        ["2"],
    )

    assert summary.units_total == 2
    assert summary.units_returned == 2
    assert summary.disposition == "not_transfused"


def test_transfused_wins_over_return_row_for_same_unit() -> None:
    # A unit carrying both a transfused (5) and a return (3) row was given; the
    # terminal precedence keeps it transfused so a real transfusion is never
    # downgraded to returned (fail safe against a false not_transfused clear).
    summary = summarize_returns(
        [
            _unit("5", DNRNO="U1", SEQNO="0"),
            _unit("3", DNRNO="U1", SEQNO="0"),
        ],
        ["1"],
    )

    assert summary.units_total == 1
    assert summary.units_transfused == 1
    assert summary.units_returned == 0
    assert summary.disposition == "transfused"


def test_incompatible_unit_is_not_transfused() -> None:
    # Unitstat=7 is crossmatch-incompatible: the unit was never given. An order
    # whose only units are returned and incompatible received no blood, so it
    # screens not_transfused (spec #119 complete-ledger: status 7 counts toward
    # the not-transfused screen, not toward transfusion).
    summary = summarize_returns(
        [
            _unit("3", DNRNO="U1", SEQNO="0"),
            _unit("7", DNRNO="U2", SEQNO="0"),
        ],
        ["2"],
    )

    assert summary.units_total == 2
    assert summary.units_returned == 1
    assert summary.units_incompat == 1
    assert summary.disposition == "not_transfused"


def test_all_incompatible_is_not_transfused() -> None:
    # Every unit was incompatible and none given: no transfusion from this order.
    summary = summarize_returns([_unit("7", DNRNO="U1", SEQNO="0")], ["1"])

    assert summary.units_incompat == 1
    assert summary.disposition == "not_transfused"


def test_incompatible_plus_dispensed_is_transfused() -> None:
    # A dispensed-not-returned unit (Unitstat=2) is presumed given; the presence
    # of a separate incompatible unit does not clear it. The order is transfused.
    summary = summarize_returns(
        [
            _unit("7", DNRNO="U1", SEQNO="0"),
            _unit("2", DNRNO="U2", SEQNO="0"),
        ],
        ["2"],
    )

    assert summary.units_incompat == 1
    assert summary.disposition == "transfused"


def test_summary_is_immutable() -> None:
    summary = summarize_returns([_unit("3")], ["1"])

    assert isinstance(summary, ReturnsSummary)
    try:
        summary.units_total = 99  # type: ignore[misc]
    except Exception:  # noqa: BLE001 - frozen model raises; exact type is pydantic-internal
        return
    raise AssertionError("ReturnsSummary must be frozen")
