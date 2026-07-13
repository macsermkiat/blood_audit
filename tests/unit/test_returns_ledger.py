"""Unit tests for the pure returns-ledger aggregator (spec #119, ticket #120).

``summarize_returns`` is the highest-value test seam: it encodes the disposition
rules that later tickets route on, and the fail-closed completeness contract that
keeps a partial ledger export from silently marking a transfused order as
not-transfused. Each test states WHY the disposition matters, not just the count.
"""

from __future__ import annotations

from bba.returns_ledger import ReturnsSummary, summarize_returns


def _unit(status: str, **extra: str) -> dict[str, str]:
    """One BDVSTTRANS unit row with an UPPERCASE ``UNITSTAT`` (as normalized)."""
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


def test_all_returned_over_dispensed_is_inconclusive() -> None:
    # units_total exceeding the ordered amount is a visible reissue / over-
    # dispense. The ledger still COVERS the order (ledger_complete stays True),
    # but the not-transfused screen additionally requires an EXACT count match:
    # a partial export could be hiding a transfused replacement unit that no
    # count-based guard can see. So an over-dispensed all-returned order falls
    # through to normal judgment instead of being screened not_transfused
    # (spec #119 NARROW go-live decision).
    summary = summarize_returns([_unit("3"), _unit("3"), _unit("3")], ["2"])

    assert summary.units_total == 3
    assert summary.ordered_unit_amount == 2
    assert summary.ledger_complete is True
    assert summary.disposition == "inconclusive"


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


def test_summary_is_immutable() -> None:
    summary = summarize_returns([_unit("3")], ["1"])

    assert isinstance(summary, ReturnsSummary)
    try:
        summary.units_total = 99  # type: ignore[misc]
    except Exception:  # noqa: BLE001 - frozen model raises; exact type is pydantic-internal
        return
    raise AssertionError("ReturnsSummary must be frozen")
