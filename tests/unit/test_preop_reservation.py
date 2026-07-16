"""Reserved-unit keying and ingest availability tests for ticket #162."""

from __future__ import annotations

from bba.component_map import ComponentFamily
from bba.ingest.schemas import get_schema
from bba.preop_reservation import reserved_units_by_component


def test_reused_reqno_does_not_merge_across_hns() -> None:
    totals = reserved_units_by_component(
        [
            {"HN": "HN-A", "REQNO": "REUSED", "BDTYPE": "LPRC", "UNITAMT": "2"},
            {"HN": "HN-B", "REQNO": "REUSED", "BDTYPE": "LPRC", "UNITAMT": "5"},
        ]
    )

    assert totals[("HN-A", "REUSED", ComponentFamily.RED_CELL)] == 2
    assert totals[("HN-B", "REUSED", ComponentFamily.RED_CELL)] == 5


def test_component_totals_are_kept_separate() -> None:
    totals = reserved_units_by_component(
        [
            {"HN": "HN-A", "REQNO": "REQ-1", "BDTYPE": "LPRC", "UNITAMT": "2"},
            {"HN": "HN-A", "REQNO": "REQ-1", "BDTYPE": "PC", "UNITAMT": "4"},
        ]
    )

    assert totals[("HN-A", "REQ-1", ComponentFamily.RED_CELL)] == 2
    assert totals[("HN-A", "REQ-1", ComponentFamily.PLATELET)] == 4


def test_invalid_unit_amounts_are_skipped_per_line() -> None:
    rows = [
        {"HN": "HN-A", "REQNO": "REQ-1", "BDTYPE": "LPRC", "UNITAMT": "3"},
        {"HN": "HN-A", "REQNO": "REQ-1", "BDTYPE": "LPRC", "UNITAMT": ""},
        {"HN": "HN-A", "REQNO": "REQ-1", "BDTYPE": "LPRC", "UNITAMT": "bad"},
        {"HN": "HN-A", "REQNO": "REQ-1", "BDTYPE": "LPRC", "UNITAMT": "0"},
        {"HN": "HN-A", "REQNO": "REQ-1", "BDTYPE": "LPRC", "UNITAMT": "-2"},
        {"HN": "HN-B", "REQNO": "REQ-2", "BDTYPE": "PC", "UNITAMT": " "},
    ]

    totals = reserved_units_by_component(rows)

    assert totals[("HN-A", "REQ-1", ComponentFamily.RED_CELL)] == 3
    assert totals[("HN-B", "REQ-2", ComponentFamily.PLATELET)] == 0


def test_result_is_deterministic_across_input_reordering() -> None:
    rows = [
        {"HN": "HN-B", "REQNO": "REQ-2", "BDTYPE": "PC", "UNITAMT": "1"},
        {"HN": "HN-A", "REQNO": "REQ-1", "BDTYPE": "LPRC", "UNITAMT": "2"},
        {"HN": "HN-A", "REQNO": "REQ-1", "BDTYPE": "LPRC", "UNITAMT": "3"},
    ]

    forward = reserved_units_by_component(rows)
    reversed_result = reserved_units_by_component(tuple(reversed(rows)))

    assert forward == reversed_result
    assert tuple(forward.items()) == tuple(reversed_result.items())


def test_bdvst_status_code_and_lookup_schemas_are_available() -> None:
    assert "BDVSTST" in get_schema("BDVST").columns
    assert {"BDVSTST", "NAME"}.issubset(get_schema("BDVSTST").columns)
