"""build_review post-flip MSBOS rendering tests for ticket #201.

A declared row that MSBOS reclassifies to PREOP_OVER_RESERVATION or
NEEDS_REVIEW must keep its MSBOS detail (pills, counts, case line, with the
picker provenance surfaced), and the glossary must no longer claim the
annotation never changes classification.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"


def _load_build_review(module_name: str) -> ModuleType:
    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))
    spec = importlib.util.spec_from_file_location(
        module_name, PILOT_DIR / "build_review.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def review() -> ModuleType:
    return _load_build_review("pilot_build_review_picker_v2")


def _over_row(classification: str, **extra: str) -> dict[str, str]:
    return {
        "classification": classification,
        "msbos_reason": "over_gm_excess",
        "msbos_reserved_units": "5",
        "msbos_recommended_units": "2",
        "msbos_token": "G/M",
        **extra,
    }


# --- summary pill across post-flip classes ------------------------------------


def test_summary_pill_renders_for_post_flip_classes(review: ModuleType) -> None:
    for classification in ("PREOP_OVER_RESERVATION", "NEEDS_REVIEW"):
        pill = review._msbos_summary_pill(_over_row(classification))
        assert "5 vs G/M 2" in pill, classification


def test_summary_pill_still_renders_for_returns_terminals(
    review: ModuleType,
) -> None:
    pill = review._msbos_summary_pill(_over_row("PERIOP_TRANSFUSION_EXEMPT"))
    assert "5 vs G/M 2" in pill


def test_summary_pill_stays_silent_without_msbos_data(review: ModuleType) -> None:
    assert (
        review._msbos_summary_pill(
            {"classification": "NEEDS_REVIEW", "msbos_reason": ""}
        )
        == "—"
    )
    assert (
        review._msbos_summary_pill(
            {"classification": "APPROPRIATE", "msbos_reason": "over_gm_excess"}
        )
        == "—"
    )


def test_render_classes_cover_returns_and_post_flip(review: ModuleType) -> None:
    assert review._MSBOS_RENDER_CLASSES == frozenset(
        {
            "RETURNED_NOT_TRANSFUSED",
            "PERIOP_TRANSFUSION_EXEMPT",
            "PREOP_OVER_RESERVATION",
            "NEEDS_REVIEW",
        }
    )


# --- case line picker provenance ----------------------------------------------


def test_case_line_surfaces_bridge_pick_provenance(review: ModuleType) -> None:
    line = review._msbos_case_line(
        _over_row(
            "PREOP_OVER_RESERVATION",
            msbos_op_pick_status="selected",
            msbos_source_code="P0614",
            msbos_bridge_icd9="3611",
            msbos_bridge_score="1.0",
            msbos_bridge_human_agreed="False",
        ),
        "PREOP_OVER_RESERVATION",
    )

    assert "Reserved 5; MSBOS tariff G/M 2" in line
    assert "pick selected via P0614" in line
    assert "3611" in line
    assert "human-disagreed" in line


def test_case_line_without_picker_columns_is_unchanged(review: ModuleType) -> None:
    line = review._msbos_case_line(
        _over_row("PERIOP_TRANSFUSION_EXEMPT"), "PERIOP_TRANSFUSION_EXEMPT"
    )

    assert line == "Reserved 5; MSBOS tariff G/M 2"


# --- dominance ceiling rendering (#210/#214) ----------------------------------


def _ceiling_row(reason: str, **extra: str) -> dict[str, str]:
    return {
        "classification": "PREOP_OVER_RESERVATION",
        "msbos_reason": reason,
        "msbos_reserved_units": "5",
        "msbos_ceiling_basis": "G/M 2 (4573,8151)",
        "msbos_op_pick_status": "ambiguous_top_rank",
        **extra,
    }


def test_over_ceiling_pill_and_case_line(review: ModuleType) -> None:
    row = _ceiling_row("over_ceiling")

    pill = review._msbos_summary_pill(row)
    line = review._msbos_case_line(row, "PREOP_OVER_RESERVATION")

    # Pill shows the short token+units; the case line carries the full basis.
    assert "over ceiling G/M 2" in pill
    assert "(4573,8151)" not in pill
    assert "over ceiling G/M 2 (4573,8151)" in line


def test_within_ceiling_pill_and_case_line(review: ModuleType) -> None:
    row = dict(
        _ceiling_row("within_ceiling"), classification="PERIOP_TRANSFUSION_EXEMPT"
    )

    pill = review._msbos_summary_pill(row)
    line = review._msbos_case_line(row, "PERIOP_TRANSFUSION_EXEMPT")

    assert "within ceiling G/M 2" in pill
    assert "within ceiling G/M 2 (4573,8151)" in line


def test_within_ceiling_is_its_own_count_bucket(review: ModuleType) -> None:
    assert review._msbos_reason_bucket("within_ceiling") == "within_ceiling"
    assert review._msbos_reason_bucket("over_ceiling") == "above"


def test_case_line_non_bridge_pick_shows_status_only(review: ModuleType) -> None:
    line = review._msbos_case_line(
        _over_row(
            "NEEDS_REVIEW",
            msbos_op_pick_status="ambiguous_top_rank",
            msbos_source_code="741",
            msbos_bridge_icd9="",
        ),
        "NEEDS_REVIEW",
    )

    assert "pick ambiguous_top_rank via 741" in line
    assert "human-" not in line


# --- counts across post-flip classes ------------------------------------------


def test_counts_do_not_crash_on_post_flip_classes(
    review: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression (first live flag-ON render): msbos_counts was initialized
    # for the returns terminals only, so a NEEDS_REVIEW row carrying msbos
    # data raised KeyError in the count pass.
    from test_pilot_build_review import _render_review_with_rows

    rendered = _render_review_with_rows(
        review,
        tmp_path,
        monkeypatch,
        manifest_csv="HN,REQNO,AN\nHN1,R1,AN1\nHN2,R2,AN2\n",
        report_csv=(
            "reqno,classification,rationale,component,msbos_reason,"
            "msbos_reserved_units,msbos_recommended_units,msbos_token,"
            "msbos_is_over,msbos_resolved_icd9,msbos_reference_hash\n"
            "R1,NEEDS_REVIEW,preop_reservation_bridge_disagreement,red_cell,"
            "unresolved_code,1,0,,False,INCPT:PX001,hash\n"
            "R2,PREOP_OVER_RESERVATION,preop_over_reservation,red_cell,"
            "over_gm_excess,5,2,G/M,True,8151,hash\n"
        ),
        llm_json="[]",
        msbos_enabled=True,
    ).decode()

    assert (
        "Over-reserved (1): 1 above / 0 within / 0 within-ceiling / 0 unresolved"
        in rendered
    )
    assert (
        "MSBOS review (1): 0 above / 0 within / 0 within-ceiling / 1 unresolved"
        in rendered
    )


# --- glossary ----------------------------------------------------------------


def test_glossary_no_longer_claims_annotation_never_reclassifies(
    review: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_pilot_build_review import _render_empty_review

    html = _render_empty_review(
        review, tmp_path, monkeypatch, returns_enabled=True, msbos_enabled=True
    ).decode("utf-8")

    assert "does not change the order's classification or scoring" not in html
    assert "MSBOS screening CAN change the classification" in html


def test_glossary_picker_entries_gated_on_picker_seam(
    review: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_pilot_build_review import _render_empty_review

    monkeypatch.setattr(review, "MSBOS_PLANNED_OP_PICKER_V2_PILOT_ENABLED", False)
    without = _render_empty_review(
        review,
        tmp_path / "off",
        monkeypatch,
        returns_enabled=True,
        msbos_enabled=True,
    ).decode("utf-8")
    monkeypatch.setattr(review, "MSBOS_PLANNED_OP_PICKER_V2_PILOT_ENABLED", True)
    with_picker = _render_empty_review(
        review,
        tmp_path / "on",
        monkeypatch,
        returns_enabled=True,
        msbos_enabled=True,
    ).decode("utf-8")

    assert "preop_reservation_bridge_disagreement" not in without
    assert "preop_reservation_bridge_disagreement" in with_picker
    assert "preop_over_reservation_bridge_unconfirmed" in with_picker
