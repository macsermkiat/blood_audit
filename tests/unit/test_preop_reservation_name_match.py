"""Tier-1 deterministic operation-name matcher tests for ticket #187.

All fixtures are fully synthetic: operation names, recommendations, and Thai
paren shapes are invented here and never copied from the pilot procedure text.
Tests drive the ``_index_from_rows`` construction seam (mirroring the reference
loader's seam); one smoke test loads the real packaged CSV.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from bba.preop_reservation.evaluate import _decide_from_row
from bba.preop_reservation.models import MsbosRow
from bba.preop_reservation.name_match import (
    OperationNameIndex,
    _index_from_rows,
    load_operation_name_index,
    match_operation_names,
    verify_proposed_operation,
    would_be_verdict,
)
from bba.preop_reservation.reference import load_msbos_reference


def _row(
    operation: str, *, msbos: str = "G/M", recommended_units: str = "2"
) -> dict[str, str]:
    return {
        "operation": operation,
        "msbos": msbos,
        "recommended_units": recommended_units,
    }


def _index(rows: Sequence[Mapping[str, str]]) -> OperationNameIndex:
    return _index_from_rows(rows, content_hash="a" * 64)


# --- forward matching + word-boundary negatives -----------------------------


def test_forward_needle_word_bounded_match_resolves_recommendation() -> None:
    index = _index([_row("Splenectomy", recommended_units="3")])

    result = match_operation_names(index, ["Planned: splenectomy tomorrow"])

    assert result.status == "matched"
    assert result.matched_operations == ("Splenectomy",)
    assert result.recommendation == MsbosRow(msbos="G/M", recommended_units=3)
    assert result.distinct_recommendation_count == 1
    assert result.matched_event_name == "Planned: splenectomy tomorrow"


def test_forward_needle_inside_larger_word_does_not_match() -> None:
    # "Arthro" must not match as a substring of "arthroscopy"; the normalizer
    # pads on word boundaries so only whole-token needles hit.
    index = _index([_row("Arthro")])

    result = match_operation_names(index, ["Knee arthroscopy performed"])

    assert result.status == "no_match", (
        "a needle must not match across a word boundary inside a larger word"
    )


# --- reverse matching + the >=2-word guard ----------------------------------


def test_multiword_event_reverse_matches_a_longer_full_operation_name() -> None:
    index = _index(
        [_row("Radical nephrectomy with thrombectomy", recommended_units="4")]
    )

    result = match_operation_names(index, ["radical nephrectomy"])

    assert result.status == "matched"
    assert result.matched_operations == ("Radical nephrectomy with thrombectomy",)
    assert result.recommendation == MsbosRow(msbos="G/M", recommended_units=4)


def test_single_word_event_does_not_reverse_match_longer_operation() -> None:
    # Acceptance: single-word "Thrombectomy" must NOT reverse-match
    # "Radical nephrectomy with thrombectomy".
    index = _index(
        [_row("Radical nephrectomy with thrombectomy", recommended_units="4")]
    )

    result = match_operation_names(index, ["Thrombectomy"])

    assert result.status == "no_match", (
        "a single-word event must not reverse-match a longer specific operation"
    )


# --- acronym vs modifier parens ---------------------------------------------


def test_real_acronym_paren_is_a_standalone_needle() -> None:
    index = _index(
        [_row("Coronary artery bypass grafting (CABG)", recommended_units="2")]
    )

    result = match_operation_names(index, ["Patient booked for CABG"])

    assert result.status == "matched"
    assert result.matched_operations == ("Coronary artery bypass grafting (CABG)",)


def test_semantic_modifier_paren_is_never_a_standalone_needle() -> None:
    # "(tumor)" is a semantic modifier, so an event mentioning only "tumor"
    # must never hit the row via the parenthesized content.
    index = _index([_row("Nephrectomy (tumor)", recommended_units="3")])

    result = match_operation_names(index, ["Removal of tumor from flank"])

    assert result.status == "no_match", (
        "a semantic modifier in parens must never become a standalone match key"
    )


def test_exact_redo_event_matches_only_the_redo_row() -> None:
    # "(redo)" is a semantic modifier (not acronym-like), so it is not a
    # standalone needle; only the exact full-name event resolves the redo row,
    # and the plain row is not reverse-hit because "gastrectomy redo" is not a
    # sub-phrase of the plain "gastrectomy" full name.
    index = _index(
        [
            _row("Gastrectomy", msbos="none", recommended_units="0"),
            _row("Gastrectomy (redo)", msbos="G/M", recommended_units="2"),
        ]
    )

    result = match_operation_names(index, ["Gastrectomy (redo)"])

    assert result.status == "matched"
    assert result.matched_operations == ("Gastrectomy (redo)",)
    assert result.recommendation == MsbosRow(msbos="G/M", recommended_units=2)


def test_paren_stripped_variant_preserves_semantic_modifier() -> None:
    # Regression (Codex P2): an operation with BOTH an acronym paren and a
    # semantic-modifier paren must strip ONLY the acronym paren for its
    # paren-stripped needle. The modifier stays, so a generic one-word event
    # cannot smuggle into the modified operation via the stripped variant.
    index = _index(
        [_row("Hepatectomy (major) (ABC)", msbos="G/M", recommended_units="4")]
    )

    # The acronym itself still matches.
    assert match_operation_names(index, ["Booked for ABC"]).status == "matched"
    # A bare one-word "Hepatectomy" must NOT resolve via a stripped needle that
    # dropped the "(major)" modifier.
    assert match_operation_names(index, ["Hepatectomy"]).status == "no_match", (
        "the paren-stripped variant must keep the semantic modifier so a generic "
        "one-word event cannot match the modified operation"
    )
    # An event that includes the modifier does resolve.
    assert (
        match_operation_names(index, ["Hepatectomy major planned"]).status == "matched"
    )


# --- needle-collision determinism -------------------------------------------


def test_needle_collision_is_row_order_independent() -> None:
    # Two operations share the acronym needle "(XYZ)" but carry the SAME
    # recommendation; building the index in either row order must behave
    # identically (collision represented as a set, never dict-overwritten).
    rows = [
        _row("Alpha procedure (XYZ)", msbos="G/M", recommended_units="2"),
        _row("Beta procedure (XYZ)", msbos="G/M", recommended_units="2"),
    ]
    forward = match_operation_names(_index(rows), ["Scheduled XYZ"])
    reverse = match_operation_names(_index(list(reversed(rows))), ["Scheduled XYZ"])

    assert forward == reverse
    assert forward.status == "matched"
    assert forward.matched_operations == (
        "Alpha procedure (XYZ)",
        "Beta procedure (XYZ)",
    ), "a shared acronym needle must report ALL colliding operations"


# --- conflict fail-closed vs same-recommendation collapse -------------------


def test_different_recommendation_collision_fails_closed() -> None:
    # The acronym "(ZZZ)" collides across two DIFFERENT recommendations -> the
    # matcher must surface a conflict rather than guess.
    index = _index(
        [
            _row("Op one (ZZZ)", msbos="G/M", recommended_units="2"),
            _row("Op two (ZZZ)", msbos="G/M", recommended_units="4"),
        ]
    )

    result = match_operation_names(index, ["Booked for ZZZ"])

    assert result.status == "conflicting_recommendations"
    assert result.recommendation is None
    assert result.distinct_recommendation_count == 2
    assert result.matched_operations == ("Op one (ZZZ)", "Op two (ZZZ)")


def test_same_recommendation_multi_operation_collapse_reports_all_names() -> None:
    index = _index(
        [
            _row("Left donor nephrectomy", msbos="G/M", recommended_units="1"),
            _row("Right donor nephrectomy", msbos="G/M", recommended_units="1"),
        ]
    )

    result = match_operation_names(
        index, ["Left donor nephrectomy", "Right donor nephrectomy"]
    )

    assert result.status == "matched"
    assert result.distinct_recommendation_count == 1
    assert result.matched_operations == (
        "Left donor nephrectomy",
        "Right donor nephrectomy",
    ), "same-recommendation collapse must still list all matched operations"


# --- Thai <-> Thai determinism and English-only ORIF conflict ---------------

# Synthetic mirror of the real reference shape: two rows with the SAME English
# stem, differing ONLY in Thai paren content, with DIFFERENT recommendations.
_THAI_YES = "ORIF synthetic bone (ก ได้)"
_THAI_NO = "ORIF synthetic bone (ก ไม่ได้)"


def _thai_index() -> OperationNameIndex:
    return _index(
        [
            _row(_THAI_YES, msbos="none", recommended_units="0"),
            _row(_THAI_NO, msbos="T/S", recommended_units="1"),
        ]
    )


def test_thai_paren_event_resolves_to_exactly_one_variant() -> None:
    result = match_operation_names(_thai_index(), [_THAI_YES])

    assert result.status == "matched"
    assert result.matched_operations == (_THAI_YES,)
    assert result.recommendation == MsbosRow(msbos="none", recommended_units=0), (
        "an event carrying the exact Thai string resolves to exactly one variant"
    )


def test_english_only_stem_reverse_matches_both_thai_variants_and_conflicts() -> None:
    # The English-only event "ORIF synthetic bone" is a sub-phrase of BOTH full
    # Thai names (>=2 words), reverse-matching both -> two distinct
    # recommendations (none/0 vs T/S/0) -> must fail closed.
    result = match_operation_names(_thai_index(), ["ORIF synthetic bone"])

    assert result.status == "conflicting_recommendations"
    assert set(result.matched_operations) == {_THAI_YES, _THAI_NO}
    assert result.distinct_recommendation_count == 2, (
        "an English-only stem cannot distinguish the Thai-differentiated variants"
    )


# --- blank-code row matchable by name ---------------------------------------


def test_blank_code_row_is_indexed_and_matchable_by_name() -> None:
    # The code index drops blank-code rows; the name index keeps them. A
    # "Complex EVAR"-shaped synthetic blank-code row must be matchable by name.
    index = _index(
        [{"operation": "Complex EVAR", "msbos": "G/M", "recommended_units": "2"}]
    )

    result = match_operation_names(index, ["Complex EVAR planned"])

    assert result.status == "matched"
    assert result.matched_operations == ("Complex EVAR",)
    assert result.recommendation == MsbosRow(msbos="G/M", recommended_units=2)


# --- T/S units zeroed before uniqueness -------------------------------------


def test_type_and_screen_units_zeroed_before_uniqueness_collapse() -> None:
    # A T/S-1 row and a T/S-2 row for two matched operations must collapse to
    # ONE recommendation (units zeroed pre-uniqueness), not a spurious conflict.
    index = _index(
        [
            _row("Alpha screen op", msbos="T/S", recommended_units="1"),
            _row("Beta screen op", msbos="T/S", recommended_units="2"),
        ]
    )

    result = match_operation_names(index, ["Alpha screen op", "Beta screen op"])

    assert result.status == "matched"
    assert result.distinct_recommendation_count == 1
    assert result.recommendation == MsbosRow(msbos="T/S", recommended_units=0), (
        "T/S recommended_units must be normalized to 0 before the uniqueness check"
    )
    assert result.matched_operations == ("Alpha screen op", "Beta screen op")


# --- would-be parity with the code-resolved path ----------------------------


def test_would_be_verdict_matches_code_resolved_decision_for_same_row() -> None:
    row = MsbosRow(msbos="G/M", recommended_units=2)

    produced = would_be_verdict(row=row, reserved_units=4, reference_hash="h" * 64)
    expected = _decide_from_row(
        row=row,
        reserved_units=4,
        resolved_icd9="",
        reference_hash="h" * 64,
        note_resolved=False,
    )

    assert produced == expected
    assert produced.is_over is True
    assert produced.reason == "over_gm_excess"
    assert produced.resolved_icd9 == "", "a name match carries no ICD-9 code"
    assert produced.note_resolved is False


def test_would_be_verdict_type_and_screen_over_when_units_reserved() -> None:
    row = MsbosRow(msbos="T/S", recommended_units=0)

    produced = would_be_verdict(row=row, reserved_units=2, reference_hash="h" * 64)

    assert produced.is_over is True
    assert produced.reason == "over_type_and_screen_crossmatched"


# --- verification helper (Tier-2) -------------------------------------------


def test_verification_accepts_exact_full_name_membership() -> None:
    index = _index([_row("Whipple procedure", recommended_units="3")])

    result = verify_proposed_operation(index, "whipple  procedure")

    assert result.accepted is True
    assert result.operation == "Whipple procedure"
    assert result.recommendation == MsbosRow(msbos="G/M", recommended_units=3)


def test_verification_rejects_near_miss_and_non_member() -> None:
    index = _index([_row("Whipple procedure", recommended_units="3")])

    assert verify_proposed_operation(index, "Whipple procedures").accepted is False
    assert verify_proposed_operation(index, "Totally unrelated op").accepted is False


def test_verification_rejects_acronym_only_string() -> None:
    # The LLM is told to copy an exact operation NAME; an acronym variant is not
    # a full operation name, so an acronym-only proposal is rejected.
    index = _index([_row("Coronary artery bypass grafting (CABG)")])

    assert verify_proposed_operation(index, "CABG").accepted is False


def test_verification_rejects_ambiguous_full_name_collision() -> None:
    # Two operations normalize to the SAME full name but carry DIFFERENT
    # recommendations -> exact membership is ambiguous -> reject.
    index = _index(
        [
            _row("Repair defect", msbos="G/M", recommended_units="2"),
            _row("Repair  defect", msbos="G/M", recommended_units="4"),
        ]
    )

    assert verify_proposed_operation(index, "Repair defect").accepted is False


# --- longest-match-wins subphrase disqualification --------------------------


def test_longest_match_wins_disqualifies_subphrase_within_one_event() -> None:
    # Within one event name, a needle that is a proper sub-phrase of another
    # matched needle for that event is disqualified, so the generic short
    # operation does not fire when the specific long operation is present.
    index = _index(
        [
            _row("Bypass", msbos="none", recommended_units="0"),
            _row("Femoral bypass graft", msbos="G/M", recommended_units="2"),
        ]
    )

    result = match_operation_names(index, ["Femoral bypass graft done"])

    assert result.status == "matched"
    assert result.matched_operations == ("Femoral bypass graft",), (
        "the shorter 'Bypass' needle is disqualified as a sub-phrase of the "
        "longer matched operation within the same event"
    )
    assert result.recommendation == MsbosRow(msbos="G/M", recommended_units=2)


# --- exact forward match preserved vs reverse-only longer op (P1 fix, #191) --


def test_exact_event_not_reassigned_to_longer_reverse_only_operation() -> None:
    # Regression (Codex P1): an event that exactly names a shorter operation is
    # a forward (present-in-event) match; a longer operation that only reverse-
    # matches (the event is a fragment of it) must NOT cannibalize it. With a
    # different recommendation the two must fail closed as a conflict, never
    # silently resolve to the longer op. Mirrors the packaged
    # "Radical nephrectomy" (G/M 2) / "... with thrombectomy" (G/M 4) pair.
    index = _index(
        [
            _row("Excise organ", msbos="G/M", recommended_units="2"),
            _row("Excise organ with graft", msbos="G/M", recommended_units="4"),
        ]
    )

    result = match_operation_names(index, ["Excise organ"])

    assert result.status == "conflicting_recommendations", (
        "an exact forward match must not be cannibalized by a reverse-only "
        "longer operation carrying a different recommendation"
    )
    assert set(result.matched_operations) == {
        "Excise organ",
        "Excise organ with graft",
    }
    assert result.distinct_recommendation_count == 2


def test_exact_event_reverse_longer_same_recommendation_collapses() -> None:
    # When the exact shorter op and the reverse-matched longer op share ONE
    # recommendation, retaining both is a benign collapse: still matched, all
    # names reported.
    index = _index(
        [
            _row("Excise organ", msbos="G/M", recommended_units="2"),
            _row("Excise organ with graft", msbos="G/M", recommended_units="2"),
        ]
    )

    result = match_operation_names(index, ["Excise organ"])

    assert result.status == "matched"
    assert result.distinct_recommendation_count == 1
    assert set(result.matched_operations) == {
        "Excise organ",
        "Excise organ with graft",
    }


# --- duplicate identical operation names fail closed (P1 fix, #191) ----------


def test_identical_operation_name_with_conflicting_recs_fails_closed() -> None:
    # Regression (Codex P1): two rows with the EXACT same operation string but
    # different recommendations must keep BOTH (not silently keep the first), so
    # the name is ambiguous, matching fails closed, and the outcome is
    # independent of row order.
    rows = [
        _row("Duplicate operation", msbos="G/M", recommended_units="2"),
        _row("Duplicate operation", msbos="G/M", recommended_units="4"),
    ]
    forward = match_operation_names(_index(rows), ["Duplicate operation"])
    reverse = match_operation_names(
        _index(list(reversed(rows))), ["Duplicate operation"]
    )

    assert forward.status == "conflicting_recommendations"
    assert forward.distinct_recommendation_count == 2
    assert forward == reverse, "duplicate-name resolution must be row-order-independent"
    assert (
        verify_proposed_operation(_index(rows), "Duplicate operation").accepted is False
    ), "an ambiguous duplicate name must also be rejected by the verification helper"


# --- no-match on empty / whitespace events ----------------------------------


def test_empty_and_whitespace_events_yield_no_match() -> None:
    index = _index([_row("Splenectomy")])

    assert match_operation_names(index, []).status == "no_match"
    assert match_operation_names(index, ["", "   "]).status == "no_match"


# --- packaged-CSV smoke test ------------------------------------------------

_BLANK_CODE_OPERATIONS = (
    "Complex EVAR",
    "Endovascular aortic aneurysm repair (EVAR)",
    "Endovascular intervention : normal case",
    "Endovascular intervention : risk for rupture",
    "Ext, no tourniquet",
    "Extremity with tourniquet",
    "Hip and spine",
)


def test_packaged_index_indexes_blank_code_operations_and_matches_reference_hash() -> (
    None
):
    index = load_operation_name_index()
    operations = index.operations()

    for operation in _BLANK_CODE_OPERATIONS:
        assert operation in operations, (
            f"blank-code operation {operation!r} must be indexed by name "
            "(the code index drops it)"
        )

    # "Complex EVAR" is a blank-code row; it must be matchable by name.
    result = match_operation_names(index, ["Complex EVAR"])
    assert result.status == "matched"
    assert "Complex EVAR" in result.matched_operations

    # Regression (Codex P1) against the REAL schedule: an event that exactly
    # names "Radical nephrectomy" (G/M 2) must NOT be silently reassigned to
    # "Radical nephrectomy with thrombectomy" (G/M 4); the differing
    # recommendations must fail closed as a conflict.
    nephrectomy = match_operation_names(index, ["Radical nephrectomy"])
    assert nephrectomy.status == "conflicting_recommendations", (
        "an exact 'Radical nephrectomy' event must not resolve to the longer "
        "thrombectomy recommendation"
    )
    assert {
        "Radical nephrectomy",
        "Radical nephrectomy with thrombectomy",
    } <= set(nephrectomy.matched_operations)

    assert index.content_hash == load_msbos_reference().content_hash, (
        "the name index reads the same bytes as the reference loader, so their "
        "content hashes must be identical"
    )
