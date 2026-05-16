"""RED-phase failing tests for issue #6 (bba.vitals_extractor).

Each ``class`` maps to one acceptance criterion in the issue body. Tests
assert contracts (the WHY), not implementation choices — see PRD §"Testing
Decisions".

No implementation exists yet; every test MUST fail in this scaffold commit.
The module-level imports double as the public-API surface check: if any
re-export is missing, collection fails before any test runs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bba.vitals_extractor import (
    BT_MAX,
    BT_MIN,
    DBP_MAX,
    DBP_MIN,
    HR_MAX,
    HR_MIN,
    RR_MAX,
    RR_MIN,
    SBP_MAX,
    SBP_MIN,
    LLMFallback,
    SourceProvenance,
    VitalSigns,
    VitalsFlag,
    VitalsNote,
    VitalsResult,
    extract_vitals,
    extract_vitals_from_text,
    is_bt_valid,
    is_dbp_valid,
    is_hr_valid,
    is_rr_valid,
    is_sbp_valid,
)


# Default order anchor used by the window-selection tests. The pipeline contract
# requires tz-aware UTC; constructing one here mirrors the ingest layer's output.
ANCHOR = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)


def _note(
    source: str,
    minutes: int,
    text: str,
    *,
    base: datetime = ANCHOR,
) -> VitalsNote:
    """Build a :class:`VitalsNote` ``minutes`` after ``base`` (negative = before)."""
    return VitalsNote(
        source=source,  # type: ignore[arg-type]  # narrowed by VitalsNote
        timestamp=base + timedelta(minutes=minutes),
        text=text,
    )


# =============================================================================
# AC: Regex patterns cover the canonical Thai + English vital formats.
# Issue body lists: BP 110/60, BP:118/63, PR108, HR 97, P 14, RR 20-23,
# BT 38.4, Temp 37. These are the literal forms the extractor MUST recognize.
# =============================================================================


class TestRegexExtractsBP:
    """Blood-pressure patterns — both space-separated and colon-attached."""

    def test_space_separated(self) -> None:
        v = extract_vitals_from_text("BP 110/60 PR 80 RR 16 BT 37")
        assert v.sbp == 110
        assert v.dbp == 60

    def test_colon_attached(self) -> None:
        v = extract_vitals_from_text("BP:118/63")
        assert v.sbp == 118
        assert v.dbp == 63

    def test_embedded_in_thai_prose(self) -> None:
        # Thai nursing prose surrounds the roman abbreviation; the regex must
        # not anchor on line-start or whitespace-only delimiters.
        v = extract_vitals_from_text("ผู้ป่วยรู้สึกตัวดี BP 120/70 ไม่มีอาการเหนื่อย")
        assert v.sbp == 120
        assert v.dbp == 70


class TestRegexExtractsHR:
    """Heart-rate patterns — both ``HR`` and ``PR`` (pulse rate)."""

    def test_pr_attached(self) -> None:
        # "PR108" — no space — appears in compressed nurse handwriting transcripts.
        v = extract_vitals_from_text("PR108")
        assert v.hr == 108

    def test_hr_space_separated(self) -> None:
        v = extract_vitals_from_text("HR 97")
        assert v.hr == 97

    def test_pr_and_hr_in_same_note_uses_one(self) -> None:
        # If both labels appear (rare), the extractor must return a single
        # populated value — not double-write, not raise.
        v = extract_vitals_from_text("BP 120/80 HR 90 PR 92 RR 16")
        assert v.hr is not None
        assert v.hr in (90, 92)


class TestRegexExtractsRR:
    """Respiratory-rate patterns — ``RR`` (canonical) and the bare ``P`` short form."""

    def test_p_short_form(self) -> None:
        # Issue body groups "P 14" with the RR examples; 14 is in the RR sanity
        # band (5-50) but not the HR band (30-200), so the only consistent
        # interpretation is RR.
        v = extract_vitals_from_text("BP 120/80 P 14")
        assert v.rr == 14

    def test_rr_simple(self) -> None:
        v = extract_vitals_from_text("RR 22")
        assert v.rr == 22

    def test_rr_range_takes_lower_bound(self) -> None:
        # "RR 20-23" expresses observed variability; the lower bound is the
        # deterministic floor (clinically interpretable as "at-or-above").
        v = extract_vitals_from_text("RR 20-23")
        assert v.rr == 20


class TestRegexExtractsBT:
    """Body-temperature patterns — ``BT`` and ``Temp``, integer or decimal."""

    def test_bt_decimal(self) -> None:
        v = extract_vitals_from_text("BT 38.4")
        assert v.bt == pytest.approx(38.4)

    def test_temp_integer(self) -> None:
        v = extract_vitals_from_text("Temp 37")
        assert v.bt == pytest.approx(37.0)

    def test_temp_case_insensitive(self) -> None:
        # Nursing notes mix "Temp", "temp", "TEMP"; case must not change the
        # recognition decision.
        v = extract_vitals_from_text("temp 36.8")
        assert v.bt == pytest.approx(36.8)


# =============================================================================
# AC: Sanity-bound enforcement; out-of-bound values discarded with
# `vitals_data_error` flag.
# =============================================================================


class TestBoundsConstants:
    """The published bounds match issue #6: SBP 60-220, DBP 30-150, HR 30-200,
    RR 5-50, BT 30-43. Hard-coding the literals here is a tripwire — silently
    widening a bound would break the auditor's published expectation."""

    def test_sbp_bounds(self) -> None:
        assert (SBP_MIN, SBP_MAX) == (60, 220)

    def test_dbp_bounds(self) -> None:
        assert (DBP_MIN, DBP_MAX) == (30, 150)

    def test_hr_bounds(self) -> None:
        assert (HR_MIN, HR_MAX) == (30, 200)

    def test_rr_bounds(self) -> None:
        assert (RR_MIN, RR_MAX) == (5, 50)

    def test_bt_bounds(self) -> None:
        assert (BT_MIN, BT_MAX) == (30.0, 43.0)


class TestBoundsPredicates:
    """Inclusive bound predicates accept the endpoints and reject just outside."""

    @pytest.mark.parametrize(
        ("v", "ok"), [(59, False), (60, True), (220, True), (221, False)]
    )
    def test_sbp(self, v: int, ok: bool) -> None:
        assert is_sbp_valid(v) is ok

    @pytest.mark.parametrize(
        ("v", "ok"), [(29, False), (30, True), (150, True), (151, False)]
    )
    def test_dbp(self, v: int, ok: bool) -> None:
        assert is_dbp_valid(v) is ok

    @pytest.mark.parametrize(
        ("v", "ok"), [(29, False), (30, True), (200, True), (201, False)]
    )
    def test_hr(self, v: int, ok: bool) -> None:
        assert is_hr_valid(v) is ok

    @pytest.mark.parametrize(
        ("v", "ok"), [(4, False), (5, True), (50, True), (51, False)]
    )
    def test_rr(self, v: int, ok: bool) -> None:
        assert is_rr_valid(v) is ok

    @pytest.mark.parametrize(
        ("v", "ok"),
        [(29.9, False), (30.0, True), (43.0, True), (43.1, False)],
    )
    def test_bt(self, v: float, ok: bool) -> None:
        assert is_bt_valid(v) is ok


class TestSanityBoundsEnforcement:
    """Out-of-bound regex hits MUST be discarded (set to None), and the result
    MUST carry the ``vitals_data_error`` flag so the auditor sees a tripwire
    rather than a silently-cleansed value."""

    def test_out_of_bound_sbp_is_none(self) -> None:
        # 999 is far out of the SBP band; regex finds the digits, bounds reject.
        v = extract_vitals_from_text("BP 999/80")
        assert v.sbp is None

    def test_out_of_bound_bt_missing_decimal(self) -> None:
        # "BT 384" is the canonical regression case for a missing decimal in
        # OCR (real "38.4" temperature). Out-of-bound → discarded.
        v = extract_vitals_from_text("BT 384")
        assert v.bt is None

    def test_pipeline_flags_data_error_on_discard(self) -> None:
        notes = [_note("IPDADMPROGRESS", -30, "BP 999/80 HR 80 RR 16 BT 37.0")]
        result = extract_vitals(anchor=ANCHOR, notes=notes)
        assert VitalsFlag.DATA_ERROR in result.flags
        assert result.vitals.sbp is None  # offending field is the one discarded
        assert result.vitals.hr == 80  # surviving fields stay populated

    def test_pipeline_no_data_error_when_all_in_bounds(self) -> None:
        notes = [_note("IPDADMPROGRESS", -30, "BP 120/70 HR 80 RR 16 BT 37.0")]
        result = extract_vitals(anchor=ANCHOR, notes=notes)
        assert VitalsFlag.DATA_ERROR not in result.flags


# =============================================================================
# Regression (codex review, 2026-05-15): overlong numeric tokens must NOT
# silently match their in-range prefix. "HR 2000" used to extract hr=200 with
# no vitals_data_error flag — sanity-bound enforcement was bypassed by a
# greedy regex truncation. The trailing (?!\d) closes that gap.
# =============================================================================


class TestRegexRejectsOverlongNumbers:
    """Each label-value regex must reject digit runs longer than its declared
    width — silent truncation would populate a wrong value with no DATA_ERROR
    flag, defeating the whole sanity-bounds contract."""

    def test_hr_four_digit_token_does_not_truncate(self) -> None:
        # "HR 2000" must NOT become hr=200; either reject or flag, never
        # silently populate.
        v = extract_vitals_from_text("HR 2000")
        assert v.hr is None

    def test_pr_four_digit_token_does_not_truncate(self) -> None:
        v = extract_vitals_from_text("PR 1500")
        assert v.hr is None

    def test_rr_three_digit_token_does_not_truncate(self) -> None:
        # "RR 500" must NOT become rr=50.
        v = extract_vitals_from_text("RR 500")
        assert v.rr is None

    def test_rr_range_overlong_second_does_not_truncate(self) -> None:
        # Range tail noise: "RR 20-2345" should still yield rr=20 from the
        # first group, but the range second digit must not consume "23" out
        # of "2345" (the regression case for noisy range parsing).
        v = extract_vitals_from_text("RR 20-2345")
        assert v.rr == 20

    def test_bt_overlong_integer_does_not_truncate(self) -> None:
        # "BT 3845" must NOT become bt=38.0 via greedy 2-digit fallback.
        v = extract_vitals_from_text("BT 3845")
        assert v.bt is None

    def test_bp_overlong_dbp_does_not_truncate(self) -> None:
        # "BP 120/1000" must NOT become dbp=100 (a clean in-bounds DBP) via
        # greedy 3-digit truncation of "1000". The "/" only protects SBP from
        # bleeding into DBP — DBP needs its own trailing-digit guard.
        v = extract_vitals_from_text("BP 120/1000")
        assert v.dbp is None

    def test_bp_overlong_sbp_does_not_truncate(self) -> None:
        # "BP 1234/80" with a 4-digit SBP must not match the regex at all
        # (no in-range SBP prefix can be followed by "/" because "4" is in
        # the way). Symmetric guarantee with HR/RR/BT.
        v = extract_vitals_from_text("BP 1234/80")
        assert v.sbp is None
        assert v.dbp is None


# =============================================================================
# AC: ±6 h window selection from anchor; most-recent-before-order preferred.
# =============================================================================


class TestWindowSelection:
    """Pipeline filters notes to +/-6 h and prefers the most-recent pre-anchor note."""

    def test_outside_window_ignored(self) -> None:
        # 7 hours before is outside the +/-6 h window — must not be chosen even
        # if it's the only candidate.
        far = _note("IPDADMPROGRESS", -60 * 7, "BP 120/70 HR 80 RR 16 BT 37.0")
        result = extract_vitals(anchor=ANCHOR, notes=[far])
        assert result.source == SourceProvenance.NONE_IN_WINDOW
        assert result.vitals == VitalSigns()

    def test_picks_most_recent_pre_anchor(self) -> None:
        old = _note("IPDADMPROGRESS", -300, "BP 100/60 HR 70 RR 14 BT 36.5")
        new = _note("IPDADMPROGRESS", -30, "BP 130/85 HR 95 RR 20 BT 37.5")
        result = extract_vitals(anchor=ANCHOR, notes=[old, new])
        # The 30-minute-old note wins over the 5-hour-old note.
        assert result.vitals.sbp == 130
        assert result.vitals.hr == 95
        assert result.note_timestamp == new.timestamp

    def test_input_order_does_not_matter(self) -> None:
        # Selection is deterministic in timestamp, not in input order.
        old = _note("IPDADMPROGRESS", -300, "BP 100/60 HR 70 RR 14 BT 36.5")
        new = _note("IPDADMPROGRESS", -30, "BP 130/85 HR 95 RR 20 BT 37.5")
        a = extract_vitals(anchor=ANCHOR, notes=[old, new])
        b = extract_vitals(anchor=ANCHOR, notes=[new, old])
        assert a.vitals == b.vitals
        assert a.note_timestamp == b.note_timestamp

    def test_at_anchor_counts_as_pre_anchor(self) -> None:
        # A note timestamped exactly at the anchor is the most-recent pre-anchor
        # observation, not a post-order fallback.
        at = _note("IPDADMPROGRESS", 0, "BP 118/72 HR 88 RR 18 BT 37.0")
        result = extract_vitals(anchor=ANCHOR, notes=[at])
        assert VitalsFlag.POST_ORDER not in result.flags


# =============================================================================
# AC: Post-order fallback with `vitals_post_order` flag.
# =============================================================================


class TestPostOrderFallback:
    """When no pre-anchor note exists in the window, the pipeline falls back to
    the earliest post-anchor note and flags the result."""

    def test_post_order_only(self) -> None:
        post = _note("IPDADMPROGRESS", 60, "BP 118/72 HR 88 RR 18 BT 37.0")
        result = extract_vitals(anchor=ANCHOR, notes=[post])
        assert VitalsFlag.POST_ORDER in result.flags
        assert result.vitals.sbp == 118
        assert result.note_timestamp == post.timestamp

    def test_pre_order_wins_over_post_order(self) -> None:
        # Even when the post-order note is closer in time, a pre-order note
        # in window wins (the audit needs the trigger state, not the response).
        pre = _note("IPDADMPROGRESS", -180, "BP 110/65 HR 80 RR 16 BT 37.0")
        post = _note("IPDADMPROGRESS", 30, "BP 130/85 HR 95 RR 20 BT 37.5")
        result = extract_vitals(anchor=ANCHOR, notes=[pre, post])
        assert VitalsFlag.POST_ORDER not in result.flags
        assert result.vitals.sbp == 110

    def test_picks_earliest_post_order_when_multiple(self) -> None:
        # Among post-order candidates, the earliest is closest to the trigger
        # state — pick that one.
        early = _note("IPDADMPROGRESS", 30, "BP 118/72 HR 88 RR 18 BT 37.0")
        late = _note("IPDADMPROGRESS", 120, "BP 130/85 HR 95 RR 20 BT 37.5")
        result = extract_vitals(anchor=ANCHOR, notes=[late, early])
        assert result.note_timestamp == early.timestamp
        assert result.vitals.sbp == 118


# =============================================================================
# AC: Source provenance recorded:
# IPDADMPROGRESS | IPDNRFOCUSDT | LLM_extracted | none_in_window
# =============================================================================


class TestSourceProvenance:
    """Every result carries exactly one provenance tag matching the AC set."""

    def test_provenance_ipdadmprogress(self) -> None:
        notes = [_note("IPDADMPROGRESS", -30, "BP 120/70 HR 80 RR 16 BT 37.0")]
        result = extract_vitals(anchor=ANCHOR, notes=notes)
        assert result.source == SourceProvenance.IPDADMPROGRESS

    def test_provenance_ipdnrfocusdt(self) -> None:
        notes = [_note("IPDNRFOCUSDT", -30, "BP 120/70 HR 80 RR 16 BT 37.0")]
        result = extract_vitals(anchor=ANCHOR, notes=notes)
        assert result.source == SourceProvenance.IPDNRFOCUSDT

    def test_provenance_none_in_window(self) -> None:
        result = extract_vitals(anchor=ANCHOR, notes=[])
        assert isinstance(result, VitalsResult)  # public-output type tripwire
        assert result.source == SourceProvenance.NONE_IN_WINDOW
        assert result.note_timestamp is None

    def test_ipdadmprogress_preferred_on_tie(self) -> None:
        # When two notes share a timestamp, IPDADMPROGRESS (cleaner SOAP) wins.
        a = _note("IPDADMPROGRESS", -30, "BP 110/65 HR 80 RR 16 BT 37.0")
        b = _note("IPDNRFOCUSDT", -30, "BP 130/85 HR 95 RR 20 BT 37.5")
        result = extract_vitals(anchor=ANCHOR, notes=[b, a])
        assert result.source == SourceProvenance.IPDADMPROGRESS
        assert result.vitals.sbp == 110


# =============================================================================
# AC: LLM fallback boundary — only invoked when regex returns null SBP or HR.
# =============================================================================


class _RecordingFallback:
    """Stub LLM fallback that records every invocation."""

    def __init__(self, returns: VitalSigns) -> None:
        self.returns = returns
        self.calls: list[str] = []

    def __call__(self, text: str) -> VitalSigns:
        self.calls.append(text)
        return self.returns


class TestLLMFallbackBoundary:
    """The LLM is the regex-vs-llm-structured-text escape hatch: it is invoked
    only when regex left SBP or HR empty, never as a default extraction path
    or to second-guess regex hits."""

    def test_not_called_when_sbp_and_hr_present(self) -> None:
        fb = _RecordingFallback(VitalSigns(sbp=999, hr=999))
        notes = [_note("IPDADMPROGRESS", -30, "BP 120/70 HR 80 RR 16 BT 37.0")]
        result = extract_vitals(anchor=ANCHOR, notes=notes, llm_fallback=fb)
        assert fb.calls == []
        # Regex values must NOT be overwritten by the unused fallback's output.
        assert result.vitals.sbp == 120
        assert result.vitals.hr == 80

    def test_called_when_sbp_missing(self) -> None:
        fb = _RecordingFallback(VitalSigns(sbp=118, dbp=72, hr=88, rr=16, bt=37.0))
        # No BP pattern, but HR present — the SBP-null path still triggers.
        notes = [_note("IPDADMPROGRESS", -30, "HR 80")]
        result = extract_vitals(anchor=ANCHOR, notes=notes, llm_fallback=fb)
        assert len(fb.calls) == 1
        assert result.vitals.sbp == 118  # filled by the fallback
        assert result.vitals.hr == 80  # regex value preserved

    def test_called_when_hr_missing(self) -> None:
        fb = _RecordingFallback(VitalSigns(hr=88))
        notes = [_note("IPDADMPROGRESS", -30, "BP 120/70")]
        result = extract_vitals(anchor=ANCHOR, notes=notes, llm_fallback=fb)
        assert len(fb.calls) == 1
        assert result.vitals.hr == 88
        assert result.vitals.sbp == 120  # regex value preserved

    def test_provenance_flips_to_llm_when_fallback_fills_a_gap(self) -> None:
        fb = _RecordingFallback(VitalSigns(sbp=118, dbp=72))
        notes = [_note("IPDADMPROGRESS", -30, "HR 80")]
        result = extract_vitals(anchor=ANCHOR, notes=notes, llm_fallback=fb)
        assert result.source == SourceProvenance.LLM_EXTRACTED

    def test_provenance_stays_regex_when_fallback_adds_nothing(self) -> None:
        # Regex left SBP null, fallback ALSO returns None for it → no
        # meaningful contribution, provenance stays with the source table.
        fb = _RecordingFallback(VitalSigns())
        notes = [_note("IPDADMPROGRESS", -30, "HR 80")]
        result = extract_vitals(anchor=ANCHOR, notes=notes, llm_fallback=fb)
        assert result.source == SourceProvenance.IPDADMPROGRESS

    def test_not_called_when_no_note_in_window(self) -> None:
        # No note → nothing to feed the LLM. The fallback must not be invoked
        # speculatively; the result is simply NONE_IN_WINDOW.
        fb = _RecordingFallback(VitalSigns(sbp=120, hr=80))
        result = extract_vitals(anchor=ANCHOR, notes=[], llm_fallback=fb)
        assert fb.calls == []
        assert result.source == SourceProvenance.NONE_IN_WINDOW

    def test_fallback_invoked_at_most_once(self) -> None:
        # Even if both SBP and HR are missing, the fallback runs once per
        # selected note (one LLM call per anchor — Phase 1 batching contract).
        fb = _RecordingFallback(VitalSigns(sbp=118, hr=88))
        notes = [_note("IPDADMPROGRESS", -30, "RR 16 BT 37.0")]
        extract_vitals(anchor=ANCHOR, notes=notes, llm_fallback=fb)
        assert len(fb.calls) == 1


# =============================================================================
# AC: Property tests for regex robustness against fuzzy formatting (hypothesis).
# =============================================================================


class TestRegexRobustness:
    """Generative invariants: the extractor must never crash on arbitrary
    text, and any value it does return must satisfy its sanity bound."""

    @given(st.text(min_size=0, max_size=200))
    @settings(max_examples=300)
    def test_never_raises(self, text: str) -> None:
        # Total function contract: no input string causes a raise.
        extract_vitals_from_text(text)

    @given(
        sbp=st.integers(min_value=SBP_MIN, max_value=SBP_MAX),
        dbp=st.integers(min_value=DBP_MIN, max_value=DBP_MAX),
        spacing=st.sampled_from([" ", ":", " :", ": "]),
    )
    @settings(max_examples=200)
    def test_bp_round_trip_when_in_bounds(
        self, sbp: int, dbp: int, spacing: str
    ) -> None:
        # For every in-bounds (sbp, dbp) pair and every separator the issue
        # documents, the extractor must round-trip the values.
        text = f"BP{spacing}{sbp}/{dbp}"
        v = extract_vitals_from_text(text)
        assert v.sbp == sbp, f"BP round-trip failed for {text!r}: got sbp={v.sbp}"
        assert v.dbp == dbp, f"BP round-trip failed for {text!r}: got dbp={v.dbp}"

    @given(hr=st.integers(min_value=HR_MIN, max_value=HR_MAX))
    @settings(max_examples=200)
    def test_hr_round_trip_when_in_bounds(self, hr: int) -> None:
        v = extract_vitals_from_text(f"HR {hr}")
        assert v.hr == hr

    @given(
        bt_tenths=st.integers(min_value=int(BT_MIN * 10), max_value=int(BT_MAX * 10))
    )
    @settings(max_examples=200)
    def test_bt_round_trip_when_in_bounds(self, bt_tenths: int) -> None:
        # Iterate on a tenths grid so the printed decimal is exact (no float
        # repr surprises) and stays inside the sanity band.
        bt = bt_tenths / 10
        v = extract_vitals_from_text(f"BT {bt:.1f}")
        assert v.bt is not None
        assert v.bt == pytest.approx(bt)

    @given(st.text(min_size=0, max_size=80))
    @settings(max_examples=300)
    def test_returned_values_always_in_bounds_or_none(self, text: str) -> None:
        # The extractor's contract: any non-None value MUST satisfy its sanity
        # bound. Out-of-bound regex hits are discarded at this layer, so the
        # output is always "clean enough to persist".
        v = extract_vitals_from_text(text)
        if v.sbp is not None:
            assert is_sbp_valid(v.sbp)
        if v.dbp is not None:
            assert is_dbp_valid(v.dbp)
        if v.hr is not None:
            assert is_hr_valid(v.hr)
        if v.rr is not None:
            assert is_rr_valid(v.rr)
        if v.bt is not None:
            assert is_bt_valid(v.bt)


# =============================================================================
# AC: Immutable public output — VitalsResult is frozen, flags is a frozenset.
# =============================================================================


class TestVitalsResultImmutability:
    """The pipeline's public output MUST be genuinely immutable so downstream
    code (audit_pipeline #24, evidence_bundle_builder #16) cannot mutate it."""

    def test_vitals_result_is_frozen(self) -> None:
        result = extract_vitals(anchor=ANCHOR, notes=[])
        # Pydantic frozen models raise on attribute assignment.
        with pytest.raises(Exception):  # ValidationError or AttributeError
            result.source = SourceProvenance.LLM_EXTRACTED  # type: ignore[misc]

    def test_vitals_result_flags_is_frozenset(self) -> None:
        result = extract_vitals(anchor=ANCHOR, notes=[])
        # frozenset, not a plain set — so callers can't add a flag post-hoc.
        assert isinstance(result.flags, frozenset)

    def test_vital_signs_is_frozen_dataclass(self) -> None:
        v = VitalSigns(sbp=120, dbp=70, hr=80, rr=16, bt=37.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            v.sbp = 130  # type: ignore[misc]


class TestLLMFallbackTypeAlias:
    """The LLMFallback alias is part of the public surface; downstream code
    type-annotates against it and must keep working as a Callable contract."""

    def test_callable_protocol(self) -> None:
        # A plain function with the right signature must satisfy the alias.
        def stub(text: str) -> VitalSigns:
            del text
            return VitalSigns()

        fb: LLMFallback = stub
        assert fb("anything") == VitalSigns()


# =============================================================================
# Regression (codex review, 2026-05-15): tz-aware timestamp contract MUST be
# enforced at the type boundary. A naive VitalsNote would crash the window
# filter; a naive anchor would silently bypass the UTC contract because two
# naive datetimes subtract cleanly but the result is meaningless across zones.
# =============================================================================


class TestTimestampContract:
    """Naive (tz-unaware) datetimes are rejected at every entry point.

    The ingest layer normalizes order anchors and row timestamps to UTC; this
    test class is the tripwire that prevents a downstream caller from passing
    a naive datetime through the validator's gap."""

    def test_vitals_note_rejects_naive_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            VitalsNote(
                source="IPDADMPROGRESS",
                timestamp=datetime(2026, 5, 14, 12, 0, 0),  # naive — no tzinfo
                text="BP 120/70",
            )

    def test_vitals_note_accepts_non_utc_aware_timestamp(self) -> None:
        # The validator is "tz-aware", not "must-be-UTC" — any non-naive zone
        # is acceptable because datetime arithmetic between aware datetimes
        # normalizes to UTC. The ingest layer happens to feed UTC, but the
        # extractor doesn't need to assert that specifically.
        bangkok = timezone(timedelta(hours=7))
        n = VitalsNote(
            source="IPDADMPROGRESS",
            timestamp=datetime(2026, 5, 14, 19, 0, 0, tzinfo=bangkok),
            text="BP 120/70",
        )
        assert n.timestamp.tzinfo is not None

    def test_extract_vitals_rejects_naive_anchor(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            extract_vitals(anchor=datetime(2026, 5, 14, 12, 0, 0), notes=[])
