"""RED-phase failing tests for issue #16 (bba.evidence_bundle_builder).

Each ``class`` maps to one acceptance criterion in the issue body. Tests
assert contracts (the WHY), not implementation choices — see PRD §"Testing
Decisions".

No implementation exists yet; every test MUST fail in this scaffold commit
(``NotImplementedError`` from the entry-point stubs, or ``ValidationError`` /
``ValueError`` from the model-level invariants — those are the contract too).
If a test fails with ``ImportError`` or ``AttributeError`` instead, the
scaffold is wrong: fix the public surface, not the test.

Acceptance-criterion → test-class map:

* AC ① "implementation in src/bba/evidence_bundle_builder/"
  → :class:`TestPublicSurface`, :class:`TestModelImmutability`
* AC ② "per-source window enforcement"
  → :class:`TestProgressNoteWindow`, :class:`TestFocusNoteWindow`,
    :class:`TestMedWindow`, :class:`TestHbHistoryWindow`,
    :class:`TestVitalsWindow`, :class:`TestDiagnosesAreNotTimeWindowed`
* AC ③ "IPDADMPROGRESS section priority on truncation"
  → :class:`TestSOAPSectionParsing`,
    :class:`TestSectionEmissionOrderInBundle`,
    :class:`TestSectionTruncationDropOrder`
* AC ④ "IPDNRFOCUSDT 5+5 split, closest-first"
  → :class:`TestFocusNote5And5Split`,
    :class:`TestFocusNoteCapTotal10InBundle`
* AC ⑤ "stable evidence IDs (E1...EN) deterministic"
  → :class:`TestStableEvidenceIDs`
* AC ⑥ "canonical JSON serialization byte-identical"
  → :class:`TestCanonicalJSONSerialization`,
    :class:`TestCanonicalJSONNFCNormalization`
* AC ⑦ "hash-stability property test under input reordering"
  → :class:`TestBundleHashStability`
* AC ⑧ "cap enforcement: 8K char proxy"
  → :class:`TestCharCapEnforcement`,
    :class:`TestProgressNoteCap8`
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bba.evidence_bundle_builder import (
    CAP_FOCUS_AFTER,
    CAP_FOCUS_BEFORE,
    CAP_PROGRESS,
    DEFAULT_CHAR_CAP,
    SECTION_PRIORITY,
    DiagnosisRecord,
    EvidenceBundle,
    EvidenceInputs,
    EvidenceItem,
    FocusNote,
    HbRecord,
    MedRecord,
    OrderAnchor,
    ProgressNote,
    VitalsRecord,
    build_evidence_bundle,
    bundle_hash,
    canonical_serialize,
    parse_soap_sections,
    split_focus_notes_5_5,
    truncate_to_char_cap,
)


# =============================================================================
# Shared fixtures
#
# A single fixed anchor lets every test use whole-hour offsets without
# daylight-saving artefacts (Asia/Bangkok has no DST, but UTC arithmetic is
# simpler regardless). Mirrors the convention from test_hb_lookup /
# test_vitals_extractor — one ANCHOR + one tiny constructor per record type
# means tests stay focused on the property under test.
# =============================================================================


ANCHOR_DT = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)


def _anchor(
    *,
    products: tuple[str, ...] = ("LPRC",),
    hn_hash: str = "hn-aaa",
    an_hash: str = "an-bbb",
) -> OrderAnchor:
    return OrderAnchor(
        order_datetime=ANCHOR_DT,
        hn_hash=hn_hash,
        an_hash=an_hash,
        products=products,
    )


def _progress(*, offset_hours: float, text: str = "S: nil\nO: stable\nA: ok\nP: observe") -> ProgressNote:
    return ProgressNote(
        timestamp=ANCHOR_DT + timedelta(hours=offset_hours),
        text=text,
    )


def _focus(*, offset_hours: float, text: str = "FOCUS: pain") -> FocusNote:
    return FocusNote(
        timestamp=ANCHOR_DT + timedelta(hours=offset_hours),
        text=text,
    )


def _med(*, offset_hours: float, drug: str = "Furosemide 20mg IV") -> MedRecord:
    return MedRecord(
        timestamp=ANCHOR_DT + timedelta(hours=offset_hours),
        drug=drug,
    )


def _hb(*, offset_hours: float, value: float = 7.5, source: str = "HEMATOLOGY") -> HbRecord:
    return HbRecord(
        timestamp=ANCHOR_DT + timedelta(hours=offset_hours),
        value_g_dl=value,
        source=source,  # type: ignore[arg-type]
    )


def _vitals(
    *,
    offset_hours: float,
    sbp: int | None = 110,
    hr: int | None = 88,
    source: str = "IPDADMPROGRESS",
) -> VitalsRecord:
    return VitalsRecord(
        timestamp=ANCHOR_DT + timedelta(hours=offset_hours),
        source=source,  # type: ignore[arg-type]
        sbp=sbp,
        hr=hr,
    )


def _build_minimal(**overrides: object) -> EvidenceBundle:
    """Build a bundle from a minimal default :class:`EvidenceInputs`.

    Lets each test override exactly the field it cares about while keeping
    the other inputs as small valid defaults — same pattern as the
    ``_row`` builder in ``test_audit_store.py``."""
    inputs_kwargs: dict[str, object] = {
        "anchor": _anchor(),
        "diagnoses": (DiagnosisRecord(icd10="D50.9"),),
    }
    inputs_kwargs.update(overrides)
    inputs = EvidenceInputs(**inputs_kwargs)  # type: ignore[arg-type]
    return build_evidence_bundle(inputs=inputs)


# =============================================================================
# AC ① — implementation in src/bba/evidence_bundle_builder/
# =============================================================================


class TestPublicSurface:
    """Module re-exports the names downstream callers depend on.

    Collection-level smoke check: every name in the issue body's contract is
    importable from the top-level package. If this class disappears, future
    refactors could silently relocate a symbol (e.g., ``build_evidence_bundle``)
    and break every consumer (#17, #21) without a single test failure here."""

    def test_top_level_exports_named_in_issue(self) -> None:
        # Any rename of these symbols is a breaking API change; the test
        # exists so that breakage shows up as a single named failure.
        from bba import evidence_bundle_builder as ebb

        for name in (
            "build_evidence_bundle",
            "EvidenceBundle",
            "EvidenceItem",
            "EvidenceInputs",
            "OrderAnchor",
            "DiagnosisRecord",
            "ProgressNote",
            "FocusNote",
            "MedRecord",
            "HbRecord",
            "VitalsRecord",
            "canonical_serialize",
            "bundle_hash",
            "parse_soap_sections",
            "split_focus_notes_5_5",
            "truncate_to_char_cap",
            "SECTION_PRIORITY",
            "CAP_PROGRESS",
            "CAP_FOCUS_BEFORE",
            "CAP_FOCUS_AFTER",
            "DEFAULT_CHAR_CAP",
        ):
            assert hasattr(ebb, name), f"missing public export: {name}"

    def test_window_constants_match_prd_section_7(self) -> None:
        # Window thresholds are the spec, not tunable knobs. A drift here
        # means the bundle's clinical relevance window changed without an
        # ADR — the test is the canary.
        from bba.evidence_bundle_builder import (
            WINDOW_FOCUS,
            WINDOW_HB_BEFORE,
            WINDOW_MED_AFTER,
            WINDOW_MED_BEFORE,
            WINDOW_PROGRESS,
            WINDOW_VITALS,
        )

        assert WINDOW_PROGRESS == timedelta(hours=24)
        assert WINDOW_FOCUS == timedelta(hours=24)
        assert WINDOW_MED_BEFORE == timedelta(hours=72)
        assert WINDOW_MED_AFTER == timedelta(hours=24)
        assert WINDOW_HB_BEFORE == timedelta(days=7)
        assert WINDOW_VITALS == timedelta(hours=6)

    def test_caps_match_issue_body(self) -> None:
        assert CAP_PROGRESS == 8
        assert CAP_FOCUS_BEFORE == 5
        assert CAP_FOCUS_AFTER == 5
        assert DEFAULT_CHAR_CAP == 8000

    def test_section_priority_is_canonical_order(self) -> None:
        # Ordering matters: this tuple is the contract for both emission
        # order in the bundle JSON and reverse-traversal for truncation.
        assert SECTION_PRIORITY == (
            "ASSESSMENT",
            "PLAN",
            "OBJECTIVE",
            "SUBJECTIVE",
        )


class TestModelImmutability:
    """Inputs are frozen and reject naive datetimes / out-of-range values.

    The tz-aware contract is the project-wide invariant (CONTEXT.md
    "tz-aware UTC"); admitting a naive timestamp here would later raise
    an opaque comparison ``TypeError`` inside the window filter. Catching
    it at construction names the offending field instead."""

    def test_anchor_rejects_naive_order_datetime(self) -> None:
        with pytest.raises(ValidationError):
            OrderAnchor(
                order_datetime=datetime(2026, 5, 15, 12, 0, 0),  # type: ignore[arg-type]
                hn_hash="x",
                an_hash="y",
                products=("LPRC",),
            )

    def test_progress_note_rejects_naive_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            ProgressNote(timestamp=datetime(2026, 5, 15, 12, 0, 0), text="x")  # type: ignore[arg-type]

    def test_focus_note_rejects_naive_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            FocusNote(timestamp=datetime(2026, 5, 15, 12, 0, 0), text="x")  # type: ignore[arg-type]

    def test_med_rejects_naive_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            MedRecord(timestamp=datetime(2026, 5, 15, 12, 0, 0), drug="x")  # type: ignore[arg-type]

    def test_hb_rejects_naive_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            HbRecord(
                timestamp=datetime(2026, 5, 15, 12, 0, 0),  # type: ignore[arg-type]
                value_g_dl=7.5,
                source="HEMATOLOGY",
            )

    def test_hb_rejects_out_of_range_value(self) -> None:
        # Mirrors HbObservation: < 2.0 or > 25.0 is a transcription error.
        with pytest.raises(ValidationError):
            HbRecord(timestamp=ANCHOR_DT, value_g_dl=1.9, source="HEMATOLOGY")
        with pytest.raises(ValidationError):
            HbRecord(timestamp=ANCHOR_DT, value_g_dl=25.1, source="HEMATOLOGY")

    def test_vitals_rejects_naive_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(
                timestamp=datetime(2026, 5, 15, 12, 0, 0),  # type: ignore[arg-type]
                source="IPDADMPROGRESS",
            )

    def test_anchor_is_frozen(self) -> None:
        a = _anchor()
        with pytest.raises(ValidationError):
            a.products = ("FFP",)  # type: ignore[misc]

    def test_inputs_is_frozen(self) -> None:
        inputs = EvidenceInputs(anchor=_anchor())
        with pytest.raises(ValidationError):
            inputs.diagnoses = (DiagnosisRecord(icd10="D50.9"),)  # type: ignore[misc]

    def test_evidence_item_id_must_start_with_E(self) -> None:
        # Stable-IDs AC: quote_grounder pattern-matches on E-prefix; an
        # arbitrary id would silently fail downstream rather than at
        # construction.
        with pytest.raises(ValidationError):
            EvidenceItem(
                id="X1",
                source="Diagnosis",
                timestamp_utc=None,
                payload={"icd10": "D50.9"},
            )

    def test_evidence_item_id_must_be_E_plus_digits(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceItem(
                id="E1a",
                source="Diagnosis",
                timestamp_utc=None,
                payload={"icd10": "D50.9"},
            )

    def test_bundle_hash_must_be_64_char_lowercase_hex(self) -> None:
        # Wrong-size digests (sha1, sha512) are quietly damaging downstream;
        # reject at construction so AuditRow.evidence_bundle_hash never
        # holds a malformed value.
        with pytest.raises(ValidationError):
            EvidenceBundle(
                items=(),
                canonical_json="[]",
                bundle_hash="abc123",  # too short
            )
        with pytest.raises(ValidationError):
            EvidenceBundle(
                items=(),
                canonical_json="[]",
                bundle_hash="A" * 64,  # uppercase = non-canonical hex
            )


# =============================================================================
# AC ② — per-source window enforcement
# =============================================================================


def _items_by_source(bundle: EvidenceBundle, source: str) -> tuple[EvidenceItem, ...]:
    return tuple(item for item in bundle.items if item.source == source)


class TestProgressNoteWindow:
    """IPDADMPROGRESS: ``[anchor - 24h, anchor + 24h]``.

    The window is symmetric — pre-order notes give context for the decision,
    post-order notes give the bedside response that the LLM can correlate
    with the order's appropriateness."""

    def test_inside_24h_before_kept(self) -> None:
        bundle = _build_minimal(progress_notes=(_progress(offset_hours=-23.9),))
        assert len(_items_by_source(bundle, "IPDADMPROGRESS")) == 1

    def test_inside_24h_after_kept(self) -> None:
        bundle = _build_minimal(progress_notes=(_progress(offset_hours=23.9),))
        assert len(_items_by_source(bundle, "IPDADMPROGRESS")) == 1

    def test_outside_24h_before_dropped(self) -> None:
        bundle = _build_minimal(progress_notes=(_progress(offset_hours=-24.1),))
        assert _items_by_source(bundle, "IPDADMPROGRESS") == ()

    def test_outside_24h_after_dropped(self) -> None:
        bundle = _build_minimal(progress_notes=(_progress(offset_hours=24.1),))
        assert _items_by_source(bundle, "IPDADMPROGRESS") == ()


class TestFocusNoteWindow:
    """IPDNRFOCUSDT: ``[anchor - 24h, anchor + 24h]`` before the 5+5 split."""

    def test_inside_24h_kept(self) -> None:
        bundle = _build_minimal(
            focus_notes=(_focus(offset_hours=-23.9), _focus(offset_hours=23.9)),
        )
        assert len(_items_by_source(bundle, "IPDNRFOCUSDT")) == 2

    def test_outside_24h_dropped(self) -> None:
        bundle = _build_minimal(
            focus_notes=(_focus(offset_hours=-24.1), _focus(offset_hours=24.1)),
        )
        assert _items_by_source(bundle, "IPDNRFOCUSDT") == ()


class TestMedWindow:
    """MED: asymmetric ``[anchor - 72 h, anchor + 24 h]``.

    Asymmetry rationale: drug history shaping the order extends back 3 days
    (typical anticoag washout), but post-order administration only matters
    for the immediate next 24 h."""

    def test_inside_minus_72h_kept(self) -> None:
        bundle = _build_minimal(meds=(_med(offset_hours=-71.9),))
        assert len(_items_by_source(bundle, "MED")) == 1

    def test_inside_plus_24h_kept(self) -> None:
        bundle = _build_minimal(meds=(_med(offset_hours=23.9),))
        assert len(_items_by_source(bundle, "MED")) == 1

    def test_outside_minus_72h_dropped(self) -> None:
        bundle = _build_minimal(meds=(_med(offset_hours=-72.1),))
        assert _items_by_source(bundle, "MED") == ()

    def test_outside_plus_24h_dropped(self) -> None:
        bundle = _build_minimal(meds=(_med(offset_hours=24.1),))
        assert _items_by_source(bundle, "MED") == ()


class TestHbHistoryWindow:
    """Lab Hb: ``[anchor - 7 d, anchor]`` (pre-order only — mirrors :mod:`bba.hb_lookup`)."""

    def test_inside_7d_kept(self) -> None:
        bundle = _build_minimal(hb_history=(_hb(offset_hours=-167.9),))  # 7d - small epsilon
        assert len(_items_by_source(bundle, "Lab")) == 1

    def test_outside_7d_dropped(self) -> None:
        bundle = _build_minimal(hb_history=(_hb(offset_hours=-168.1),))  # 7d + small epsilon
        assert _items_by_source(bundle, "Lab") == ()

    def test_post_anchor_dropped(self) -> None:
        # Hb history is the audit signal for *this* order; post-order labs
        # belong to the response analysis, not the decision evidence.
        bundle = _build_minimal(hb_history=(_hb(offset_hours=1.0),))
        assert _items_by_source(bundle, "Lab") == ()


class TestVitalsWindow:
    """Vitals: ``[anchor - 6 h, anchor + 6 h]`` (mirrors :mod:`bba.vitals_extractor`)."""

    def test_inside_6h_kept(self) -> None:
        bundle = _build_minimal(vitals=(_vitals(offset_hours=-5.9), _vitals(offset_hours=5.9)))
        assert len(_items_by_source(bundle, "Vitals")) == 2

    def test_outside_6h_dropped(self) -> None:
        bundle = _build_minimal(vitals=(_vitals(offset_hours=-6.1),))
        assert _items_by_source(bundle, "Vitals") == ()


class TestDiagnosesAreNotTimeWindowed:
    """Diagnoses are AN-scoped — the full ICD-10 list for the encounter
    appears in the bundle regardless of timestamps. Per PRD §7."""

    def test_all_diagnoses_kept(self) -> None:
        diagnoses = tuple(
            DiagnosisRecord(icd10=code)
            for code in ("D50.9", "I10", "E11.9", "K76.0", "Z51.81")
        )
        inputs = EvidenceInputs(anchor=_anchor(), diagnoses=diagnoses)
        bundle = build_evidence_bundle(inputs=inputs)
        assert len(_items_by_source(bundle, "Diagnosis")) == 5


# =============================================================================
# AC ③ — IPDADMPROGRESS section priority on truncation
# =============================================================================


_FULL_SOAP = (
    "S: ผู้ป่วยรู้สึกเหนื่อยมากขึ้น\n"
    "O: BP 95/60 PR 110 RR 22\n"
    "A: Acute anemia secondary to GI bleed\n"
    "P: PRBC 1U, recheck Hb in 6 hours"
)


class TestSOAPSectionParsing:
    """parse_soap_sections returns every key in :data:`SECTION_PRIORITY`."""

    def test_returns_all_four_keys(self) -> None:
        sections = parse_soap_sections(_FULL_SOAP)
        assert set(sections.keys()) == set(SECTION_PRIORITY)

    def test_assigns_text_to_correct_section(self) -> None:
        sections = parse_soap_sections(_FULL_SOAP)
        assert "เหนื่อย" in sections["SUBJECTIVE"]
        assert "BP 95/60" in sections["OBJECTIVE"]
        assert "Acute anemia" in sections["ASSESSMENT"]
        assert "PRBC 1U" in sections["PLAN"]

    def test_recognizes_long_form_headers(self) -> None:
        text = (
            "Subjective: tiring easily\n"
            "Objective: HR 100\n"
            "Assessment: anemia\n"
            "Plan: PRBC"
        )
        sections = parse_soap_sections(text)
        assert "tiring easily" in sections["SUBJECTIVE"]
        assert "HR 100" in sections["OBJECTIVE"]
        assert "anemia" in sections["ASSESSMENT"]
        assert "PRBC" in sections["PLAN"]

    def test_no_headers_falls_back_to_objective(self) -> None:
        # The HOSxP column is named OBJECTIVE; the no-header default is the
        # most truthful interpretation. Routing it elsewhere would be a
        # silent data move.
        text = "BP 110/70, PR 88, alert and oriented"
        sections = parse_soap_sections(text)
        assert sections["OBJECTIVE"].strip() == text
        assert sections["SUBJECTIVE"] == ""
        assert sections["ASSESSMENT"] == ""
        assert sections["PLAN"] == ""

    def test_empty_text_returns_all_empty(self) -> None:
        sections = parse_soap_sections("")
        assert sections == {k: "" for k in SECTION_PRIORITY}


class TestSectionEmissionOrderInBundle:
    """When an IPDADMPROGRESS item lands in the bundle, its sections appear
    in priority order (A, P, O, S) inside the item's payload.

    The order is the LLM's reading order — putting the clinician's
    interpretation (A) and treatment plan (P) first focuses the model on
    the decision rationale before the supporting data (O) and the patient
    quote (S)."""

    def test_payload_lists_sections_in_priority_order(self) -> None:
        bundle = _build_minimal(progress_notes=(_progress(offset_hours=-1, text=_FULL_SOAP),))
        progress_items = _items_by_source(bundle, "IPDADMPROGRESS")
        assert len(progress_items) == 1
        sections = progress_items[0].payload["sections"]
        labels = tuple(s["label"] for s in sections)
        assert labels == SECTION_PRIORITY


class TestSectionTruncationDropOrder:
    """When the bundle exceeds char_cap, sections drop SUBJECTIVE first.

    Drop order is the reverse of SECTION_PRIORITY: SUBJECTIVE → OBJECTIVE →
    PLAN → (drop the whole item only when ASSESSMENT alone still doesn't fit).
    The reverse-priority drop rule is what makes "section priority on
    truncation" meaningful — without it, truncation would be a content-blind
    char trim that could lop off the diagnosis."""

    def test_subjective_dropped_before_other_sections(self) -> None:
        # Build a bundle whose only over-budget content is the SUBJECTIVE
        # of one progress note. Force the cap below the full-bundle size by
        # pumping SUBJECTIVE up; the remaining (A, P, O) must survive.
        long_subjective = "S: " + ("ก" * 5000) + "\nO: hr 80\nA: stable\nP: observe"
        inputs = EvidenceInputs(
            anchor=_anchor(),
            progress_notes=(_progress(offset_hours=-1, text=long_subjective),),
        )
        bundle = build_evidence_bundle(inputs=inputs, char_cap=2000)
        progress_items = _items_by_source(bundle, "IPDADMPROGRESS")
        assert len(progress_items) == 1, "the progress item itself must survive"
        labels = {s["label"] for s in progress_items[0].payload["sections"]}
        assert "SUBJECTIVE" not in labels
        assert {"ASSESSMENT", "PLAN", "OBJECTIVE"}.issubset(labels)

    def test_assessment_is_last_to_drop(self) -> None:
        # When every section is huge, ASSESSMENT survives last; the test
        # documents the inversion of the priority tuple as the drop order.
        big = lambda label: f"{label}: " + ("ก" * 4000)  # noqa: E731
        text = "\n".join(big(c) for c in ("S", "O", "A", "P"))
        inputs = EvidenceInputs(
            anchor=_anchor(),
            progress_notes=(_progress(offset_hours=-1, text=text),),
        )
        bundle = build_evidence_bundle(inputs=inputs, char_cap=4500)
        progress_items = _items_by_source(bundle, "IPDADMPROGRESS")
        # The bundle must fit the cap and at least retain ASSESSMENT.
        assert len(bundle.canonical_json) <= 4500
        if progress_items:
            labels = {s["label"] for s in progress_items[0].payload["sections"]}
            assert "ASSESSMENT" in labels


# =============================================================================
# AC ④ — IPDNRFOCUSDT 5+5 split, closest-first
# =============================================================================


class TestFocusNote5And5Split:
    """split_focus_notes_5_5 returns ≤5 closest before + ≤5 closest after."""

    def test_exactly_5_before_5_after_when_abundant(self) -> None:
        before = tuple(_focus(offset_hours=-h) for h in (1, 2, 3, 4, 5, 6, 7))
        after = tuple(_focus(offset_hours=h) for h in (1, 2, 3, 4, 5, 6, 7))
        out = split_focus_notes_5_5(notes=before + after, anchor=ANCHOR_DT)
        assert len(out) == 10

    def test_before_half_is_5_closest(self) -> None:
        notes = tuple(_focus(offset_hours=-h) for h in (1, 2, 3, 4, 5, 6, 7, 8))
        out = split_focus_notes_5_5(notes=notes, anchor=ANCHOR_DT)
        before_offsets = sorted(
            (ANCHOR_DT - n.timestamp).total_seconds() / 3600.0 for n in out
        )
        assert before_offsets == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_after_half_is_5_closest(self) -> None:
        notes = tuple(_focus(offset_hours=h) for h in (1, 2, 3, 4, 5, 6, 7, 8))
        out = split_focus_notes_5_5(notes=notes, anchor=ANCHOR_DT)
        after_offsets = sorted(
            (n.timestamp - ANCHOR_DT).total_seconds() / 3600.0 for n in out
        )
        assert after_offsets == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_partial_count_when_fewer_available(self) -> None:
        # Only 3 before; the 5-cap is a ceiling, not a target — no padding.
        before = tuple(_focus(offset_hours=-h) for h in (1, 2, 3))
        after = tuple(_focus(offset_hours=h) for h in (1, 2, 3, 4, 5, 6))
        out = split_focus_notes_5_5(notes=before + after, anchor=ANCHOR_DT)
        # 3 before + 5 after
        assert len(out) == 8

    def test_at_anchor_counts_as_before(self) -> None:
        # An exactly-at-anchor note belongs to the pre-order side: at-anchor
        # is the latest possible "what was true at decision time" data point.
        notes = (_focus(offset_hours=0),)
        out = split_focus_notes_5_5(notes=notes, anchor=ANCHOR_DT)
        assert len(out) == 1
        assert out[0].timestamp == ANCHOR_DT


class TestFocusNoteCapTotal10InBundle:
    """The bundle's IPDNRFOCUSDT section never exceeds 10 entries."""

    def test_cap_at_10_when_window_is_dense(self) -> None:
        notes = tuple(
            _focus(offset_hours=h)
            for h in (-12, -10, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 10, 12)
        )
        inputs = EvidenceInputs(anchor=_anchor(), focus_notes=notes)
        bundle = build_evidence_bundle(inputs=inputs)
        assert len(_items_by_source(bundle, "IPDNRFOCUSDT")) == 10


# =============================================================================
# AC ⑤ — stable evidence IDs (E1...EN) deterministic
# =============================================================================


class TestStableEvidenceIDs:
    """IDs are E1, E2, ..., EN sequentially, and the assignment is byte-stable
    across input shuffles. The downstream :mod:`bba.quote_grounder` will
    string-match the IDs in LLM output — drift here is silent corruption."""

    def test_ids_are_E1_then_E2_then_E3(self) -> None:
        bundle = _build_minimal(
            progress_notes=(_progress(offset_hours=-1),),
            hb_history=(_hb(offset_hours=-2),),
        )
        ids = [item.id for item in bundle.items]
        assert ids == [f"E{i}" for i in range(1, len(ids) + 1)]

    def test_ids_stable_across_input_record_order(self) -> None:
        meds_a = (_med(offset_hours=-1, drug="A"), _med(offset_hours=-2, drug="B"))
        meds_b = tuple(reversed(meds_a))
        ids_a = [
            (it.source, it.payload, it.id)
            for it in _build_minimal(meds=meds_a).items
        ]
        ids_b = [
            (it.source, it.payload, it.id)
            for it in _build_minimal(meds=meds_b).items
        ]
        # Same bag of records (just reordered) → same ID assignment per item.
        # Sort by id for comparison; payloads must match.
        assert sorted(ids_a, key=lambda t: t[2]) == sorted(ids_b, key=lambda t: t[2])

    def test_ids_are_unique(self) -> None:
        bundle = _build_minimal(
            progress_notes=tuple(_progress(offset_hours=-h) for h in (1, 2, 3)),
            focus_notes=tuple(_focus(offset_hours=-h) for h in (1, 2, 3)),
            meds=tuple(_med(offset_hours=-h) for h in (1, 2, 3)),
        )
        ids = [item.id for item in bundle.items]
        assert len(ids) == len(set(ids))

    def test_canonical_emission_is_by_source_then_timestamp(self) -> None:
        # Per the issue body's "stable evidence IDs assigned deterministically",
        # the emission order is canonical: the literal source order in
        # :data:`EvidenceSource` (Diagnosis, IPDADMPROGRESS, IPDNRFOCUSDT,
        # MED, Lab, Vitals), then by timestamp ascending within each source.
        bundle = _build_minimal(
            progress_notes=(_progress(offset_hours=-2), _progress(offset_hours=-1)),
            meds=(_med(offset_hours=-1, drug="A"),),
        )
        sources = [item.source for item in bundle.items]
        assert sources.index("Diagnosis") < sources.index("IPDADMPROGRESS")
        assert sources.index("IPDADMPROGRESS") < sources.index("MED")
        progress_items = _items_by_source(bundle, "IPDADMPROGRESS")
        progress_timestamps = [it.timestamp_utc for it in progress_items]
        assert progress_timestamps == sorted(progress_timestamps)  # type: ignore[type-var]


# =============================================================================
# AC ⑥ — canonical JSON serialization byte-identical
# =============================================================================


class TestCanonicalJSONSerialization:
    """canonical_serialize: sorted keys, NFC strings, fixed indent."""

    def test_keys_sorted_recursively(self) -> None:
        out = canonical_serialize({"b": 1, "a": {"y": 2, "x": 1}})
        loaded = json.loads(out)
        # Outer keys
        assert list(loaded.keys()) == ["a", "b"]
        # Nested keys
        assert list(loaded["a"].keys()) == ["x", "y"]
        # Byte stability of the actual string (sorted keys + 2-space indent)
        assert "\n" in out, "canonical output uses fixed 2-space indentation"

    def test_byte_identical_for_equivalent_dicts(self) -> None:
        a = canonical_serialize({"alpha": 1, "beta": [{"y": 2, "x": 1}]})
        b = canonical_serialize({"beta": [{"x": 1, "y": 2}], "alpha": 1})
        assert a == b

    def test_no_trailing_newline(self) -> None:
        # Hashing the bytes the LLM will see — a stray trailing newline
        # would change every bundle hash in the wild.
        out = canonical_serialize({"a": 1})
        assert not out.endswith("\n")

    def test_bundle_canonical_json_round_trips(self) -> None:
        bundle = _build_minimal(progress_notes=(_progress(offset_hours=-1),))
        # Re-running build on the same input gives the same canonical bytes.
        bundle2 = _build_minimal(progress_notes=(_progress(offset_hours=-1),))
        assert bundle.canonical_json == bundle2.canonical_json
        assert bundle.bundle_hash == bundle2.bundle_hash

    def test_bundle_hash_is_sha256_of_canonical_bytes(self) -> None:
        bundle = _build_minimal()
        expected = hashlib.sha256(bundle.canonical_json.encode("utf-8")).hexdigest()
        assert bundle.bundle_hash == expected


class TestCanonicalJSONNFCNormalization:
    """Adversarial: NFD input must canonicalize to the same hash as NFC.

    The PRD §7 contract is "UTF-8 NFC" — admitting NFD would let the same
    bundle produce two hashes. The fixture uses Latin precomposed-vs-
    combining diacritics ("café" / "naïve") because NFD canonical
    decomposition exists for those codepoints; Thai consonants + tone marks
    mostly have NO canonical decomposition (U+0E33 decomposes only under
    NFKD/compatibility), so a Thai-only test would be a no-op masquerading
    as a real NFD assertion. The contract is script-agnostic, so a Latin
    fixture is sufficient evidence that ``canonical_serialize`` applies
    NFC normalization globally."""

    def test_nfd_and_nfc_produce_identical_canonical_output(self) -> None:
        nfc = unicodedata.normalize("NFC", "café")
        nfd = unicodedata.normalize("NFD", "café")
        assert nfc != nfd, "fixture is malformed if NFD == NFC for café"
        a = canonical_serialize({"text": nfc})
        b = canonical_serialize({"text": nfd})
        assert a == b

    def test_nfd_progress_note_hashes_same_as_nfc(self) -> None:
        # Embed precomposed-vs-decomposed diacritics inside a SOAP note so
        # the contract is exercised through the full bundle pipeline, not
        # just the standalone serializer.
        nfc_text = unicodedata.normalize(
            "NFC", "S: feels naïve\nO: BP café-au-lait spots\nA: anemia\nP: PRBC"
        )
        nfd_text = unicodedata.normalize("NFD", nfc_text)
        assert nfc_text != nfd_text, "fixture is malformed if NFD == NFC"
        bundle_nfc = _build_minimal(progress_notes=(_progress(offset_hours=-1, text=nfc_text),))
        bundle_nfd = _build_minimal(progress_notes=(_progress(offset_hours=-1, text=nfd_text),))
        assert bundle_nfc.bundle_hash == bundle_nfd.bundle_hash


# =============================================================================
# AC ⑦ — hash-stability property test under input reordering
# =============================================================================


def _shuffled_progress_notes(seed: int, n: int = 6) -> tuple[ProgressNote, ...]:
    """Build n notes within +/- 12 h of the anchor with a deterministic shuffle."""
    base = tuple(
        _progress(offset_hours=-h, text=f"S: s{h}\nO: o{h}\nA: a{h}\nP: p{h}")
        for h in range(1, n + 1)
    )
    # Deterministic shuffle by sort key derived from seed.
    return tuple(sorted(base, key=lambda n: hash((seed, n.timestamp))))


class TestBundleHashStability:
    """Property: any reordering of the input records yields the same hash."""

    @given(seed_a=st.integers(0, 10_000), seed_b=st.integers(0, 10_000))
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_progress_notes_reorder_invariance(self, seed_a: int, seed_b: int) -> None:
        a = _shuffled_progress_notes(seed_a)
        b = _shuffled_progress_notes(seed_b)
        bundle_a = _build_minimal(progress_notes=a)
        bundle_b = _build_minimal(progress_notes=b)
        assert bundle_a.bundle_hash == bundle_b.bundle_hash

    @given(perm=st.permutations([1, 2, 3, 4, 5]))
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_diagnosis_order_does_not_change_hash(self, perm: list[int]) -> None:
        codes = ("D50.9", "I10", "E11.9", "K76.0", "Z51.81")
        diagnoses = tuple(DiagnosisRecord(icd10=codes[i - 1]) for i in perm)
        h = build_evidence_bundle(
            inputs=EvidenceInputs(anchor=_anchor(), diagnoses=diagnoses)
        ).bundle_hash
        canonical_h = build_evidence_bundle(
            inputs=EvidenceInputs(
                anchor=_anchor(),
                diagnoses=tuple(DiagnosisRecord(icd10=c) for c in codes),
            )
        ).bundle_hash
        assert h == canonical_h

    def test_focus_note_reorder_does_not_change_hash(self) -> None:
        notes_a = tuple(_focus(offset_hours=h, text=f"f{h}") for h in (-3, -2, -1, 1, 2, 3))
        notes_b = tuple(reversed(notes_a))
        a = _build_minimal(focus_notes=notes_a)
        b = _build_minimal(focus_notes=notes_b)
        assert a.bundle_hash == b.bundle_hash

    def test_med_reorder_does_not_change_hash(self) -> None:
        meds_a = tuple(_med(offset_hours=-h, drug=f"D{h}") for h in (1, 2, 3, 4))
        meds_b = tuple(reversed(meds_a))
        a = _build_minimal(meds=meds_a)
        b = _build_minimal(meds=meds_b)
        assert a.bundle_hash == b.bundle_hash


# =============================================================================
# AC ⑧ — cap enforcement: 8K char proxy + per-source caps
# =============================================================================


class TestCharCapEnforcement:
    """The serialized bundle never exceeds the configured char_cap."""

    def test_default_cap_8000_chars(self) -> None:
        # Pump in enough content to exceed 8K characters easily.
        progress = tuple(
            _progress(offset_hours=-h, text=f"S: {'ข' * 600}\nO: o\nA: a\nP: p")
            for h in (1, 2, 3, 4, 5, 6, 7, 8)
        )
        focus = tuple(_focus(offset_hours=h, text="ก" * 600) for h in (-6, -4, -2, 1, 3, 5))
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(anchor=_anchor(), progress_notes=progress, focus_notes=focus),
        )
        assert len(bundle.canonical_json) <= DEFAULT_CHAR_CAP

    def test_caller_supplied_cap_is_respected(self) -> None:
        progress = tuple(
            _progress(offset_hours=-h, text=f"S: {'ข' * 400}\nO: o\nA: a\nP: p")
            for h in (1, 2, 3, 4)
        )
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(anchor=_anchor(), progress_notes=progress),
            char_cap=2000,
        )
        assert len(bundle.canonical_json) <= 2000

    def test_truncate_to_char_cap_returns_subsequence(self) -> None:
        # Property: the truncator never invents items, only drops or shrinks
        # them. The returned items must be a subsequence (by id) of the input.
        items = tuple(
            EvidenceItem(
                id=f"E{i}",
                source="MED",
                timestamp_utc=ANCHOR_DT - timedelta(hours=i),
                payload={"drug": "X" * 200},
            )
            for i in range(1, 6)
        )
        out = truncate_to_char_cap(items=items, char_cap=400)
        out_ids = [it.id for it in out]
        in_ids = [it.id for it in items]
        # Every output id is one of the input ids, in original order.
        assert all(oid in in_ids for oid in out_ids)
        assert out_ids == [oid for oid in in_ids if oid in out_ids]


class TestProgressNoteCap8:
    """When more than 8 IPDADMPROGRESS notes fit the time window, the
    bundle keeps the 8 closest to the anchor. Per issue #16 body."""

    def test_cap_at_8_entries_keeps_closest(self) -> None:
        # 12 candidates, all in window; expect 8 closest by absolute offset.
        notes = tuple(_progress(offset_hours=h) for h in (-12, -10, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8))
        # Assemble a bundle without focus_notes to keep the test focused.
        inputs = EvidenceInputs(anchor=_anchor(), progress_notes=notes)
        bundle = build_evidence_bundle(inputs=inputs)
        progress_items = _items_by_source(bundle, "IPDADMPROGRESS")
        assert len(progress_items) == 8
        kept_offsets = sorted(
            abs((it.timestamp_utc - ANCHOR_DT).total_seconds() / 3600.0)  # type: ignore[operator]
            for it in progress_items
        )
        # The 8 smallest absolute offsets among the 12 candidates.
        # Symmetric inputs at +/-{1,2,4,6} all make the cut; +/-8 onwards drop.
        assert kept_offsets == [1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0]


# =============================================================================
# Edge cases
# =============================================================================


class TestEmptyInputs:
    """A minimal anchor-only bundle still serializes + hashes deterministically."""

    def test_no_records_yields_empty_items_and_valid_hash(self) -> None:
        bundle = build_evidence_bundle(inputs=EvidenceInputs(anchor=_anchor()))
        assert bundle.items == ()
        # Hash is still a 64-char sha256 hex of the canonical JSON of the
        # anchor-only bundle.
        assert len(bundle.bundle_hash) == 64
        assert bundle.bundle_hash == hashlib.sha256(
            bundle.canonical_json.encode("utf-8")
        ).hexdigest()
