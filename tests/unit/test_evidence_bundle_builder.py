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
    EvidenceBundleTooLargeError,
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


def _hb(
    *,
    offset_hours: float,
    value: float = 7.5,
    source: str = "HEMATOLOGY",
    item_no: int = 1,
) -> HbRecord:
    return HbRecord(
        timestamp=ANCHOR_DT + timedelta(hours=offset_hours),
        value_g_dl=value,
        source=source,  # type: ignore[arg-type]
        item_no=item_no,
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


def _valid_anchor_envelope() -> dict[str, object]:
    """Return a minimal valid anchor envelope dict for canonical-JSON tests.

    The model_validator on EvidenceBundle requires the anchor to have
    exactly {order_datetime, hn_hash, an_hash, products} with the right
    types. Tests that construct EvidenceBundle directly need this shape;
    builder-emitted bundles already produce it via _bundle_envelope."""
    return {
        "an_hash": "an-test",
        "hn_hash": "hn-test",
        "order_datetime": "2026-05-15T12:00:00+00:00",
        "products": ["LPRC"],
    }


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
                item_no=1,
            )

    def test_hb_rejects_out_of_range_value(self) -> None:
        # Mirrors HbObservation: < 2.0 or > 25.0 is a transcription error.
        with pytest.raises(ValidationError):
            HbRecord(
                timestamp=ANCHOR_DT,
                value_g_dl=1.9,
                source="HEMATOLOGY",
                item_no=1,
            )
        with pytest.raises(ValidationError):
            HbRecord(
                timestamp=ANCHOR_DT,
                value_g_dl=25.1,
                source="HEMATOLOGY",
                item_no=1,
            )

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

    def test_evidence_item_rejects_naive_timestamp_utc(self) -> None:
        # The UTC contract is the project-wide invariant (CONTEXT.md
        # 'tz-aware UTC'); the field name promises UTC and the canonical
        # serializer would otherwise emit a no-offset string for a naive
        # value, breaking replay across time zones. Reject at construction.
        with pytest.raises(ValidationError):
            EvidenceItem(
                id="E1",
                source="IPDADMPROGRESS",
                timestamp_utc=datetime(2026, 5, 15, 12, 0, 0),  # naive
                payload={"sections": []},
            )

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

    def test_bundle_rejects_mismatched_hash_vs_canonical_json(self) -> None:
        # Audit-chain invariant: if a downstream caller pairs a real
        # canonical_json with a forged or stale hash, the bundle would
        # silently lie about its identity. The model recomputes
        # sha256(canonical_json) and rejects on mismatch.
        canonical_json = canonical_serialize({"anchor": _valid_anchor_envelope(), "items": []})
        wrong_hash = "0" * 64  # well-formed but wrong
        with pytest.raises(ValidationError, match="does not match"):
            EvidenceBundle(
                items=(),
                canonical_json=canonical_json,
                bundle_hash=wrong_hash,
            )

    def test_bundle_accepts_correctly_computed_hash(self) -> None:
        # Sanity: when the hash is genuinely the sha256 of canonical_json
        # AND canonical_json is a proper envelope, construction succeeds.
        canonical_json = canonical_serialize({"anchor": _valid_anchor_envelope(), "items": []})
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        bundle = EvidenceBundle(
            items=(), canonical_json=canonical_json, bundle_hash=right_hash
        )
        assert bundle.bundle_hash == right_hash

    def test_bundle_rejects_invalid_json_in_canonical_json(self) -> None:
        # canonical_json must parse — bytes that pass the hash check but
        # are not JSON shouldn't fool the audit chain.
        bad = "not json at all"
        bad_hash = hashlib.sha256(bad.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="not valid JSON"):
            EvidenceBundle(items=(), canonical_json=bad, bundle_hash=bad_hash)

    def test_bundle_rejects_non_canonical_form(self) -> None:
        # canonical_json must equal canonical_serialize(json.loads(...)).
        # A JSON-valid but non-canonical formatting (no indent, unsorted
        # keys, missing NFC) is rejected so downstream tooling can't
        # carry a self-consistent hash for non-canonical bytes.
        non_canonical = '{"items": [], "anchor": {}}'  # unsorted keys, no indent
        non_canonical_hash = hashlib.sha256(
            non_canonical.encode("utf-8")
        ).hexdigest()
        with pytest.raises(ValidationError, match="canonical form"):
            EvidenceBundle(
                items=(),
                canonical_json=non_canonical,
                bundle_hash=non_canonical_hash,
            )

    def test_bundle_rejects_items_count_mismatch(self) -> None:
        # canonical_json says 1 item but EvidenceBundle.items is empty.
        # Without this check, a forged bundle could carry items in the
        # bytes (and thus the hash) that don't match the model's items.
        envelope = {
            "anchor": _valid_anchor_envelope(),
            "items": [
                {
                    "id": "E1",
                    "source": "Diagnosis",
                    "timestamp_utc": None,
                    "payload": {"icd10": "D50.9"},
                }
            ],
        }
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="items count"):
            EvidenceBundle(
                items=(),  # empty model items vs 1 in canonical_json
                canonical_json=canonical_json,
                bundle_hash=right_hash,
            )

    def test_bundle_rejects_items_id_mismatch(self) -> None:
        # canonical_json's item ID disagrees with EvidenceBundle.items[0].id.
        item = EvidenceItem(
            id="E1",
            source="Diagnosis",
            timestamp_utc=None,
            payload={"icd10": "D50.9"},
        )
        # Build canonical_json with a DIFFERENT id at the same position.
        envelope = {
            "anchor": _valid_anchor_envelope(),
            "items": [
                {
                    "id": "E2",  # mismatched id
                    "source": "Diagnosis",
                    "timestamp_utc": None,
                    "payload": {"icd10": "D50.9"},
                }
            ],
        }
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="items disagree"):
            EvidenceBundle(
                items=(item,),
                canonical_json=canonical_json,
                bundle_hash=right_hash,
            )

    def test_bundle_rejects_items_payload_mismatch(self) -> None:
        # Same id/source/timestamp but different payload contents.
        item = EvidenceItem(
            id="E1",
            source="Diagnosis",
            timestamp_utc=None,
            payload={"icd10": "D50.9"},
        )
        envelope = {
            "anchor": _valid_anchor_envelope(),
            "items": [
                {
                    "id": "E1",
                    "source": "Diagnosis",
                    "timestamp_utc": None,
                    "payload": {"icd10": "X99.9"},  # different code
                }
            ],
        }
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="items disagree"):
            EvidenceBundle(
                items=(item,),
                canonical_json=canonical_json,
                bundle_hash=right_hash,
            )

    def test_bundle_rejects_envelope_missing_anchor(self) -> None:
        # Envelope must have exactly {anchor, items} — round-11 only
        # checked items presence, but a missing anchor is silent loss
        # of the decision-time context the audit chain depends on.
        envelope = {"items": []}
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="envelope"):
            EvidenceBundle(
                items=(),
                canonical_json=canonical_json,
                bundle_hash=right_hash,
            )

    def test_bundle_rejects_envelope_with_extra_top_level_key(self) -> None:
        # Builder emits exactly {anchor, items}. Extras are upstream
        # drift; reject so the audit chain has a single canonical shape.
        envelope = {"anchor": _valid_anchor_envelope(), "items": [], "extra": "leak"}
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="envelope"):
            EvidenceBundle(
                items=(),
                canonical_json=canonical_json,
                bundle_hash=right_hash,
            )

    def test_bundle_rejects_empty_anchor(self) -> None:
        # Round-12 only required anchor to be a dict; an empty {} would
        # silently lose all decision context (an_hash, hn_hash,
        # order_datetime, products) while keeping a self-consistent
        # bundle_hash. Round-14 locks the anchor key set.
        envelope = {"anchor": {}, "items": []}
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="anchor"):
            EvidenceBundle(
                items=(),
                canonical_json=canonical_json,
                bundle_hash=right_hash,
            )

    def test_bundle_rejects_anchor_with_extra_key(self) -> None:
        anchor = _valid_anchor_envelope()
        anchor["extra_chart_field"] = "leak"
        envelope = {"anchor": anchor, "items": []}
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="anchor"):
            EvidenceBundle(
                items=(),
                canonical_json=canonical_json,
                bundle_hash=right_hash,
            )

    def test_bundle_rejects_anchor_with_non_string_hn_hash(self) -> None:
        anchor = _valid_anchor_envelope()
        anchor["hn_hash"] = 12345  # type: ignore[assignment]
        envelope = {"anchor": anchor, "items": []}
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="hn_hash"):
            EvidenceBundle(
                items=(),
                canonical_json=canonical_json,
                bundle_hash=right_hash,
            )

    def test_bundle_rejects_unparseable_order_datetime(self) -> None:
        # The anchor moment must be reconstructable for replay; a non-
        # ISO-8601 string would silently carry an unreconstructable
        # decision time alongside a self-consistent hash.
        anchor = _valid_anchor_envelope()
        anchor["order_datetime"] = "not a date"
        envelope = {"anchor": anchor, "items": []}
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="ISO 8601"):
            EvidenceBundle(
                items=(),
                canonical_json=canonical_json,
                bundle_hash=right_hash,
            )

    def test_bundle_rejects_naive_order_datetime(self) -> None:
        # Project tz contract: every persisted timestamp is tz-aware.
        # A naive ISO string would lose the time-zone reference and
        # break re-windowing across regions.
        anchor = _valid_anchor_envelope()
        anchor["order_datetime"] = "2026-05-15T12:00:00"  # no offset
        envelope = {"anchor": anchor, "items": []}
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="tz-aware"):
            EvidenceBundle(
                items=(),
                canonical_json=canonical_json,
                bundle_hash=right_hash,
            )

    def test_bundle_rejects_non_utc_order_datetime(self) -> None:
        # Builder always normalizes to UTC; an Asia/Bangkok-offset
        # string is upstream drift — reject so persisted bundles are
        # all comparable on a single clock.
        anchor = _valid_anchor_envelope()
        anchor["order_datetime"] = "2026-05-15T19:00:00+07:00"
        envelope = {"anchor": anchor, "items": []}
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="UTC"):
            EvidenceBundle(
                items=(),
                canonical_json=canonical_json,
                bundle_hash=right_hash,
            )

    def test_bundle_rejects_anchor_with_non_string_products_element(self) -> None:
        anchor = _valid_anchor_envelope()
        anchor["products"] = ["LPRC", 42]  # type: ignore[list-item]
        envelope = {"anchor": anchor, "items": []}
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="products"):
            EvidenceBundle(
                items=(),
                canonical_json=canonical_json,
                bundle_hash=right_hash,
            )

    def test_bundle_rejects_anchor_not_a_dict(self) -> None:
        envelope = {"anchor": "not a dict", "items": []}
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="anchor"):
            EvidenceBundle(
                items=(),
                canonical_json=canonical_json,
                bundle_hash=right_hash,
            )

    def test_bundle_rejects_items_source_mismatch(self) -> None:
        # Same id but different source classification.
        item = EvidenceItem(
            id="E1",
            source="Diagnosis",
            timestamp_utc=None,
            payload={"icd10": "D50.9"},
        )
        envelope = {
            "anchor": _valid_anchor_envelope(),
            "items": [
                {
                    "id": "E1",
                    "source": "MED",  # mismatched source
                    "timestamp_utc": None,
                    "payload": {"icd10": "D50.9"},
                }
            ],
        }
        canonical_json = canonical_serialize(envelope)
        right_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        with pytest.raises(ValidationError, match="items disagree"):
            EvidenceBundle(
                items=(item,),
                canonical_json=canonical_json,
                bundle_hash=right_hash,
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

    def test_exact_7d_boundary_dropped_matching_hb_lookup(self) -> None:
        # bba.hb_lookup uses STRICT inequality (anchor - timestamp <
        # _LOOKBACK), so an Hb at exactly 7 d old is invisible to the
        # deterministic classifier. The bundle MUST drop it too — otherwise
        # the LLM sees evidence the classifier never considered, breaking
        # the audit's input consistency.
        bundle = _build_minimal(hb_history=(_hb(offset_hours=-168.0),))  # exactly 7d
        assert _items_by_source(bundle, "Lab") == ()

    def test_just_inside_7d_boundary_kept(self) -> None:
        # 7 d minus 1 second is still inside (matches hb_lookup).
        bundle = _build_minimal(
            hb_history=(_hb(offset_hours=-168.0 + (1.0 / 3600.0)),)
        )
        assert len(_items_by_source(bundle, "Lab")) == 1


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

    def test_inline_headers_on_one_line_split_correctly(self) -> None:
        # The clinical reality: many notes pack S/O/A/P onto a single line.
        # The line-anchored predecessor classified the entire run as
        # SUBJECTIVE, which inverted the AC's A/P-first priority and let
        # truncation drop the clinician's diagnosis (ASSESSMENT) first.
        text = "S: tired O: BP 90/60 A: anemia P: PRBC"
        sections = parse_soap_sections(text)
        assert sections["SUBJECTIVE"] == "tired"
        assert sections["OBJECTIVE"] == "BP 90/60"
        assert sections["ASSESSMENT"] == "anemia"
        assert sections["PLAN"] == "PRBC"

    def test_pre_first_header_text_falls_into_objective_preamble(self) -> None:
        # Chart metadata before the first SOAP header (e.g. timestamp,
        # author signature) is implicitly OBJECTIVE per the column-name
        # default — never silently dropped.
        text = "  preamble note S: tired"
        sections = parse_soap_sections(text)
        assert sections["OBJECTIVE"] == "preamble note"
        assert sections["SUBJECTIVE"] == "tired"

    def test_BP_does_not_match_P_header(self) -> None:
        # "BP 90/60" must not produce a PLAN match — "P" is not preceded
        # by a whitespace boundary.
        sections = parse_soap_sections("BP 90/60 PR 80")
        assert sections["PLAN"] == ""
        assert sections["OBJECTIVE"] == "BP 90/60 PR 80"


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


class TestFocusEmissionOrderInBundle:
    """Bundle-level: focus items emit in closest-first per side, before|after.

    The 5+5 split helper ranks by proximity to the anchor — this test locks
    in that the BUILDER preserves that order rather than re-sorting. The
    AC ("closest to order time first") is about evidence-ID assignment in
    the final bundle, not just helper output, because the LLM cites items
    by their bundle ID and reads them in emission order."""

    def test_focus_emission_order_is_closest_first_per_side(self) -> None:
        # Inputs deliberately shuffled to confirm the order is derived from
        # proximity, not input position.
        notes = tuple(_focus(offset_hours=h) for h in (-3, 2, -1, 3, -2, 1))
        inputs = EvidenceInputs(anchor=_anchor(), focus_notes=notes)
        bundle = build_evidence_bundle(inputs=inputs)
        focus_items = _items_by_source(bundle, "IPDNRFOCUSDT")
        offsets = [
            round((it.timestamp_utc - ANCHOR_DT).total_seconds() / 3600.0, 1)  # type: ignore[operator]
            for it in focus_items
        ]
        # Before-side closest-first (-1, -2, -3), then after-side closest-first
        # (+1, +2, +3). Chronological-ascending order would be (-3, -2, -1,
        # +1, +2, +3) — the WRONG answer the codex review flagged.
        assert offsets == [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0]

    def test_focus_emission_at_5_5_cap_emits_closest_first(self) -> None:
        # Saturate both sides; the 5 nearest pre-anchor and 5 nearest
        # post-anchor must come back in closest-first sequence.
        before = tuple(_focus(offset_hours=-h) for h in (1, 2, 3, 4, 5, 6, 7))
        after = tuple(_focus(offset_hours=h) for h in (1, 2, 3, 4, 5, 6, 7))
        inputs = EvidenceInputs(
            anchor=_anchor(),
            focus_notes=before + after,
        )
        bundle = build_evidence_bundle(inputs=inputs)
        focus_items = _items_by_source(bundle, "IPDNRFOCUSDT")
        offsets = [
            round((it.timestamp_utc - ANCHOR_DT).total_seconds() / 3600.0, 1)  # type: ignore[operator]
            for it in focus_items
        ]
        assert offsets == [-1.0, -2.0, -3.0, -4.0, -5.0, 1.0, 2.0, 3.0, 4.0, 5.0]


class TestTimestampTieDeterminism:
    """When records share an exact timestamp, the kept-set + emission order
    + bundle hash must be a function of CONTENT, not input position.

    Python's stable sort means a key that ties on timestamp leaves input
    order untouched — operationally invisible most of the time, but lethal
    for the AC ('input reorderings produce same hash') because chart
    exports do produce minute-resolution timestamps that collide. The fix
    is a TOTAL sort key with a content-derived tiebreak (text / drug /
    vital values), tested below for each affected source."""

    def test_focus_notes_at_same_timestamp_hash_invariant_under_reorder(self) -> None:
        # Two pre-anchor focus notes charted at the same minute (-1h),
        # plus two post-anchor at the same minute (+2h).
        notes_a = (
            _focus(offset_hours=-1, text="alpha"),
            _focus(offset_hours=-1, text="bravo"),
            _focus(offset_hours=2, text="charlie"),
            _focus(offset_hours=2, text="delta"),
        )
        notes_b = tuple(reversed(notes_a))
        bundle_a = _build_minimal(focus_notes=notes_a)
        bundle_b = _build_minimal(focus_notes=notes_b)
        assert bundle_a.bundle_hash == bundle_b.bundle_hash
        # And the emission order is content-derived: alphabetical text
        # tiebreak under reverse=True for the before side, ascending for the
        # after side. Both sides start with text-DESC under the helper's
        # ``reverse=True`` (timestamp DESC implies text DESC too).
        focus_ids = [it.id for it in _items_by_source(bundle_a, "IPDNRFOCUSDT")]
        focus_payloads = [
            it.payload["text"]
            for it in _items_by_source(bundle_a, "IPDNRFOCUSDT")
        ]
        assert len(focus_ids) == 4
        # Before side: timestamp DESC, then text DESC under the same reverse.
        # Both at -1h, so text-DESC: "bravo" then "alpha".
        # After side: timestamp ASC, then text ASC.
        # Both at +2h, so text-ASC: "charlie" then "delta".
        assert focus_payloads == ["bravo", "alpha", "charlie", "delta"]

    def test_progress_notes_at_same_timestamp_kept_set_invariant_under_reorder(self) -> None:
        # 9 progress notes — 8 distinct + 2 sharing offset -1h with different
        # text. Cap is 8; one of the -1h pair must drop deterministically.
        notes_a = (
            _progress(offset_hours=-1, text="alpha"),
            _progress(offset_hours=-1, text="bravo"),
            _progress(offset_hours=-2),
            _progress(offset_hours=-3),
            _progress(offset_hours=-4),
            _progress(offset_hours=-5),
            _progress(offset_hours=-6),
            _progress(offset_hours=-7),
            _progress(offset_hours=-8),
        )
        notes_b = tuple(reversed(notes_a))
        bundle_a = _build_minimal(progress_notes=notes_a)
        bundle_b = _build_minimal(progress_notes=notes_b)
        assert bundle_a.bundle_hash == bundle_b.bundle_hash

    def test_diagnosis_none_vs_empty_description_hash_invariant(self) -> None:
        # Same ICD-10, one record with description=None, one with "".
        # Their PAYLOADS differ (None omits the field; "" emits
        # "description": ""), but the sort key must order them
        # deterministically so reversed input yields the same E1/E2
        # assignment and the same bundle hash.
        d_none = DiagnosisRecord(icd10="D50.9", description=None)
        d_empty = DiagnosisRecord(icd10="D50.9", description="")
        a = build_evidence_bundle(
            inputs=EvidenceInputs(anchor=_anchor(), diagnoses=(d_none, d_empty))
        )
        b = build_evidence_bundle(
            inputs=EvidenceInputs(anchor=_anchor(), diagnoses=(d_empty, d_none))
        )
        assert a.bundle_hash == b.bundle_hash

    def test_vitals_at_same_timestamp_and_source_hash_invariant(self) -> None:
        # Two vitals snapshots from the same source at the same moment with
        # different SBP. Operationally rare but possible if upstream pushes
        # duplicate-timestamp rows; without the tiebreak, input order would
        # leak through.
        v1 = VitalsRecord(
            timestamp=ANCHOR_DT, source="IPDADMPROGRESS", sbp=110, hr=80
        )
        v2 = VitalsRecord(
            timestamp=ANCHOR_DT, source="IPDADMPROGRESS", sbp=120, hr=88
        )
        a = _build_minimal(vitals=(v1, v2))
        b = _build_minimal(vitals=(v2, v1))
        assert a.bundle_hash == b.bundle_hash


class TestCanonicalJSONNFCNormalizesKeys:
    """Mapping keys must be NFC-normalized too, not just values.

    The canonical-JSON contract is a property of THE WHOLE PAYLOAD: every
    string in the bundle is NFC. If only values are normalized,
    ``{"café": 1}`` (NFC key) and ``{"cafe\\u0301": 1}`` (NFD key)
    produce different bytes and different bundle hashes despite carrying
    the same data — direct AC violation. Failing loud on collisions
    (two source keys that normalize to the same string) prevents
    silent overwrite, which would otherwise hide a real upstream bug."""

    def test_nfd_and_nfc_keys_produce_identical_canonical_output(self) -> None:
        nfc_key = unicodedata.normalize("NFC", "café")
        nfd_key = unicodedata.normalize("NFD", "café")
        assert nfc_key != nfd_key, "fixture is malformed if NFD == NFC"
        a = canonical_serialize({nfc_key: 1})
        b = canonical_serialize({nfd_key: 1})
        assert a == b

    def test_collision_after_normalization_raises(self) -> None:
        nfc_key = unicodedata.normalize("NFC", "café")
        nfd_key = unicodedata.normalize("NFD", "café")
        # Both spellings in the same dict → after NFC normalization they
        # collide. Silently overwriting would hide the upstream encoding
        # bug; raise instead.
        with pytest.raises(ValueError, match="duplicate key"):
            canonical_serialize({nfc_key: 1, nfd_key: 2})

    def test_nested_dict_keys_also_normalized(self) -> None:
        # Defense in depth: the recursion must apply key normalization at
        # every nesting level, not just the top-level dict.
        nfc = unicodedata.normalize("NFC", "café")
        nfd = unicodedata.normalize("NFD", "café")
        a = canonical_serialize({"outer": {nfc: 1}})
        b = canonical_serialize({"outer": {nfd: 1}})
        assert a == b


class TestCanonicalJSONRejectsNonFiniteFloats:
    """Non-finite floats (NaN, +Inf, -Inf) must NEVER reach the bundle.

    Python's ``json.dumps`` defaults emit ``NaN`` / ``Infinity`` /
    ``-Infinity``, which are NOT valid JSON per RFC 7159 / 8259. The
    bundle-hash AC requires bytes that round-trip through every JSON
    parser; downstream tooling (deid_redactor, prompt_builder, audit
    re-hydration) would either reject these tokens or interpret them
    inconsistently. The serializer fails loud at the boundary so the
    upstream bug (a buggy vitals extraction, an arithmetic 0/0) is
    surfaced where it can be fixed."""

    def test_nan_in_payload_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            canonical_serialize({"x": float("nan")})

    def test_positive_infinity_in_payload_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            canonical_serialize({"x": float("inf")})

    def test_negative_infinity_in_payload_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            canonical_serialize({"x": float("-inf")})

    def test_nan_inside_nested_list_raises(self) -> None:
        # Defense-in-depth: the recursion must fail at any depth, not just
        # at the top-level dict.
        with pytest.raises(ValueError, match="non-finite"):
            canonical_serialize({"items": [{"bt": float("nan")}]})

    def test_finite_floats_still_serialize(self) -> None:
        # Sanity: 36.5 (a normal body-temperature value) still works.
        out = canonical_serialize({"bt": 36.5})
        assert "36.5" in out


class TestVitalsRecordEnforcesSanityBounds:
    """VitalsRecord mirrors :mod:`bba.vitals_extractor.bounds` so a buggy
    upstream caller cannot persist clinically impossible vitals (sbp=-1,
    hr=999) as canonical evidence. Hb already mirrors hb_lookup bounds;
    Vitals consistency closes the same gap for the other half of the
    deterministic-classifier inputs."""

    def test_sbp_below_lower_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", sbp=59)

    def test_sbp_above_upper_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", sbp=221)

    def test_dbp_below_lower_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", dbp=29)

    def test_dbp_above_upper_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", dbp=151)

    def test_hr_below_lower_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", hr=29)

    def test_hr_above_upper_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", hr=999)

    def test_rr_below_lower_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", rr=4)

    def test_rr_above_upper_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", rr=51)

    def test_bt_below_lower_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", bt=29.9)

    def test_bt_above_upper_bound_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", bt=43.1)

    def test_inclusive_bounds_accepted(self) -> None:
        # Boundary values themselves should be valid (inclusive bounds).
        v = VitalsRecord(
            timestamp=ANCHOR_DT,
            source="IPDADMPROGRESS",
            sbp=60,
            dbp=30,
            hr=30,
            rr=5,
            bt=30.0,
        )
        assert v.sbp == 60
        v2 = VitalsRecord(
            timestamp=ANCHOR_DT,
            source="IPDADMPROGRESS",
            sbp=220,
            dbp=150,
            hr=200,
            rr=50,
            bt=43.0,
        )
        assert v2.bt == 43.0


class TestVitalsRecordRejectsNonFiniteBt:
    """Defense in depth: VitalsRecord.bt rejects NaN / Inf at construction
    so the bundle-hash failure surfaces at the upstream call site, not
    deep inside :func:`build_evidence_bundle`."""

    def test_nan_bt_raises(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", bt=float("nan"))

    def test_positive_inf_bt_raises(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", bt=float("inf"))

    def test_negative_inf_bt_raises(self) -> None:
        with pytest.raises(ValidationError):
            VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", bt=float("-inf"))

    def test_finite_bt_still_constructs(self) -> None:
        v = VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", bt=37.0)
        assert v.bt == 37.0


class TestEmptyProgressItemsNeverConstructed:
    """Blank or header-only progress notes must NEVER produce an
    EvidenceItem — not even under the normal-cap path. The round-5 fix
    handled the truncation path; this class locks in that the construction
    path also drops empty payloads, since `_enforce_char_cap` returns
    early when the bundle fits and would never see them otherwise."""

    def test_blank_progress_note_does_not_emit_item(self) -> None:
        # Empty text → parse_soap_sections returns all-empty → payload
        # has no sections → no EvidenceItem.
        bundle = _build_minimal(progress_notes=(_progress(offset_hours=-1, text=""),))
        assert _items_by_source(bundle, "IPDADMPROGRESS") == ()

    def test_header_only_progress_note_does_not_emit_item(self) -> None:
        # Headers with no content → all sections empty → no EvidenceItem.
        bundle = _build_minimal(
            progress_notes=(_progress(offset_hours=-1, text="S:\nO:\nA:\nP:"),)
        )
        assert _items_by_source(bundle, "IPDADMPROGRESS") == ()

    def test_all_null_vitals_does_not_emit_item(self) -> None:
        # An in-window VitalsRecord with every measurement None has
        # nothing for the LLM to cite — only note_source provenance
        # metadata. Emitting it would burn a bundle E_N slot on a
        # dead reference (same dead-citation shape we close for
        # blank progress notes).
        v = VitalsRecord(
            timestamp=ANCHOR_DT,
            source="IPDADMPROGRESS",
            sbp=None,
            dbp=None,
            hr=None,
            rr=None,
            bt=None,
        )
        bundle = _build_minimal(vitals=(v,))
        assert _items_by_source(bundle, "Vitals") == ()

    def test_partial_vitals_still_emits_item(self) -> None:
        # At least one measurement populated → real evidence; emit it.
        v = VitalsRecord(timestamp=ANCHOR_DT, source="IPDADMPROGRESS", sbp=110)
        bundle = _build_minimal(vitals=(v,))
        vitals_items = _items_by_source(bundle, "Vitals")
        assert len(vitals_items) == 1
        assert vitals_items[0].payload["sbp"] == 110

    def test_blank_focus_does_not_consume_cap_slots(self) -> None:
        # Mirrors the round-7 progress fix for IPDNRFOCUSDT: 5 closer-
        # to-anchor blank focus notes on each side could otherwise
        # consume CAP_FOCUS_BEFORE + CAP_FOCUS_AFTER and evict valid
        # farther notes. The pre-cap content filter ensures the valid
        # notes survive.
        blanks_before = tuple(
            _focus(offset_hours=-h, text="") for h in (0.5, 1.0, 1.5, 2.0, 2.5)
        )
        blanks_after = tuple(
            _focus(offset_hours=h, text="   ") for h in (0.5, 1.0, 1.5, 2.0, 2.5)
        )
        valid_before = _focus(offset_hours=-12.0, text="pain managed")
        valid_after = _focus(offset_hours=12.0, text="post-transfusion stable")
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(
                anchor=_anchor(),
                focus_notes=blanks_before + blanks_after + (valid_before, valid_after),
            )
        )
        focus_items = _items_by_source(bundle, "IPDNRFOCUSDT")
        assert len(focus_items) == 2
        texts = {it.payload["text"] for it in focus_items}
        assert texts == {"pain managed", "post-transfusion stable"}

    def test_blank_progress_does_not_consume_cap_slots(self) -> None:
        # Realistic pathology: 8 closer-to-anchor header-only progress
        # entries (which a busy nurse leaves as SOAP shells before
        # writing real content) plus one farther but content-bearing
        # note. Without a pre-cap content filter, the 8 blanks would
        # consume CAP_PROGRESS=8, exclude the valid 9th note, and then
        # all be skipped at item construction — silent evidence loss.
        # The pre-cap filter ensures the valid note survives.
        blanks = tuple(
            _progress(offset_hours=-h, text="S:\nO:\nA:\nP:")
            for h in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
        )
        valid = _progress(
            offset_hours=-12.0,
            text="A: anemia clear, plan PRBC 1U",
        )
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(
                anchor=_anchor(),
                progress_notes=blanks + (valid,),
            )
        )
        progress_items = _items_by_source(bundle, "IPDADMPROGRESS")
        assert len(progress_items) == 1, (
            "blank notes consumed cap slots; the valid farther note was "
            "evicted and the bundle has zero progress evidence"
        )
        labels = {s["label"] for s in progress_items[0].payload["sections"]}
        assert "ASSESSMENT" in labels

    def test_partially_blank_progress_note_emits_only_non_empty_sections(self) -> None:
        # ASSESSMENT has content, others are blank — item ships with just
        # the non-empty section; no dead headers in the payload.
        bundle = _build_minimal(
            progress_notes=(
                _progress(offset_hours=-1, text="S:\nO:\nA: anemia\nP:"),
            )
        )
        progress_items = _items_by_source(bundle, "IPDADMPROGRESS")
        assert len(progress_items) == 1
        sections = progress_items[0].payload["sections"]
        labels = {s["label"] for s in sections}
        assert labels == {"ASSESSMENT"}


class TestEmptyProgressItemsPrunedAfterTruncation:
    """When section truncation drops every section of a progress item, the
    item itself must drop too. Otherwise the bundle ships an E_N citation
    pointing at a sections=[] payload — the LLM sees an evidence ID with
    no quoteable content and quote_grounder treats it as a dead reference."""

    def test_progress_item_pruned_when_all_sections_dropped(self) -> None:
        # Build a payload where each section is huge enough that even
        # ASSESSMENT alone (~5K chars) exceeds the small cap. The
        # truncator should drop S, O, P, then A — leaving sections=[]
        # — then prune the empty item entirely.
        big = lambda label: f"{label}: " + ("ก" * 5000)  # noqa: E731
        text = "\n".join(big(c) for c in ("S", "O", "A", "P"))
        # Add a small Hb so we can confirm OTHER sources still appear
        # after the empty progress is pruned.
        inputs = EvidenceInputs(
            anchor=_anchor(),
            progress_notes=(_progress(offset_hours=-1, text=text),),
            hb_history=(_hb(offset_hours=-1, value=7.5),),
        )
        bundle = build_evidence_bundle(inputs=inputs, char_cap=2500)
        assert len(bundle.canonical_json) <= 2500
        # The empty progress item must NOT be in the bundle.
        for item in bundle.items:
            if item.source == "IPDADMPROGRESS":
                assert item.payload.get("sections"), (
                    "IPDADMPROGRESS item with no sections leaks an empty "
                    "evidence ID into the bundle"
                )

    def test_lab_survives_when_progress_is_pruned(self) -> None:
        # Same setup as above; Lab has higher source-order index but
        # carries actionable Hb data. After pruning the empty progress,
        # the Lab item must survive (not get dropped by the whole-item
        # tail-truncation that would otherwise prefer to drop it last).
        big = lambda label: f"{label}: " + ("ก" * 5000)  # noqa: E731
        text = "\n".join(big(c) for c in ("S", "O", "A", "P"))
        inputs = EvidenceInputs(
            anchor=_anchor(),
            progress_notes=(_progress(offset_hours=-1, text=text),),
            hb_history=(_hb(offset_hours=-1, value=7.5),),
        )
        bundle = build_evidence_bundle(inputs=inputs, char_cap=2500)
        lab_items = _items_by_source(bundle, "Lab")
        assert len(lab_items) == 1


class TestImpossibleCharCapRaises:
    """Cap enforcement fails loud when even the anchor envelope exceeds cap.

    Returning silently with an over-budget bundle would violate the AC
    ("bundles never exceed token-budget proxy") and let the prompt_builder
    pass an oversized prompt to the LLM with no signal to route to a
    longer-context tier. Per the project's "fail loud" principle, the
    pipeline raises EvidenceBundleTooLargeError so the caller can react."""

    def test_tiny_cap_raises_on_anchor_only_inputs(self) -> None:
        inputs = EvidenceInputs(anchor=_anchor())
        with pytest.raises(EvidenceBundleTooLargeError):
            build_evidence_bundle(inputs=inputs, char_cap=10)

    def test_huge_anchor_field_raises_under_default_cap(self) -> None:
        # 5K + 5K = 10K of hash chars alone, well over the 8K default cap.
        inputs = EvidenceInputs(
            anchor=OrderAnchor(
                order_datetime=ANCHOR_DT,
                hn_hash="x" * 5000,
                an_hash="y" * 5000,
                products=("LPRC",),
            )
        )
        with pytest.raises(EvidenceBundleTooLargeError):
            build_evidence_bundle(inputs=inputs)

    def test_error_message_names_the_anchor_size_and_cap(self) -> None:
        # Operator-friendly message: the value the cap could have been to fit,
        # plus which fields are likely candidates to trim.
        inputs = EvidenceInputs(anchor=_anchor())
        with pytest.raises(EvidenceBundleTooLargeError) as exc_info:
            build_evidence_bundle(inputs=inputs, char_cap=5)
        msg = str(exc_info.value)
        assert "char_cap=5" in msg
        assert "anchor" in msg.lower()


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


class TestHbEmissionAndTruncationPriority:
    """Hb emission is newest-first, so the whole-item tail-drop in
    _enforce_char_cap discards stale Hb before recent Hb. The most-recent
    pre-order Hb is the decision-time anemia signal — losing it changes
    what the LLM is auditing."""

    def test_hb_newest_first_in_emission_order(self) -> None:
        # 4 Hb records spread over the lookback window.
        hbs = tuple(_hb(offset_hours=-h, value=10.0 + i * 0.1) for i, h in enumerate((24, 48, 72, 96)))
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(anchor=_anchor(), hb_history=hbs)
        )
        lab_items = _items_by_source(bundle, "Lab")
        offsets = [
            round((it.timestamp_utc - ANCHOR_DT).total_seconds() / 3600.0, 1)  # type: ignore[operator]
            for it in lab_items
        ]
        # Newest first: -24, -48, -72, -96 (oldest last).
        assert offsets == [-24.0, -48.0, -72.0, -96.0]

    def test_med_truncation_drops_oldest_first_within_med(self) -> None:
        # MED emission is NEWEST-first, so under cap pressure the
        # OLDEST (farthest from anchor) drops first. Without this, a
        # stale -72h medication would survive while a -1h medication
        # (immediate decision context that shaped the transfusion
        # request) would be evicted — exactly inverted from clinical
        # relevance.
        near = _med(offset_hours=-1, drug="ImmediateMed-" + "x" * 200)
        stale = _med(offset_hours=-72, drug="StaleMed-" + "y" * 200)
        anchor = OrderAnchor(
            order_datetime=ANCHOR_DT,
            hn_hash="x" * 200,
            an_hash="y" * 200,
            products=("LPRC",),
        )
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(anchor=anchor, meds=(near, stale)),
            char_cap=1100,
        )
        med_items = _items_by_source(bundle, "MED")
        assert len(med_items) == 1, (
            "Expected exactly one MED to survive under cap pressure"
        )
        assert med_items[0].payload["drug"].startswith("ImmediateMed"), (
            "Stale MED (-72h) survived while immediate MED (-1h) dropped; "
            "MED emission order is not newest-first"
        )

    def test_med_post_order_drops_before_pre_order_under_cap(self) -> None:
        # Round-19 added newest-first MED emission, but didn't split
        # pre-anchor (decision context) from post-anchor (treatment
        # AFTER the order). Without the split, +2h administration
        # could outlive a -1h decision-context med under cap pressure.
        # Audit-wise that's exactly inverted: the bundle would show
        # the LLM what was given AFTER the order while losing what
        # shaped the order.
        pre_immediate = _med(
            offset_hours=-1, drug="ImmediatePre-" + "x" * 200
        )
        pre_stale = _med(offset_hours=-72, drug="StalePre-" + "y" * 200)
        post = _med(offset_hours=2, drug="PostOrder-" + "z" * 200)
        anchor = OrderAnchor(
            order_datetime=ANCHOR_DT,
            hn_hash="x" * 200,
            an_hash="y" * 200,
            products=("LPRC",),
        )
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(
                anchor=anchor, meds=(post, pre_immediate, pre_stale)
            ),
            char_cap=1300,
        )
        med_items = _items_by_source(bundle, "MED")
        # Under cap pressure, expect the post-order MED to drop first
        # (lower priority); pre-order meds survive longer. The
        # immediate pre (-1h) survives until last.
        survivor_drugs = [it.payload["drug"] for it in med_items]
        # The post-order item must NOT survive on its own when pre meds
        # drop. When at least one MED survives, it must be a pre-order one.
        if med_items:
            assert any(d.startswith("ImmediatePre") for d in survivor_drugs), (
                f"Pre-order immediate MED was dropped before post-order; "
                f"survivors: {survivor_drugs}"
            )
            # If only one MED survives, it should be the immediate pre.
            if len(med_items) == 1:
                assert med_items[0].payload["drug"].startswith("ImmediatePre")

    def test_med_emits_pre_order_before_post_order(self) -> None:
        # Normal-cap bundle: pre-order meds emit BEFORE post-order so
        # the LLM reads decision context first, treatment-after-order
        # second.
        pre = _med(offset_hours=-1, drug="Pre")
        post = _med(offset_hours=2, drug="Post")
        bundle = _build_minimal(meds=(post, pre))
        med_items = _items_by_source(bundle, "MED")
        assert len(med_items) == 2
        assert med_items[0].payload["drug"] == "Pre"
        assert med_items[1].payload["drug"] == "Post"

    def test_med_emits_newest_first_in_normal_bundle(self) -> None:
        # Normal-cap bundle should also lead with the newest MED so the
        # LLM reads the most-relevant decision context first.
        near = _med(offset_hours=-1, drug="A")
        far = _med(offset_hours=-72, drug="B")
        bundle = _build_minimal(meds=(far, near))
        med_items = _items_by_source(bundle, "MED")
        assert len(med_items) == 2
        assert med_items[0].payload["drug"] == "A"  # newest first
        assert med_items[1].payload["drug"] == "B"

    def test_med_dropped_before_hb_under_tight_cap(self) -> None:
        # A dense MED list must NOT silently evict the decision-time Hb.
        # Without the source-aware drop priority, the global tail-drop
        # would discard every Lab item before any MED item (Lab ID >
        # MED ID under the canonical-source emission order).
        hbs = (_hb(offset_hours=-1, value=7.0),)
        meds = tuple(
            _med(offset_hours=-h, drug=f"Drug{h}-" + ("x" * 200))
            for h in (1, 2, 3, 4, 5, 6, 7, 8)
        )
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(
                anchor=_anchor(),
                hb_history=hbs,
                meds=meds,
            ),
            char_cap=1500,
        )
        lab_items = _items_by_source(bundle, "Lab")
        med_items = _items_by_source(bundle, "MED")
        assert len(lab_items) == 1, (
            "Hb dropped under tight cap while MED items survived; the "
            "decision-time anemia signal must outlast lower-priority MED."
        )
        assert len(med_items) < 8, (
            "MEDs should drop FIRST under tight cap; if all 8 survive, "
            "DROP_PRIORITY is not applied"
        )

    def test_hematology_emits_before_newer_poct_in_normal_bundle(self) -> None:
        # PRD §3 / bba.hb_lookup contract: HEMATOLOGY > POCT regardless
        # of recency. The NORMAL bundle (no cap pressure) must already
        # show HEMATOLOGY first so the LLM reads the same primary Hb
        # signal the deterministic classifier picked. Round-13 only
        # protected the cap-pressure case; this test locks the normal
        # emission order too.
        hb_hema = HbRecord(
            timestamp=ANCHOR_DT - timedelta(hours=6),
            value_g_dl=7.5,
            source="HEMATOLOGY",
            item_no=1,
        )
        hb_poct = HbRecord(
            timestamp=ANCHOR_DT - timedelta(hours=2),
            value_g_dl=8.0,
            source="POCT",
            item_no=1,
        )
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(
                anchor=_anchor(), hb_history=(hb_hema, hb_poct)
            )
        )
        lab_items = _items_by_source(bundle, "Lab")
        assert len(lab_items) == 2
        assert lab_items[0].payload["lab_source"] == "HEMATOLOGY", (
            "POCT (newer) emitted before HEMATOLOGY (older); the LLM "
            "would read a different primary Hb signal than the "
            "deterministic classifier"
        )
        assert lab_items[1].payload["lab_source"] == "POCT"

    def test_corrected_hb_row_kept_over_stale_under_tight_cap(self) -> None:
        # bba.hb_lookup._select_current resolves same-(source,timestamp)
        # ties by max item_no — later Lab rows are corrected/amended
        # results. The bundle must mirror that semantics under cap
        # pressure: when only one of two same-time HEMATOLOGY rows fits,
        # the higher item_no (corrected) survives. Otherwise the LLM
        # would see a stale value that the deterministic classifier
        # already overrode.
        stale_hb = HbRecord(
            timestamp=ANCHOR_DT - timedelta(hours=2),
            value_g_dl=8.0,  # stale higher value
            source="HEMATOLOGY",
            item_no=1,
        )
        corrected_hb = HbRecord(
            timestamp=ANCHOR_DT - timedelta(hours=2),
            value_g_dl=7.0,  # corrected lower value
            source="HEMATOLOGY",
            item_no=2,  # higher item_no = later insert / correction
        )
        anchor = OrderAnchor(
            order_datetime=ANCHOR_DT,
            hn_hash="x" * 200,
            an_hash="y" * 200,
            products=("LPRC",),
        )
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(
                anchor=anchor, hb_history=(stale_hb, corrected_hb)
            ),
            char_cap=900,
        )
        lab_items = _items_by_source(bundle, "Lab")
        assert len(lab_items) == 1, "Expected exactly one Hb to survive"
        assert lab_items[0].payload["item_no"] == 2, (
            "Stale Hb (item_no=1) survived over corrected (item_no=2); "
            "bundle does not mirror hb_lookup._select_current's "
            "max-item_no tiebreak"
        )
        assert lab_items[0].payload["value_g_dl"] == 7.0

    def test_hematology_kept_over_NEWER_poct_under_tight_cap(self) -> None:
        # PRD §3 / bba.hb_lookup: HEMATOLOGY (LABEXM 290095) is preferred
        # over POCT (LABEXM 500001) when ANY HEMATOLOGY result exists in
        # the 7-day lookback — INDEPENDENT of recency. The bundle must
        # honor this even under tight cap pressure where only one Hb
        # fits, so the LLM never sees a POCT-only Hb history when a
        # HEMATOLOGY value existed.
        hb_hema = HbRecord(
            timestamp=ANCHOR_DT - timedelta(hours=6),
            value_g_dl=7.5,
            source="HEMATOLOGY",
            item_no=1,
        )
        hb_poct = HbRecord(
            timestamp=ANCHOR_DT - timedelta(hours=2),
            value_g_dl=8.0,
            source="POCT",
            item_no=1,
        )
        anchor = OrderAnchor(
            order_datetime=ANCHOR_DT,
            hn_hash="x" * 200,
            an_hash="y" * 200,
            products=("LPRC",),
        )
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(anchor=anchor, hb_history=(hb_hema, hb_poct)),
            char_cap=850,
        )
        lab_items = _items_by_source(bundle, "Lab")
        assert len(lab_items) == 1, "Expected exactly one Hb to survive"
        assert lab_items[0].payload["lab_source"] == "HEMATOLOGY", (
            "POCT (newer) survived over HEMATOLOGY (older); bundle does "
            "not honor PRD §3 source preference under cap pressure"
        )

    def test_hematology_kept_before_poct_under_tight_cap(self) -> None:
        # PRD §3 / bba.hb_lookup contract: HEMATOLOGY (LABEXM 290095)
        # is preferred over POCT (LABEXM 500001). For tied
        # timestamp/value Hb pairs, the bundle must keep HEMATOLOGY
        # under tight cap pressure — without the source-rank tiebreak,
        # POCT would survive (alphabetic 'POCT' > 'HEMATOLOGY' under
        # reverse=True), inverting the source preference.
        hb_hema = HbRecord(
            timestamp=ANCHOR_DT, value_g_dl=7.5, source="HEMATOLOGY", item_no=1
        )
        hb_poct = HbRecord(
            timestamp=ANCHOR_DT, value_g_dl=7.5, source="POCT", item_no=1
        )
        # Pad anchor to push the bundle near the cap so one Hb must drop.
        anchor = OrderAnchor(
            order_datetime=ANCHOR_DT,
            hn_hash="x" * 200,
            an_hash="y" * 200,
            products=("LPRC",),
        )
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(anchor=anchor, hb_history=(hb_hema, hb_poct)),
            char_cap=850,
        )
        lab_items = _items_by_source(bundle, "Lab")
        assert len(lab_items) == 1, "Expected exactly one Hb to survive"
        assert lab_items[0].payload["lab_source"] == "HEMATOLOGY", (
            "POCT survived over HEMATOLOGY; source-rank tiebreak missing"
        )

    def test_diagnosis_survives_longest_under_extreme_cap(self) -> None:
        # Diagnosis is at the END of DROP_PRIORITY — it is the encounter
        # context and survives until the bundle structurally cannot fit.
        diagnoses = (DiagnosisRecord(icd10="D50.9"),)
        meds = tuple(
            _med(offset_hours=-h, drug=f"Drug{h}-" + ("x" * 200))
            for h in (1, 2, 3, 4)
        )
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(
                anchor=_anchor(),
                diagnoses=diagnoses,
                meds=meds,
            ),
            char_cap=900,
        )
        diag_items = _items_by_source(bundle, "Diagnosis")
        assert len(diag_items) == 1, (
            "Diagnosis must survive longest — it's the encounter context "
            "and bottommost in DROP_PRIORITY"
        )

    def test_hb_truncation_preserves_most_recent_hb(self) -> None:
        # Force whole-item tail-drop by padding the anchor near the cap.
        # With NEWEST-FIRST emission, the most recent Hb (-1 h) survives;
        # older labs drop first.
        hbs = tuple(_hb(offset_hours=-h, value=8.0) for h in (1, 24, 48, 72))
        anchor = OrderAnchor(
            order_datetime=ANCHOR_DT,
            hn_hash="x" * 200,
            an_hash="y" * 200,
            products=("LPRC",),
        )
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(anchor=anchor, hb_history=hbs),
            char_cap=950,
        )
        lab_items = _items_by_source(bundle, "Lab")
        assert len(lab_items) >= 1, "at least one Hb must survive truncation"
        # The first surviving Lab item is the most recent (offset -1 h).
        first_offset = round(
            (lab_items[0].timestamp_utc - ANCHOR_DT).total_seconds() / 3600.0, 1  # type: ignore[operator]
        )
        assert first_offset == -1.0


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
