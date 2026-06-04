"""Failing-first tests for the peri-operative narrative scan (Case 107).

The peri-op scan is a SEPARATE pass from :func:`scan_hemodynamics`. It scans
EVERY in-window note for three peri-operative facts that the LLM missed on
Case 107 / REQNO 68074627 because the structured procedure rows were empty and
the surgical detail lived only in a free-text IPDNRFOCUSDT nursing note:

* **surgical context** — a charted operation (ORIF/CRIF, "post-op", ผ่าตัด,
  "under GA/spinal", craniotomy, ...);
* **estimated blood loss** — the EBL volume, normalized to millilitres so a
  litre or "cc" charting cannot hide the magnitude;
* **intra-operative transfusion** — a specific blood component (LPRC, PRBC,
  FFP, ...) given *during* the operation, co-located with an intra-op marker.

The result is a fact-only :class:`PeriopSummary`. It carries no appropriateness
language and no "indicated"/"justified" verdict — peri-op context is a
supporting factor for the auditor and the LLM, never a standalone transfusion
verdict. These tests assert the WHY (surgery in free-text counts even when the
structured row is empty; EBL is reported in mL regardless of charted unit;
generic "blood" near "intra-op" is NOT an intra-op transfusion), not regex shape.

No implementation exists yet; the module-level imports double as the public-API
surface check.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bba.vitals_extractor import (
    PeriopFinding,
    PeriopSummary,
    VitalsNote,
    scan_periop,
)

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
# AC: A surgery named only in free-text is surgical context. Case 107's whole
# failure was the LLM trusting an empty structured procedure row over the prose;
# the scan must assert YES from the note alone.
# =============================================================================
class TestSurgicalContext:
    def test_orif_sets_surgical_context(self) -> None:
        summary = scan_periop([_note("IPDNRFOCUSDT", -60, "s/p ORIF Lt femur")])
        assert summary.surgical_context is True

    def test_crif_sets_surgical_context(self) -> None:
        # Case 107 charted CRIF/ORIF; CRIF must be recognised alongside ORIF.
        summary = scan_periop([_note("IPDNRFOCUSDT", -60, "post CRIF wrist")])
        assert summary.surgical_context is True

    def test_post_op_variants_set_surgical_context(self) -> None:
        for text in ("post-op day 1", "postop ward", "post op care"):
            summary = scan_periop([_note("IPDADMPROGRESS", 30, text)])
            assert summary.surgical_context is True, text

    def test_thai_surgery_word_sets_surgical_context(self) -> None:
        summary = scan_periop([_note("IPDNRFOCUSDT", -120, "ผู้ป่วยนัดมาทำผ่าตัด")])
        assert summary.surgical_context is True

    def test_under_anesthesia_sets_surgical_context(self) -> None:
        for text in ("under GA", "operation under spinal", "under SAB"):
            summary = scan_periop([_note("IPDADMPROGRESS", -30, text)])
            assert summary.surgical_context is True, text

    def test_benign_note_is_not_surgical_context(self) -> None:
        summary = scan_periop(
            [_note("IPDADMPROGRESS", -30, "patient comfortable, tolerating diet")]
        )
        assert summary.surgical_context is False


# =============================================================================
# AC: EBL is reported in millilitres, whatever unit the chart used. A litre or
# "cc" notation must not understate the magnitude the auditor weighs.
# =============================================================================
class TestBloodLoss:
    def test_ebl_ml(self) -> None:
        summary = scan_periop([_note("IPDNRFOCUSDT", 60, "EBL 1500 ml")])
        assert summary.blood_loss_ml == 1500

    def test_blood_loss_phrase_with_comma(self) -> None:
        summary = scan_periop([_note("IPDNRFOCUSDT", 60, "blood loss 1,500 mL")])
        assert summary.blood_loss_ml == 1500

    def test_litre_is_converted_to_ml(self) -> None:
        # "EBL 1.5 L" is 1500 mL; reporting "1" or "1.5" would understate a
        # major haemorrhage by three orders of magnitude.
        summary = scan_periop([_note("IPDNRFOCUSDT", 60, "EBL 1.5 L")])
        assert summary.blood_loss_ml == 1500

    def test_cc_is_treated_as_ml(self) -> None:
        summary = scan_periop([_note("IPDNRFOCUSDT", 60, "EBL 800 cc")])
        assert summary.blood_loss_ml == 800

    def test_max_volume_across_notes_is_kept(self) -> None:
        # Several notes mention EBL at different times; the auditor needs the
        # worst (largest) loss, not the latest charted value.
        notes = [
            _note("IPDNRFOCUSDT", 30, "EBL 200 ml in PACU"),
            _note("IPDNRFOCUSDT", -30, "intraop EBL 1500 ml"),
            _note("IPDADMPROGRESS", 60, "EBL 300 ml"),
        ]
        summary = scan_periop(notes)
        assert summary.blood_loss_ml == 1500

    def test_no_ebl_is_none(self) -> None:
        summary = scan_periop([_note("IPDNRFOCUSDT", 60, "post-op stable, no drains")])
        assert summary.blood_loss_ml is None

    def test_zero_ebl_is_not_a_signal(self) -> None:
        # "EBL 0 ml" means no significant loss; treating it as a peri-op blood
        # signal would mislead the auditor into seeing haemorrhage where the
        # chart explicitly recorded none.
        summary = scan_periop([_note("IPDNRFOCUSDT", 60, "EBL 0 ml")])
        assert summary.blood_loss_ml is None


# =============================================================================
# AC: Intra-op transfusion requires a SPECIFIC blood component co-located with
# an intra-op marker. Generic "blood" near "intra-op" (e.g. "intra-op blood
# loss") must NOT be read as a transfusion — precision over recall.
# =============================================================================
class TestIntraopTransfusion:
    def test_intraop_lprc_is_a_transfusion(self) -> None:
        summary = scan_periop([_note("IPDNRFOCUSDT", -30, "intraop LPRC 1 unit")])
        assert summary.intraop_transfusion is True

    def test_component_then_marker_is_a_transfusion(self) -> None:
        summary = scan_periop([_note("IPDADMPROGRESS", -30, "PRC 2 u given intra-op")])
        assert summary.intraop_transfusion is True

    def test_intraop_blood_loss_is_not_a_transfusion(self) -> None:
        # The marker is present but "blood loss" is not a transfused component.
        summary = scan_periop([_note("IPDNRFOCUSDT", -30, "intra-op blood loss 1500 ml")])
        assert summary.intraop_transfusion is False

    def test_component_without_intraop_marker_is_not_flagged(self) -> None:
        # A ward transfusion order (no intra-op marker) is the case being
        # audited, not an intra-op event; flagging it would beg the question.
        summary = scan_periop([_note("IPDADMPROGRESS", 120, "transfuse LPRC 1 unit on ward")])
        assert summary.intraop_transfusion is False


# =============================================================================
# AC: Findings carry provenance and are bounded to one per category, so the
# pinned bundle item stays compact and deterministic.
# =============================================================================
class TestFindings:
    def test_surgery_finding_has_category_and_provenance(self) -> None:
        note = _note("IPDNRFOCUSDT", -60, "s/p ORIF Lt femur")
        summary = scan_periop([note])
        surgery = [f for f in summary.findings if f.category == "surgery"]
        assert len(surgery) == 1
        assert surgery[0].at == note.timestamp
        assert surgery[0].source == "IPDNRFOCUSDT"
        assert "ORIF" in surgery[0].snippet

    def test_one_finding_per_category(self) -> None:
        # Two surgery cues in one note must not double-count.
        summary = scan_periop(
            [_note("IPDNRFOCUSDT", -60, "post-op s/p ORIF, under GA")]
        )
        assert len([f for f in summary.findings if f.category == "surgery"]) == 1

    def test_all_three_signals_yield_three_findings(self) -> None:
        text = "post-op ORIF, intraop LPRC 1 u, EBL 1500 ml"
        summary = scan_periop([_note("IPDNRFOCUSDT", -30, text)])
        cats = {f.category for f in summary.findings}
        assert cats == {"surgery", "blood_loss", "intraop_transfusion"}

    def test_blood_loss_finding_points_at_the_max_note(self) -> None:
        big = _note("IPDNRFOCUSDT", -30, "intraop EBL 1500 ml")
        notes = [_note("IPDNRFOCUSDT", 30, "EBL 200 ml"), big]
        summary = scan_periop(notes)
        loss = next(f for f in summary.findings if f.category == "blood_loss")
        assert loss.at == big.timestamp


# =============================================================================
# AC (Case 107 worked example): a single post-op nursing note with ORIF + a
# large EBL is exactly what the LLM ignored. The scan must surface all of it.
# =============================================================================
class TestCase107Shape:
    def test_orif_with_large_ebl(self) -> None:
        text = "Post-op day 0 s/p ORIF Lt femur, EBL 1500 ml, stable"
        summary = scan_periop([_note("IPDNRFOCUSDT", 180, text)])
        assert summary.surgical_context is True
        assert summary.blood_loss_ml == 1500
        assert summary.is_empty is False


# =============================================================================
# AC: Absence is null/empty, and the scan is total over its input.
# =============================================================================
class TestEmptyAndNullable:
    def test_empty_note_list_is_empty_summary(self) -> None:
        summary = scan_periop([])
        assert isinstance(summary, PeriopSummary)
        assert summary.is_empty is True
        assert summary.surgical_context is False
        assert summary.blood_loss_ml is None
        assert summary.intraop_transfusion is False
        assert summary.findings == ()

    def test_benign_notes_are_empty(self) -> None:
        summary = scan_periop(
            [_note("IPDADMPROGRESS", -30, "patient comfortable, ambulating")]
        )
        assert summary.is_empty is True

    def test_is_empty_false_when_anything_present(self) -> None:
        only_surgery = scan_periop([_note("IPDNRFOCUSDT", -30, "s/p ORIF")])
        only_loss = scan_periop([_note("IPDNRFOCUSDT", 30, "EBL 800 ml")])
        only_tx = scan_periop([_note("IPDNRFOCUSDT", -30, "intraop FFP 2 u")])
        assert only_surgery.is_empty is False
        assert only_loss.is_empty is False
        assert only_tx.is_empty is False


# =============================================================================
# BINDING GUARDRAIL: the summary carries facts only. No appropriateness /
# verdict field may be added — peri-op context is a supporting factor, never a
# standalone transfusion verdict (mirrors the HemodynamicSummary contract).
# =============================================================================
class TestFactOnlyContract:
    def test_summary_fields_are_exactly_the_fact_set(self) -> None:
        assert set(PeriopSummary.model_fields) == {
            "surgical_context",
            "blood_loss_ml",
            "intraop_transfusion",
            "findings",
        }

    def test_finding_fields_are_exactly_the_fact_set(self) -> None:
        assert set(PeriopFinding.model_fields) == {
            "category",
            "snippet",
            "at",
            "source",
        }


# =============================================================================
# AC: PeriopSummary / PeriopFinding are immutable (frozen) for audit-chain
# reproducibility, mirroring HemodynamicSummary.
# =============================================================================
class TestImmutability:
    def test_summary_is_frozen(self) -> None:
        summary = scan_periop([_note("IPDNRFOCUSDT", -30, "s/p ORIF")])
        with pytest.raises((TypeError, AttributeError, ValueError)):
            summary.surgical_context = False  # type: ignore[misc]

    def test_finding_is_frozen(self) -> None:
        finding = PeriopFinding(
            category="surgery",
            snippet="s/p ORIF",
            at=ANCHOR,
            source="IPDNRFOCUSDT",
        )
        with pytest.raises((TypeError, AttributeError, ValueError)):
            finding.category = "blood_loss"  # type: ignore[misc]
