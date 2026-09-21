"""Response audit between the LLM leg and build_review.py (issue #239).

The first real-data platelet run produced answers that contradicted themselves
(label APPROPRIATE, reasoning "should be classified as INAPPROPRIATE"). Nothing
between the LLM leg and the review page looked at the responses, so a clinician
found it on case 1. Code decides what code can; a consortium of judge models
reads the conclusion out of the reasoning, and code compares it with the label.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"

_ORDERED = (
    "indications",
    "negative_evidence",
    "reasoning_summary_en",
    "reasoning_summary_th",
    "active_bleeding",
    "procedure_indication",
    "prophylactic_marrow_failure",
    "intracranial_bleed_indication",
    "classification",
)
_NO_SIGNALS = {
    "active_bleeding": False,
    "procedure_indication": False,
    "prophylactic_marrow_failure": False,
    "intracranial_bleed_indication": False,
}
_CITED = ({"code": "5", "quote": "plt 8", "source_id": "E3", "confidence": 0.9},)


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "pilot_audit_llm_responses", PILOT_DIR / "audit_llm_responses.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _record(audit: ModuleType, **overrides: object) -> object:
    base: dict[str, object] = {
        "reqno": "R1",
        "component": "platelet",
        "label": "INAPPROPRIATE",
        "final": "INAPPROPRIATE",
        "review_reason": None,
        "field_order": _ORDERED,
        "signals": dict(_NO_SIGNALS),
        "reasoning_en": (
            "Platelet count 22,000 /uL, no bleeding, no procedure. "
            "This order is INAPPROPRIATE."
        ),
        "reasoning_th": "เกล็ดเลือด 22,000 ไม่มีข้อบ่งชี้ จึงไม่เหมาะสม",
        "indications": (),
        "trigger_value": 22.0,
    }
    return audit.ResponseRecord(**{**base, **overrides})


def _findings(audit: ModuleType, record: object) -> dict[str, object]:
    return {f.code: f for f in audit.check_record(record)}


class TestConcludedClass:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Therefore this should be classified as INAPPROPRIATE.", "INAPPROPRIATE"),
            # The rejected alternative comes last in the sentence; it is not the
            # conclusion. This phrasing made a naive "last class named" scan
            # report 58 false contradictions on the real run.
            (
                "this is INAPPROPRIATE rather than INSUFFICIENT_EVIDENCE.",
                "INAPPROPRIATE",
            ),
            (
                "warranting NEEDS_REVIEW rather than a clean APPROPRIATE or "
                "INAPPROPRIATE determination.",
                "NEEDS_REVIEW",
            ),
            (
                "result in an INAPPROPRIATE classification, not "
                "INSUFFICIENT_EVIDENCE, since the notes are not silent.",
                "INAPPROPRIATE",
            ),
            # A correction AFTER the rejected class is the conclusion (Codex
            # review): the rejection must not swallow the rest of the sentence.
            (
                "Initially APPROPRIATE. On review, this is not APPROPRIATE but "
                "INAPPROPRIATE.",
                "INAPPROPRIATE",
            ),
            ("this is not INAPPROPRIATE but NEEDS_REVIEW.", "NEEDS_REVIEW"),
            ("The count was 8,000 /uL on chemotherapy.", None),
        ],
    )
    def test_reads_the_conclusion_not_the_rejected_alternative(
        self, audit: ModuleType, text: str, expected: str | None
    ) -> None:
        assert audit.concluded_class(text) == expected


class TestChecks:
    def test_consistent_answer_is_clean(self, audit: ModuleType) -> None:
        assert _findings(audit, _record(audit)) == {}

    def test_label_that_contradicts_its_own_reasoning_is_high(
        self, audit: ModuleType
    ) -> None:
        # REQNO 68000037: label APPROPRIATE, reasoning concludes INAPPROPRIATE.
        record = _record(
            audit,
            label="APPROPRIATE",
            final="NEEDS_REVIEW",
            review_reason="platelet_llm_overclear_suspect",
            indications=_CITED,
        )
        assert (
            _findings(audit, record)["label_contradicts_reasoning"].severity == "HIGH"
        )

    def test_label_written_before_the_reasoning_is_flagged_on_platelets(
        self, audit: ModuleType
    ) -> None:
        # All 13 contradictions the consortium found on the real run were
        # answers that wrote the label before the reasoning.
        early = ("classification", *_ORDERED[:-1])
        assert "label_before_reasoning" in _findings(
            audit, _record(audit, field_order=early)
        )

    def test_rbc_label_first_is_by_design(self, audit: ModuleType) -> None:
        record = _record(
            audit,
            component="red_cell",
            field_order=("classification", "indications", "reasoning_summary_en"),
            signals={},
            reasoning_en="Hb 9.2 g/dL, stable. This order is INAPPROPRIATE.",
            trigger_value=9.2,
        )
        assert "label_before_reasoning" not in _findings(audit, record)

    def test_withholding_label_with_a_true_hard_signal_is_high(
        self, audit: ModuleType
    ) -> None:
        # A true signal says an indication is met; INAPPROPRIATE says none is.
        record = _record(audit, signals={**_NO_SIGNALS, "active_bleeding": True})
        assert _findings(audit, record)["signal_contradicts_label"].severity == "HIGH"

    def test_a_string_false_is_not_a_true_signal(self, audit: ModuleType) -> None:
        record = _record(audit, signals={**_NO_SIGNALS, "active_bleeding": "false"})
        assert "signal_contradicts_label" not in _findings(audit, record)

    def test_final_verdict_that_differs_from_the_label_needs_a_reason(
        self, audit: ModuleType
    ) -> None:
        # Every guardrail that overrides the model stamps a review_reason. A
        # silent change means a verdict nobody can trace.
        record = _record(audit, final="APPROPRIATE", review_reason=None)
        assert (
            _findings(audit, record)["final_differs_without_reason"].severity == "HIGH"
        )

    def test_guardrail_override_with_its_reason_is_not_a_finding(
        self, audit: ModuleType
    ) -> None:
        record = _record(
            audit,
            label="APPROPRIATE",
            final="NEEDS_REVIEW",
            review_reason="platelet_trend_unsupported",
            signals={**_NO_SIGNALS, "prophylactic_marrow_failure": True},
            reasoning_en="Count 22,000 /uL, expected to fall. This is APPROPRIATE.",
            indications=_CITED,
        )
        assert "final_differs_without_reason" not in _findings(audit, record)

    def test_clear_without_a_cited_indication_is_flagged(
        self, audit: ModuleType
    ) -> None:
        record = _record(
            audit,
            label="APPROPRIATE",
            final="APPROPRIATE",
            signals={**_NO_SIGNALS, "active_bleeding": True},
            reasoning_en="Count 22,000 /uL with melena. This is APPROPRIATE.",
        )
        assert "appropriate_without_citation" in _findings(audit, record)

    def test_clear_resting_only_on_low_confidence_citations_is_flagged(
        self, audit: ModuleType
    ) -> None:
        record = _record(
            audit,
            label="APPROPRIATE",
            final="APPROPRIATE",
            signals={**_NO_SIGNALS, "active_bleeding": True},
            reasoning_en="Count 22,000 /uL with melena. This is APPROPRIATE.",
            indications=({**_CITED[0], "confidence": 0.3},),
        )
        assert "low_confidence_appropriate" in _findings(audit, record)

    def test_citation_without_a_quote_or_source_is_flagged(
        self, audit: ModuleType
    ) -> None:
        record = _record(audit, indications=({"code": "5", "confidence": 0.9},))
        assert "malformed_indication" in _findings(audit, record)

    def test_reasoning_that_never_states_the_trigger_count_is_flagged(
        self, audit: ModuleType
    ) -> None:
        # Smoke run 2026-09-20: counts were rendered blank and the model judged
        # platelet orders without the trigger value.
        record = _record(
            audit, reasoning_en="No bleeding documented. This is INAPPROPRIATE."
        )
        assert "trigger_value_not_in_reasoning" in _findings(audit, record)

    @pytest.mark.parametrize(
        ("component", "value", "text"),
        [
            ("platelet", 22.0, "Count 122,000 /uL. This is INAPPROPRIATE."),
            ("red_cell", 9.2, "Hb 19.2 g/dL. This is INAPPROPRIATE."),
        ],
    )
    def test_a_longer_number_does_not_count_as_the_trigger_value(
        self, audit: ModuleType, component: str, value: float, text: str
    ) -> None:
        record = _record(
            audit, component=component, trigger_value=value, reasoning_en=text
        )
        assert "trigger_value_not_in_reasoning" in _findings(audit, record)

    def test_a_mostly_thai_english_summary_is_flagged(self, audit: ModuleType) -> None:
        record = _record(
            audit,
            reasoning_en="เกล็ดเลือด 22,000 /uL ไม่มีเลือดออก ไม่มีหัตถการ INAPPROPRIATE",
        )
        assert "language_or_tag_leak" in _findings(audit, record)

    def test_an_english_thai_summary_is_flagged(self, audit: ModuleType) -> None:
        record = _record(audit, reasoning_th="Platelet count 22,000, no indication.")
        assert "language_or_tag_leak" in _findings(audit, record)

    def test_a_quoted_thai_note_in_the_english_summary_is_not_a_leak(
        self, audit: ModuleType
    ) -> None:
        # 208 of 364 real summaries quote a Thai note verbatim; that is a
        # citation, not a language mix-up.
        record = _record(
            audit,
            reasoning_en=(
                "Platelet count 22,000 /uL; the note says 'Rheumato ดู >plan "
                "Arthrocentesis' but no procedure is scheduled in the window, and "
                "no bleeding is documented. This order is INAPPROPRIATE."
            ),
        )
        assert "language_or_tag_leak" not in _findings(audit, record)

    def test_tool_tags_inside_a_summary_are_flagged(self, audit: ModuleType) -> None:
        record = _record(
            audit,
            reasoning_en=(
                "Count 22,000 /uL. This is INAPPROPRIATE.</reasoning_summary_en>"
            ),
        )
        assert "language_or_tag_leak" in _findings(audit, record)

    def test_parse_failure_is_high_and_rerunnable(self, audit: ModuleType) -> None:
        record = _record(audit, final="NEEDS_REVIEW", review_reason="schema_mismatch")
        finding = _findings(audit, record)["parse_failure"]

        assert finding.severity == "HIGH"
        assert finding.code in audit.RERUN_CODES


class TestConsortium:
    def test_majority_disagreeing_with_the_label_is_a_contradiction(
        self, audit: ModuleType
    ) -> None:
        verdict = audit.consortium_verdict(
            "APPROPRIATE", ["INAPPROPRIATE", "INAPPROPRIATE", "NEEDS_REVIEW"]
        )
        assert verdict == "contradiction"

    def test_only_unanimous_judges_certify_the_label(self, audit: ModuleType) -> None:
        unanimous = ["INAPPROPRIATE", "INAPPROPRIATE", "INAPPROPRIATE"]
        assert audit.consortium_verdict("INAPPROPRIATE", unanimous) == "consistent"

    def test_one_dissenting_judge_is_surfaced_not_outvoted(
        self, audit: ModuleType
    ) -> None:
        # Codex review: a 2-1 majority for the label hid the dissent. The
        # consortium exists to surface disagreement, so dissent is a split.
        votes = ["APPROPRIATE", "APPROPRIATE", "INAPPROPRIATE"]
        assert audit.consortium_verdict("APPROPRIATE", votes) == "split"

    def test_a_single_stated_vote_certifies_nothing(self, audit: ModuleType) -> None:
        votes = ["APPROPRIATE", "NONE_STATED", "NONE_STATED"]
        assert audit.consortium_verdict("APPROPRIATE", votes) == "no_conclusion_stated"

    def test_judge_outage_is_reported_not_passed(self, audit: ModuleType) -> None:
        votes = ["JUDGE_ERROR", "JUDGE_ERROR", "INAPPROPRIATE"]
        assert audit.consortium_verdict("INAPPROPRIATE", votes) == "unavailable"


class TestLoadFindings:
    def test_report_entry_with_no_stored_response_is_reported(
        self, audit: ModuleType
    ) -> None:
        # A record that silently drops out of the audit is a record nobody
        # checked.
        finding = audit.unmatched_report_finding("R9", "no audit-store row")

        assert finding.severity == "HIGH"
        assert finding.code == "response_not_auditable"

    def test_tool_payload_is_read_from_the_stores_read_only_mappings(
        self, audit: ModuleType
    ) -> None:
        # The audit store hands back mappingproxy objects, not dicts. A dict-only
        # type check made all 360 real responses "not auditable".
        response = MappingProxyType(
            {
                "content": (
                    MappingProxyType(
                        {
                            "type": "tool_use",
                            "input": MappingProxyType(
                                {"classification": "APPROPRIATE"}
                            ),
                        }
                    ),
                )
            }
        )
        assert audit._tool_input(response) == {"classification": "APPROPRIATE"}

    def test_a_missing_response_payload_reads_as_empty(self, audit: ModuleType) -> None:
        assert audit._tool_input(None) == {}

    def test_report_entry_whose_llm_result_is_missing_is_high_and_rerunnable(
        self, audit: ModuleType
    ) -> None:
        # Codex P1 on #241: run_llm_leg writes llm_final: null when a batch row
        # is dropped or unparsable. Filtering those out let the audit exit 0
        # while the page showed "LLM verdict missing" for the case.
        entries = [
            {"audit_id": "a1", "reqno": "R1", "llm_final": None},
            {
                "audit_id": "a2",
                "reqno": "R2",
                "llm_final": {"final_classification": "X"},
            },
        ]
        findings = audit.missing_result_findings(entries)

        assert [(f.reqno, f.code, f.severity) for f in findings] == [
            ("R1", "llm_result_missing", "HIGH")
        ]
        assert "llm_result_missing" in audit.RERUN_CODES


class TestJudgeFindings:
    _EN_OK = ("INAPPROPRIATE", "INAPPROPRIATE", "INAPPROPRIATE")

    def test_thai_summary_is_held_to_the_label_on_its_own(
        self, audit: ModuleType
    ) -> None:
        # Codex P1 on #241: with no English majority the cross-language check
        # is silent, and the Thai text is what a Thai reviewer reads.
        record = _record(audit)
        en = ("NONE_STATED", "NONE_STATED", "INAPPROPRIATE")
        th = ("APPROPRIATE", "APPROPRIATE", "APPROPRIATE")

        codes = {f.code: f for f in audit.judge_findings(record, en, th)}

        assert codes["consortium_thai_label_contradiction"].severity == "HIGH"

    def test_summaries_that_disagree_with_each_other_are_high(
        self, audit: ModuleType
    ) -> None:
        record = _record(audit)
        th = ("NEEDS_REVIEW", "NEEDS_REVIEW", "NEEDS_REVIEW")

        codes = {f.code for f in audit.judge_findings(record, self._EN_OK, th)}

        assert "en_th_conclusion_mismatch" in codes

    def test_agreeing_summaries_and_label_are_clean(self, audit: ModuleType) -> None:
        record = _record(audit)
        assert audit.judge_findings(record, self._EN_OK, self._EN_OK) == ()

    def test_english_contradiction_is_rerunnable(self, audit: ModuleType) -> None:
        record = _record(audit, label="APPROPRIATE")
        findings = audit.judge_findings(record, self._EN_OK, self._EN_OK)

        assert {f.code for f in findings} >= {"consortium_label_contradiction"}
        assert all(f.code in audit.RERUN_CODES for f in findings)


def test_the_audit_file_records_how_thoroughly_it_judged(audit: ModuleType) -> None:
    # Codex P1 on #241: build_review must be able to tell a full consortium
    # audit from a regex-only one, which missed 7 of 13 real contradictions.
    payload = audit.audit_payload("candidates", "abc123", 5, ())

    assert payload == {
        "judge": "candidates",
        "report_sha256": "abc123",
        "records": 5,
        "findings": [],
    }


def test_full_judging_is_the_default(audit: ModuleType) -> None:
    assert audit.build_parser().parse_args([]).judge == "all"


class TestRegexCandidateAfterJudging:
    def _candidate(self, audit: ModuleType) -> object:
        return audit.Finding("R1", "label_contradicts_reasoning", "HIGH", "x")

    def test_judges_that_settled_it_take_over(self, audit: ModuleType) -> None:
        settled = audit.settle_regex_candidates([self._candidate(audit)], {"R1"})
        assert settled[0].severity == "LOW"

    def test_an_unsettled_candidate_keeps_blocking(self, audit: ModuleType) -> None:
        # Local Codex review: judges answering INAPPROPRIATE, NONE_STATED,
        # NONE_STATED (or splitting) settle nothing, yet the candidate was
        # downgraded and the contradiction reached the page.
        kept = audit.settle_regex_candidates([self._candidate(audit)], set())
        assert kept[0].severity == "HIGH"

    def test_only_a_certified_or_contradicted_label_counts_as_settled(
        self, audit: ModuleType
    ) -> None:
        assert audit.english_settled("INAPPROPRIATE", ("INAPPROPRIATE",) * 3)
        assert audit.english_settled("APPROPRIATE", ("INAPPROPRIATE",) * 3)
        assert not audit.english_settled(
            "APPROPRIATE", ("INAPPROPRIATE", "NONE_STATED", "NONE_STATED")
        )
        assert not audit.english_settled(
            "APPROPRIATE", ("APPROPRIATE", "APPROPRIATE", "INAPPROPRIATE")
        )


class TestRowIdentity:
    _FINAL = {
        "final_classification": "INAPPROPRIATE",
        "review_reason": None,
        "reasoning_en": "Count 22,000 /uL, melena. INAPPROPRIATE.",
        "reasoning_th": "ไม่เหมาะสม",
    }

    def _row(self, **overrides: object) -> SimpleNamespace:
        base = {
            "final_classification": "INAPPROPRIATE",
            "review_reason": None,
            "reasoning_summary_en": "Count 22,000 /uL, melena. INAPPROPRIATE.",
            "reasoning_summary_thai": "ไม่เหมาะสม",
        }
        return SimpleNamespace(**{**base, **overrides})

    def test_the_row_that_produced_the_report_entry_matches(
        self, audit: ModuleType
    ) -> None:
        assert audit.row_produced_entry(self._row(), self._FINAL)

    def test_a_later_run_with_the_same_verdict_is_a_different_response(
        self, audit: ModuleType
    ) -> None:
        # Local Codex review of #241: an interrupted re-run persists a newer row
        # with the same verdict and reason but another payload. Pairing it with
        # the report's summaries would audit a response nobody will read.
        rerun = self._row(reasoning_summary_en="Count 22,000 /uL. INAPPROPRIATE.")
        assert not audit.row_produced_entry(rerun, self._FINAL)
