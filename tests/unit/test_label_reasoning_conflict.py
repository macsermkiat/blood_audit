"""An answer whose label contradicts its own reasoning goes to a human.

Full hematology rerun 2026-09-22: 6 of 364 answers ended their reasoning on one
class ("Therefore this qualifies as APPROPRIATE") and put another in
``classification`` (NEEDS_REVIEW). Five of the six wrote the label last, as the
schema asks, so field order alone does not close this; and the pilot's audit
script, the only thing that caught it, does not run in production. The catch
now lives in the library: the prompt asks for one fixed closing sentence, code
reads it, and a mismatch floors the row to review with its own reason.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bba.audit_orders import AuditOrder
from bba.audit_pipeline import PipelineRowContext, apply_batch_results
from bba.audit_pipeline.replay import LABEL_REASONING_CONFLICT_REVIEW_REASON
from bba.audit_store import AuditStore
from bba.audit_store.models import AuditStoreConfig
from bba.llm_client import RawBatchResponse
from bba.llm_client.conclusion import closing_classification, concluded_class
from bba.llm_client.models import SONNET_MODEL_ID, BatchSubmissionResult
from bba.platelet_lookup.models import PlateletLookupResult
from bba.prompt_builder.system_prompt import platelet_system_prompt, system_prompt_for

_RUN_TS = datetime(2026, 9, 22, 4, 0, tzinfo=UTC)
_AUDIT_ID = "audit-conflict"


class TestClosingSentence:
    def test_reads_the_fixed_closing_sentence(self) -> None:
        assert (
            closing_classification(
                "... count 37,000. Final classification: APPROPRIATE"
            )
            == "APPROPRIATE"
        )

    def test_tolerates_trailing_punctuation_and_case(self) -> None:
        assert (
            closing_classification("Final classification: needs_review.")
            == "NEEDS_REVIEW"
        )

    def test_absent_sentence_is_none(self) -> None:
        assert (
            closing_classification("Therefore this qualifies as APPROPRIATE.") is None
        )

    def test_sentence_in_the_middle_does_not_count(self) -> None:
        # Only the closing line is binding; an earlier draft that changed is not.
        text = "Final classification: INAPPROPRIATE. On reflection... Final classification: APPROPRIATE"
        assert closing_classification(text) == "APPROPRIATE"

    def test_unknown_class_is_none(self) -> None:
        assert closing_classification("Final classification: MAYBE") is None

    @pytest.mark.parametrize(
        "text",
        [
            "**Final Classification: appropriate.**",
            "**Final Classification**: appropriate.",
            "Final classification: **APPROPRIATE**",
            "Final classification - APPROPRIATE",
            "Final classification: APPROPRIATE (indication 6).",
        ],
    )
    def test_markdown_and_layout_variants_are_read(self, text: str) -> None:
        # Codex on the PR: a bold or lowercase closing line returned None, so
        # a NEEDS_REVIEW label with APPROPRIATE reasoning fell through to the
        # native-review guardrail, which asserts INAPPROPRIATE.
        assert closing_classification(text) == "APPROPRIATE"


class TestConcludedClassFallback:
    # The regex reader from the pilot audit, now in the library so old stored
    # answers (no closing sentence) get the same check.
    def test_reads_the_stated_conclusion(self) -> None:
        assert (
            concluded_class("Therefore this should be classified as INAPPROPRIATE.")
            == "INAPPROPRIATE"
        )

    def test_rejected_alternative_is_not_the_conclusion(self) -> None:
        assert (
            concluded_class("this is INAPPROPRIATE rather than INSUFFICIENT_EVIDENCE.")
            == "INAPPROPRIATE"
        )

    def test_correction_after_a_rejection_wins(self) -> None:
        assert concluded_class("not APPROPRIATE but INAPPROPRIATE.") == "INAPPROPRIATE"

    def test_no_class_named_is_none(self) -> None:
        assert concluded_class("The count was 8,000 /uL on chemotherapy.") is None

    def test_postposed_negation_does_not_become_the_conclusion(self) -> None:
        # Codex on the PR: "an APPROPRIATE classification is not supported"
        # names the class before the negation.
        text = "This is INAPPROPRIATE; an APPROPRIATE classification is not supported."
        assert concluded_class(text) == "INAPPROPRIATE"

    def test_negating_the_alternative_keeps_the_affirmative(self) -> None:
        # Codex round 2: "and not INAPPROPRIATE" negates the other class, not
        # this one; returning None here bypassed the guardrail.
        assert (
            concluded_class("This is APPROPRIATE and not INAPPROPRIATE.")
            == "APPROPRIATE"
        )
        assert concluded_class("APPROPRIATE, not INAPPROPRIATE.") == "APPROPRIATE"
        # Codex round 3: any clause boundary or the next class mention ends the
        # window in which a negation can belong to this class.
        assert (
            concluded_class(
                "This is APPROPRIATE because INAPPROPRIATE is not supported."
            )
            == "APPROPRIATE"
        )
        assert (
            concluded_class("This is APPROPRIATE, while INAPPROPRIATE is not.")
            == "APPROPRIATE"
        )

    def test_class_used_as_an_adjective_of_a_rejected_noun(self) -> None:
        text = "No APPROPRIATE indication exists, so the order is INAPPROPRIATE."
        assert concluded_class(text) == "INAPPROPRIATE"


class TestPromptAsksForTheSentence:
    @pytest.mark.parametrize(
        "mode", ["HB_7_10_REVIEW", "HB_GT_10_OVERRIDE", "RESERVE_AHEAD_REVIEW"]
    )
    def test_rbc_templates(self, mode: str) -> None:
        assert "Final classification:" in system_prompt_for(
            task_mode=mode, cohort_threshold=7.0
        )

    def test_platelet_prompt(self) -> None:
        assert "Final classification:" in platelet_system_prompt()


def _ctx() -> PipelineRowContext:
    return PipelineRowContext.for_platelet(
        order=AuditOrder(
            audit_id=_AUDIT_ID,
            hn="HN-1",
            an="AN-1",
            reqno="REQ-1",
            order_datetime=_RUN_TS,
            anchor_imputed=False,
            products_ordered=("PLT-POOL",),
            diagnosis_codes=("C92.0",),
            component="platelet",
        ),
        platelet_result=PlateletLookupResult(
            value_k_ul=37.0,
            datetime_utc=_RUN_TS,
            source="HEMATOLOGY",
            freshness="fresh",
        ),
        hn_hash="hn",
        an_hash="an",
        redactor_version="0.4.1+test",
        redactor_model_sha="sha",
        policy_version="kcmh-pr17.2-2024",
        prompt_hash="ph",
        evidence_bundle_hash="bh",
    )


def _response(
    label: str, reasoning: str, *, intracranial: bool = True
) -> RawBatchResponse:
    payload = {
        "indications": [
            {
                "code": "6",
                "quote": "acute on chronic SDH",
                "source_id": "E16",
                "confidence": 0.9,
            }
        ],
        "negative_evidence": [],
        "reasoning_summary_en": reasoning,
        "reasoning_summary_th": "สรุปภาษาไทย",
        "active_bleeding": False,
        "procedure_indication": False,
        "prophylactic_marrow_failure": False,
        "intracranial_bleed_indication": intracranial,
        "classification": label,
    }
    return RawBatchResponse(
        batch_id="msgbatch_conflict",
        results=(
            BatchSubmissionResult(
                custom_id=_AUDIT_ID,
                model_id=SONNET_MODEL_ID,  # type: ignore[arg-type]
                raw_response_json={
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "classify_transfusion_order",
                            "input": payload,
                        }
                    ]
                },
                request_json={"messages": [{"role": "user", "content": "..."}]},
                response_headers={"anthropic-version": "2023-06-01"},
                request_timestamp=_RUN_TS,
                latency_ms=500,
                anthropic_version="2023-06-01",
                prompt_cache_id=None,
                extended_thinking_blocks=None,
            ),
        ),
    )


def _apply(tmp_path, response: RawBatchResponse):
    store = AuditStore(
        AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
    )
    apply_batch_results(
        response, audit_store=store, run_id="run-conflict", contexts={_AUDIT_ID: _ctx()}
    )
    return store.read_audit_results()[0]


class TestReplayGuardrail:
    def test_label_that_contradicts_the_closing_sentence_goes_to_review(
        self, tmp_path
    ) -> None:
        # 68000711 on the 2026-09-22 rerun: reasoning applies the 7-day rule
        # and closes on APPROPRIATE; the label said NEEDS_REVIEW. Neither is
        # trusted: a human decides, with the reason stated.
        row = _apply(
            tmp_path,
            _response(
                "NEEDS_REVIEW",
                "SDH 4 days old, count 37,000. Final classification: APPROPRIATE",
            ),
        )
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LABEL_REASONING_CONFLICT_REVIEW_REASON
        assert row.needs_human_review is True

    def test_a_contradicting_clear_is_floored_not_cleared(self, tmp_path) -> None:
        row = _apply(
            tmp_path,
            _response(
                "APPROPRIATE",
                "No indication is met. Final classification: INAPPROPRIATE",
            ),
        )
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LABEL_REASONING_CONFLICT_REVIEW_REASON

    def test_consistent_answer_is_untouched(self, tmp_path) -> None:
        row = _apply(
            tmp_path,
            _response(
                "APPROPRIATE", "Indication 6 met. Final classification: APPROPRIATE"
            ),
        )
        assert row.final_classification == "APPROPRIATE"
        assert row.review_reason is None

    def test_old_answer_without_the_sentence_uses_the_fallback_reader(
        self, tmp_path
    ) -> None:
        # Stored answers predate the closing sentence; the regex reader still
        # catches an explicit contradiction there.
        row = _apply(
            tmp_path,
            _response("NEEDS_REVIEW", "Therefore this qualifies as APPROPRIATE."),
        )
        assert row.review_reason == LABEL_REASONING_CONFLICT_REVIEW_REASON

    def test_reasoning_that_names_no_class_is_left_alone(self, tmp_path) -> None:
        row = _apply(
            tmp_path,
            _response("APPROPRIATE", "Acute SDH at 37,000 meets indication 6."),
        )
        assert row.final_classification == "APPROPRIATE"
