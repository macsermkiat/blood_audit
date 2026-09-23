"""Aplastic anaemia on active therapy is an indication, not an exclusion.

Case 19 of the platelet review (68011945): very severe aplastic anaemia on ATG
and cyclosporine, count 2,000 /uL, projected 0, judged INAPPROPRIATE because
exclusion D ("aplastic anemia without active bleeding") barred every
prophylaxis rule. That line copies NICE NG24's generic "chronic bone marrow
failure" carve-out. The disease-specific BSH aplastic anaemia guideline
(Killick 2016; 2024 update) says the opposite for this subgroup: prophylaxis
below 10 x 10^9/L for patients on active therapy aimed at reversing the
thrombocytopenia, and below 20 x 10^9/L during the ATG course or with sepsis.
Clinician ruling 2026-09-23: follow the guideline. Ten orders in the hematology
cohort (counts 1,000 to 9,000, all on ATG or cyclosporine) turn on this.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import bba.feature_flags as feature_flags
import pytest

from bba.audit_orders import AuditOrder
from bba.audit_pipeline import PipelineRowContext, apply_batch_results
from bba.audit_store import AuditStore
from bba.audit_store.models import AuditStoreConfig
from bba.llm_client import RawBatchResponse
from bba.llm_client.models import SONNET_MODEL_ID, BatchSubmissionResult
from bba.llm_client.parser import parse_platelet_structured_response
from bba.llm_client.transport import _PLATELET_TOOL_INPUT_SCHEMA
from bba.platelet_guardrail import (
    platelet_overclear_suspect,
    platelet_trend_unsupported,
)
from bba.platelet_guardrail.models import PlateletHardSignals
from bba.platelet_lookup.models import PlateletLookupResult
from bba.prompt_builder.system_prompt import platelet_system_prompt


def _prompt() -> str:
    return platelet_system_prompt()


class TestIndicationEight:
    def test_active_therapy_is_a_numbered_indication_at_ten_thousand(self) -> None:
        prompt = _prompt()
        assert re.search(
            r"8\. Aplastic an(a)?emia on active therapy[^.]*<10,000",
            prompt,
            re.IGNORECASE,
        )

    def test_names_the_therapies_that_count_as_active(self) -> None:
        text = _prompt().split("8. Aplastic", 1)[1][:900].lower()
        for therapy in ("atg", "cyclosporine", "eltrombopag", "transplant"):
            assert therapy in text

    def test_twenty_thousand_during_atg_or_sepsis(self) -> None:
        text = _prompt().split("8. Aplastic", 1)[1][:900]
        assert re.search(r"ATG course[^.]*<20,000|<20,000[^.]*ATG course", text)
        assert "sepsis" in text.lower()


class TestExclusionDIsNarrowed:
    def test_exclusion_d_covers_only_untreated_or_stable_disease(self) -> None:
        line = re.search(r"D\. [^\n]*", _prompt())
        assert line is not None
        text = line.group(0).lower()
        assert "aplastic" in text
        assert re.search(r"chronic|stable|not (under|on|receiving) active", text)
        assert "indication 8" in text

    def test_override_sentence_routes_active_therapy_to_indication_eight(self) -> None:
        prompt = _prompt()
        override = prompt.split("EXCLUSION OVERRIDE", 1)[1][:1200]
        assert "NOT" in override and "chemotherapy" in override
        assert "indication 8" in override


class TestHardSignalCoversIt:
    def test_prompt_hard_signal_definition_names_indication_eight(self) -> None:
        # Without this the over-clear guardrail would floor every such clear:
        # a clear needs at least one true hard signal.
        text = _prompt().split("prophylactic_marrow_failure", 1)[1][:400]
        assert "indication 8" in text or "aplastic" in text.lower()

    def test_schema_description_matches(self) -> None:
        desc = _PLATELET_TOOL_INPUT_SCHEMA["properties"]["prophylactic_marrow_failure"][
            "description"
        ]
        assert "aplastic" in desc.lower()

    def test_guardrail_docstring_matches(self) -> None:
        assert (
            "aplastic" in (PlateletHardSignals.__doc__ or "").lower()
            or "aplastic"
            in (
                __import__("bba.platelet_guardrail.models", fromlist=["x"]).__doc__
                or ""
            ).lower()
        )


def test_terminal_line_counts_indication_eight() -> None:
    assert "no indication 4–8 below its threshold" in _prompt()


# --- Fifth hard signal (Codex on the PR) -------------------------------------
# The count-trend floor (#237) fires when prophylactic_marrow_failure is the
# ONLY true signal, the count is >= 10,000 and the projection is unsupported.
# An ATG patient correctly cleared at 15,000 under the new <20,000 rule would
# be floored to review. Indication 8 therefore carries its own signal,
# aplastic_active_therapy_indication, mirroring intracranial_bleed_indication.


_RUN_TS = datetime(2026, 9, 23, 4, 0, tzinfo=UTC)
_AUDIT_ID = "audit-atg"
_SIGNALS = (
    "active_bleeding",
    "procedure_indication",
    "prophylactic_marrow_failure",
    "intracranial_bleed_indication",
    "aplastic_active_therapy_indication",
)


class TestFifthSignal:
    def test_schema_requires_it_before_the_label(self) -> None:
        props = list(_PLATELET_TOOL_INPUT_SCHEMA["properties"])
        assert props.index("aplastic_active_therapy_indication") < props.index(
            "classification"
        )
        assert (
            "aplastic_active_therapy_indication"
            in _PLATELET_TOOL_INPUT_SCHEMA["required"]
        )
        assert (
            "after both reasoning summaries"
            in _PLATELET_TOOL_INPUT_SCHEMA["properties"][
                "aplastic_active_therapy_indication"
            ]["description"]
        )

    def test_prompt_lists_it(self) -> None:
        assert "aplastic_active_therapy_indication" in _prompt()

    def test_it_counts_as_a_grounded_signal_for_the_overclear_guardrail(self) -> None:
        signals = PlateletHardSignals(aplastic_active_therapy_indication=True)
        assert signals.any_signal()
        assert not platelet_overclear_suspect("APPROPRIATE", "NEEDS_REVIEW", signals)

    def test_it_exempts_the_count_trend_floor(self) -> None:
        # ATG patient at 15,000, flat projection: valid under indication 8.
        signals = PlateletHardSignals(
            prophylactic_marrow_failure=True, aplastic_active_therapy_indication=True
        )
        assert not platelet_trend_unsupported("APPROPRIATE", signals, 15.0, 15.0)
        assert not platelet_trend_unsupported("APPROPRIATE", signals, 15.0, None)

    def test_stored_four_signal_answers_still_parse(self) -> None:
        # Every response persisted before 2026-09-23 lacks the field.
        outcome = parse_platelet_structured_response(
            _response(classification="INAPPROPRIATE", extra_signal=None).results[0]
        )
        assert not outcome.parse_failure
        assert outcome.platelet_hard_signals is not None
        assert outcome.platelet_hard_signals.aplastic_active_therapy_indication is False


def _ctx(count: float, projected: float | None) -> PipelineRowContext:
    return PipelineRowContext.for_platelet(
        order=AuditOrder(
            audit_id=_AUDIT_ID,
            hn="HN-1",
            an="AN-1",
            reqno="REQ-1",
            order_datetime=_RUN_TS,
            anchor_imputed=False,
            products_ordered=("PLT-POOL",),
            diagnosis_codes=("D61.3",),
            component="platelet",
        ),
        platelet_result=PlateletLookupResult(
            value_k_ul=count,
            datetime_utc=_RUN_TS,
            source="HEMATOLOGY",
            freshness="fresh",
        ),
        platelet_projected_24h_k_ul=projected,
        platelet_projection_computed=True,
        hn_hash="hn",
        an_hash="an",
        redactor_version="0.4.1+test",
        redactor_model_sha="sha",
        policy_version="kcmh-pr17.2-2024",
        prompt_hash="ph",
        evidence_bundle_hash="bh",
    )


def _response(*, classification: str, extra_signal: bool | None) -> RawBatchResponse:
    payload = {
        "indications": [
            {
                "code": "8",
                "quote": "ATG day 3, plt 15,000",
                "source_id": "E4",
                "confidence": 0.9,
            }
        ],
        "negative_evidence": [],
        "reasoning_summary_en": "Very severe aplastic anaemia on ATG, count 15,000. Final classification: "
        + classification,
        "reasoning_summary_th": "สรุปภาษาไทย",
        "active_bleeding": False,
        "procedure_indication": False,
        "prophylactic_marrow_failure": True,
        "intracranial_bleed_indication": False,
    }
    if extra_signal is not None:
        payload["aplastic_active_therapy_indication"] = extra_signal
    payload["classification"] = classification
    return RawBatchResponse(
        batch_id="msgbatch_atg",
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


def test_replay_keeps_an_atg_clear_at_fifteen_thousand(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codex on the PR: with the trend floor on, this row was floored to
    # NEEDS_REVIEW / platelet_trend_unsupported.
    monkeypatch.setattr(feature_flags, "PLATELET_TREND_GUARDRAIL_ENABLED", True)
    store = AuditStore(
        AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
    )
    apply_batch_results(
        _response(classification="APPROPRIATE", extra_signal=True),
        audit_store=store,
        run_id="run-atg",
        contexts={_AUDIT_ID: _ctx(15.0, 15.0)},
    )
    row = store.read_audit_results()[0]
    assert row.final_classification == "APPROPRIATE"
    assert row.review_reason is None
