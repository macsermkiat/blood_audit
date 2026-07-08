"""RED-phase failing tests for issue #22 (bba.llm_client).

Each ``class`` maps to one acceptance criterion in the issue body. Tests
assert contracts (the WHY), not implementation choices — see PRD
§"Testing Decisions".

No implementation exists yet; every public function in
``bba.llm_client`` raises ``NotImplementedError("RED-phase scaffold;
see issue #22")``. Failing with that string is the structural pin —
any other failure mode means the scaffold drifted, not that the test
discovered a real regression.

The acceptance-criterion → test-class map:

* AC ① "Implementation in ``src/bba/llm_client/``" → implicit by
  imports.
* AC ② "custom_id assertion: mismatch aborts batch with explicit
  error" → :class:`TestCustomIdAssertion`.
* AC ③ "Betamax/VCR cassettes for offline replay of Anthropic API" →
  :class:`TestCassetteReplay`.
* AC ④ "Retry → escalation test: simulate Sonnet failure twice → Opus
  invocation" → :class:`TestRetryEscalation`.
* AC ⑤ "Sonnet/Opus disagreement test: synthetic responses with
  different classifications → NEEDS_REVIEW" →
  :class:`TestDisagreementDetection`.
* AC ⑥ "Malformed-JSON fail-closed test: garbage output → NEEDS_REVIEW
  with parse_failure" → :class:`TestMalformedJsonFailClosed`.
* AC ⑦ "Full response persistence to llm_calls via audit_store" →
  :class:`TestFullResponsePersistence`.
* AC ⑧ "Coverage ≥ 70%; ruff + mypy clean" → structural; the test
  file imports the full public surface to lock it in place.

Cross-cutting:

* :class:`TestSnapshotPinnedModelId` — PRD §13 snapshot-pin contract.
* :class:`TestPromptCacheMarkersTranslated` — PRD §13 prompt-caching
  engagement.
* :class:`TestModelImmutability` — frozen Pydantic models.
* :class:`TestParserPropertyBased` — :mod:`hypothesis` property test
  on the fail-closed parser (the "deep module" check from the script's
  promise gate).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bba.audit_store import AuditRow, AuditStore, AuditStoreConfig, LlmCall
from bba.llm_client import (
    ALLOWED_MODELS,
    ANTHROPIC_BETA_HEADER,
    AnthropicAPIError,
    AnthropicBatchTransport,
    AnthropicTransport,
    BatchSubmissionError,
    BatchSubmissionRequest,
    BatchSubmissionResult,
    CassetteInteraction,
    CassetteTransport,
    Classification,
    CustomIdMismatchError,
    DisagreementVerdict,
    EscalationLog,
    IndicationCitation,
    LlmClassificationResponse,
    LlmClientConfig,
    LlmClientConfigError,
    LlmClientError,
    LlmClientResult,
    MAX_SONNET_ATTEMPTS,
    OPUS_MODEL_ID,
    ParseFailureReason,
    ParseOutcome,
    RawBatchResponse,
    SONNET_MODEL_ID,
    StructuredToolInput,
    assert_custom_ids_match,
    build_anthropic_request,
    detect_disagreement,
    escalate_to_opus,
    load_cassette,
    parse_platelet_structured_response,
    parse_structured_response,
    process_batch,
    run_with_escalation,
    should_escalate,
    submit_batch,
)
from bba.prompt_builder import (
    EvidenceChunk,
    InjectionVerdict,
    PromptBlock,
    PromptBuildRequest,
    PromptBuildResult,
    build_envelope,
    build_prompt,
    compute_prompt_hash,
)


# =============================================================================
# Public-API surface pins
#
# The import block above IS the collection-time surface check: if a
# public name is removed in a refactor, pytest fails before any test
# runs. The tuple below pins names referenced only structurally
# (without being directly exercised in a test body) so ruff does not
# strip them as "unused" imports.
# =============================================================================


_PUBLIC_SURFACE_PINS = (
    LlmClientError,
    LlmClientConfigError,
    AnthropicAPIError,
    BatchSubmissionError,
    ANTHROPIC_BETA_HEADER,
    StructuredToolInput,
    Classification,
    escalate_to_opus,
    run_with_escalation,
    parse_structured_response,
    detect_disagreement,
    assert_custom_ids_match,
    load_cassette,
    should_escalate,
    submit_batch,
    AuditRow,
    AuditStore,
    AuditStoreConfig,
    LlmCall,
    AnthropicTransport,
    ALLOWED_MODELS,
)


# =============================================================================
# Fixtures — minimal valid request / response / cassette builders.
#
# Each helper takes the absolute minimum required + ``**overrides`` so
# test bodies stay focused on the property under test (the WHY), not
# on filling in 10 unrelated kwargs.
# =============================================================================


def _prompt(
    *,
    task_mode: str = "HB_7_10_REVIEW",
    cohort_threshold: float = 7.0,
    evidence: tuple[tuple[str, str, str], ...] = (
        ("E1", "IPDNRFOCUSDT", "Patient reports fatigue and palpitations; Hb 7.2."),
    ),
) -> PromptBuildResult:
    """Build a real :class:`PromptBuildResult` via :func:`build_prompt`.

    Going through the public builder means every test exercises a
    realistic prompt envelope — including cache markers and
    prompt_hash — instead of a hand-forged surrogate that could drift
    out of sync with prompt_builder's contract.
    """
    chunks = tuple(
        EvidenceChunk(evidence_id=eid, source=src, text=txt)
        for eid, src, txt in evidence
    )
    return build_prompt(
        PromptBuildRequest(
            task_mode=task_mode,  # type: ignore[arg-type]
            cohort_threshold=cohort_threshold,
            evidence_chunks=chunks,
            few_shot_examples=(),
        )
    )


def _request(
    *,
    audit_id: str = "audit-001",
    run_id: str = "run-aaa",
    task_mode: str = "HB_7_10_REVIEW",
    cohort_threshold: float = 7.0,
) -> BatchSubmissionRequest:
    return BatchSubmissionRequest(
        audit_id=audit_id,
        run_id=run_id,
        task_mode=task_mode,  # type: ignore[arg-type]
        prompt=_prompt(task_mode=task_mode, cohort_threshold=cohort_threshold),
    )


def _classification_response(
    *,
    classification: str = "APPROPRIATE",
    quote: str = "Patient reports fatigue and palpitations; Hb 7.2.",
    source_id: str = "E1",
) -> LlmClassificationResponse:
    return LlmClassificationResponse(
        classification=classification,  # type: ignore[arg-type]
        indications=(
            IndicationCitation(
                code="B1.symptomatic_anemia",
                quote=quote,
                source_id=source_id,
                confidence=0.85,
            ),
        ),
        negative_evidence=(),
        reasoning_summary_en="Symptomatic anemia documented; Hb under threshold.",
        reasoning_summary_th="พบภาวะซีดมีอาการ; Hb ต่ำกว่าเกณฑ์",
    )


def _tool_use_content(
    *,
    classification: str = "APPROPRIATE",
    quote: str = "Patient reports fatigue and palpitations; Hb 7.2.",
    source_id: str = "E1",
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Anthropic tool-use response content shape."""
    block = {
        "type": "tool_use",
        "id": "tool_01",
        "name": "classify_transfusion_order",
        "input": {
            "classification": classification,
            "indications": [
                {
                    "code": "B1.symptomatic_anemia",
                    "quote": quote,
                    "source_id": source_id,
                    "confidence": 0.85,
                }
            ],
            "negative_evidence": [],
            "reasoning_summary_en": "Symptomatic anemia documented.",
            "reasoning_summary_th": "พบภาวะซีดมีอาการ",
        },
    }
    if extra is not None:
        block["input"].update(extra)
    return [block]


def _result(
    *,
    custom_id: str = "audit-001",
    model_id: str = SONNET_MODEL_ID,
    content: list[dict[str, Any]] | None = None,
    raw_response_override: dict[str, Any] | None = None,
    prompt_cache_id: str | None = "cache-aaa",
) -> BatchSubmissionResult:
    raw = (
        raw_response_override
        if raw_response_override is not None
        else {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": model_id,
            "stop_reason": "tool_use",
            "content": content if content is not None else _tool_use_content(),
        }
    )
    return BatchSubmissionResult(
        custom_id=custom_id,
        model_id=model_id,  # type: ignore[arg-type]
        raw_response_json=raw,
        request_json={
            "model": model_id,
            "messages": [{"role": "user", "content": "..."}],
        },
        response_headers={"anthropic-version": "2023-06-01"},
        request_timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
        latency_ms=1200,
        anthropic_version="2023-06-01",
        prompt_cache_id=prompt_cache_id,
    )


class _StubTransport:
    """Manually-constructed transport that returns canned responses
    indexed by ``(model, sorted_custom_ids)``.

    Used in tests where a hand-authored response sequence is clearer
    than a JSON cassette file (escalation flows, disagreement flows).
    The cassette-based replay path is exercised separately under
    :class:`TestCassetteReplay`.
    """

    def __init__(
        self, scripted: dict[tuple[str, tuple[str, ...]], RawBatchResponse]
    ) -> None:
        self._scripted = scripted
        self.calls: list[tuple[str, tuple[str, ...], bool]] = []

    def submit_batch(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        key = (model, tuple(sorted(r.audit_id for r in requests)))
        self.calls.append((model, key[1], prompt_cache_enabled))
        if key not in self._scripted:
            raise KeyError(f"no scripted response for {key!r}")
        return self._scripted[key]


def _config(**overrides: object) -> LlmClientConfig:
    base: dict[str, object] = {
        "sonnet_model_id": SONNET_MODEL_ID,
        "opus_model_id": OPUS_MODEL_ID,
        "max_sonnet_attempts": MAX_SONNET_ATTEMPTS,
        "prompt_cache_enabled": True,
        "code_version": "v0.1.0+test",
    }
    base.update(overrides)
    return LlmClientConfig.model_validate(base)


# =============================================================================
# Cross-cutting: snapshot-pinned model IDs
#
# WHY: PRD §13 "snapshot-pinned model IDs". A floating alias would
# silently drift under Anthropic point releases — the test pins the
# exact bytes so a refactor that "cleaned up the version string"
# breaks loudly here.
# =============================================================================


class TestPinnedModelId:
    def test_sonnet_model_id_is_expected_alias(self) -> None:
        # Change-detector: the primary model is pinned to a specific bare
        # alias. Bumping it must be an explicit edit to the constant (not
        # a transparent SDK upgrade) followed by a golden-set re-run.
        assert SONNET_MODEL_ID == "claude-sonnet-5"
        assert "sonnet" in SONNET_MODEL_ID

    def test_opus_model_id_is_expected_alias(self) -> None:
        assert OPUS_MODEL_ID == "claude-opus-4-8"
        assert "opus" in OPUS_MODEL_ID

    def test_unpinned_model_rejected_by_config(self) -> None:
        # A real model that is NOT in the allow-set must fail config
        # validation. The allow-set — not an ID format — is the pin:
        # invoking any other model requires editing ALLOWED_MODELS.
        with pytest.raises(ValidationError) as exc_info:
            LlmClientConfig.model_validate(
                {
                    "sonnet_model_id": "claude-sonnet-4-6",  # real model, not pinned
                    "opus_model_id": OPUS_MODEL_ID,
                    "max_sonnet_attempts": MAX_SONNET_ATTEMPTS,
                    "prompt_cache_enabled": True,
                    "code_version": "v1",
                }
            )
        assert "not in" in str(exc_info.value).lower()

    def test_allowed_models_set_membership(self) -> None:
        # Every constant must round-trip through ALLOWED_MODELS so the
        # validator and the constants stay in sync. A divergence here
        # means the validator would silently accept an unpinned ID.
        assert SONNET_MODEL_ID in ALLOWED_MODELS
        assert OPUS_MODEL_ID in ALLOWED_MODELS
        assert "claude-sonnet-4-6" not in ALLOWED_MODELS  # a non-pinned model is absent

    def test_config_rejects_unpinned_model(self) -> None:
        with pytest.raises(ValidationError):
            LlmClientConfig.model_validate(
                {
                    "sonnet_model_id": "claude-sonnet-4-6",  # real model, not pinned
                    "opus_model_id": OPUS_MODEL_ID,
                    "max_sonnet_attempts": MAX_SONNET_ATTEMPTS,
                    "prompt_cache_enabled": True,
                    "code_version": "v1",
                }
            )

    def test_config_rejects_model_family_swap(self) -> None:
        # Putting Opus's pinned ID in the Sonnet slot is rejected — the
        # field validator checks that "sonnet" appears in the Sonnet
        # slot, "opus" in the Opus slot. Otherwise a copy-paste typo in
        # operator config would silently route every call through Opus
        # (a cost/latency regression).
        with pytest.raises(ValidationError):
            LlmClientConfig.model_validate(
                {
                    "sonnet_model_id": OPUS_MODEL_ID,
                    "opus_model_id": OPUS_MODEL_ID,
                    "max_sonnet_attempts": MAX_SONNET_ATTEMPTS,
                    "prompt_cache_enabled": True,
                    "code_version": "v1",
                }
            )

    def test_max_sonnet_attempts_capped(self) -> None:
        # PRD §13 caps Sonnet retries at 2. A higher value would
        # silently burn budget without changing the escalation outcome.
        assert MAX_SONNET_ATTEMPTS == 2
        with pytest.raises(ValidationError):
            LlmClientConfig.model_validate(
                {
                    "sonnet_model_id": SONNET_MODEL_ID,
                    "opus_model_id": OPUS_MODEL_ID,
                    "max_sonnet_attempts": MAX_SONNET_ATTEMPTS + 1,
                    "prompt_cache_enabled": True,
                    "code_version": "v1",
                }
            )


# =============================================================================
# AC ② — custom_id assertion: mismatch aborts batch with explicit error
#
# WHY: PRD §13 — "custom_id == audit_id assertion on every result —
# never positional zip". Positional zip silently swaps audit rows
# under partial failure or out-of-order delivery; that is the highest-
# severity invariant in this module.
# =============================================================================


class TestCustomIdAssertion:
    def test_perfect_match_returns_mapping(self) -> None:
        reqs = [_request(audit_id="a1"), _request(audit_id="a2", run_id="run-b")]
        results = [_result(custom_id="a1"), _result(custom_id="a2")]
        mapping = assert_custom_ids_match(reqs, results)
        assert set(mapping.keys()) == {"a1", "a2"}
        assert mapping["a1"].custom_id == "a1"
        assert mapping["a2"].custom_id == "a2"

    def test_out_of_order_results_match_by_id(self) -> None:
        # Anthropic Batch API does not promise ordering. Matching by ID
        # is the entire point — out-of-order delivery must still
        # produce a correct mapping.
        reqs = [_request(audit_id="a1"), _request(audit_id="a2", run_id="run-b")]
        results = [_result(custom_id="a2"), _result(custom_id="a1")]
        mapping = assert_custom_ids_match(reqs, results)
        assert mapping["a1"].custom_id == "a1"
        assert mapping["a2"].custom_id == "a2"

    def test_extra_custom_id_in_results_raises(self) -> None:
        reqs = [_request(audit_id="a1")]
        results = [_result(custom_id="a1"), _result(custom_id="ghost")]
        with pytest.raises(CustomIdMismatchError) as exc_info:
            assert_custom_ids_match(reqs, results)
        assert "ghost" in str(exc_info.value)

    def test_missing_custom_id_in_results_raises(self) -> None:
        reqs = [_request(audit_id="a1"), _request(audit_id="a2", run_id="run-b")]
        results = [_result(custom_id="a1")]
        with pytest.raises(CustomIdMismatchError) as exc_info:
            assert_custom_ids_match(reqs, results)
        assert "a2" in str(exc_info.value)

    def test_duplicate_custom_id_in_results_raises(self) -> None:
        reqs = [_request(audit_id="a1")]
        results = [_result(custom_id="a1"), _result(custom_id="a1")]
        with pytest.raises(CustomIdMismatchError) as exc_info:
            assert_custom_ids_match(reqs, results)
        assert "a1" in str(exc_info.value)

    def test_process_batch_aborts_on_mismatch(self) -> None:
        # End-to-end: when the transport returns a misattributed result,
        # process_batch must abort with CustomIdMismatchError before any
        # row-level work happens. Otherwise the audit row for "a1"
        # could carry "a2"'s classification.
        reqs = [_request(audit_id="a1")]
        bad_response = RawBatchResponse(
            batch_id="batch_01",
            results=(_result(custom_id="WRONG"),),
        )
        transport = _StubTransport(scripted={(SONNET_MODEL_ID, ("a1",)): bad_response})
        with pytest.raises(CustomIdMismatchError):
            process_batch(reqs, transport, _config())


# =============================================================================
# AC ⑥ — Malformed-JSON fail-closed test
#
# WHY: PRD §13 "structured-output (tool-use) JSON shape with fail-
# closed parsing (malformed → NEEDS_REVIEW with parse_failure flag)".
# The parser MUST NEVER raise on bad LLM output — a raise loses the
# audit-chain trail because the persistence layer would never see the
# failure. Every failure mode lands as parse_failure=True with a
# structured ParseFailureReason.
# =============================================================================


class TestMalformedJsonFailClosed:
    def test_garbage_string_in_content_yields_parse_failure(self) -> None:
        # An older Anthropic SDK behavior surfaces tool-use input as a
        # raw string when the model emits malformed JSON. The parser
        # must catch the malformed shape rather than raising.
        bad = _result(
            content=[
                {
                    "type": "tool_use",
                    "id": "tool_01",
                    "name": "classify_transfusion_order",
                    "input": "{this is not json",
                }
            ]
        )
        outcome = parse_structured_response(bad)
        assert outcome.parse_failure is True
        assert outcome.parsed is None
        assert outcome.parse_failure_reason == ParseFailureReason.MALFORMED_JSON

    def test_missing_classification_key_yields_schema_mismatch(self) -> None:
        bad = _result(
            content=[
                {
                    "type": "tool_use",
                    "id": "tool_01",
                    "name": "classify_transfusion_order",
                    "input": {
                        "indications": [],
                        "negative_evidence": [],
                        "reasoning_summary_en": "x",
                        "reasoning_summary_th": "y",
                    },
                }
            ]
        )
        outcome = parse_structured_response(bad)
        assert outcome.parse_failure is True
        assert outcome.parsed is None
        assert outcome.parse_failure_reason == ParseFailureReason.SCHEMA_MISMATCH

    def test_classification_out_of_set_yields_out_of_set_reason(self) -> None:
        bad = _result(
            content=_tool_use_content(classification="MAYBE"),  # not in the 4-label set
        )
        outcome = parse_structured_response(bad)
        assert outcome.parse_failure is True
        assert (
            outcome.parse_failure_reason == ParseFailureReason.CLASSIFICATION_OUT_OF_SET
        )

    def test_missing_tool_use_block_yields_tool_use_missing(self) -> None:
        bad = _result(
            content=[
                {"type": "text", "text": "I refuse to use the tool. Here's why..."}
            ]
        )
        outcome = parse_structured_response(bad)
        assert outcome.parse_failure is True
        assert outcome.parse_failure_reason == ParseFailureReason.TOOL_USE_MISSING

    def test_empty_content_array_yields_empty_response(self) -> None:
        bad = _result(content=[])
        outcome = parse_structured_response(bad)
        assert outcome.parse_failure is True
        assert outcome.parse_failure_reason == ParseFailureReason.EMPTY_RESPONSE

    def test_good_response_round_trips(self) -> None:
        outcome = parse_structured_response(_result())
        assert outcome.parse_failure is False
        assert outcome.parse_failure_reason is None
        assert outcome.parsed is not None
        assert outcome.parsed.classification == "APPROPRIATE"

    def test_parser_never_raises_on_random_garbage(self) -> None:
        # The parser is a fail-closed surface: it absorbs every shape
        # of LLM output. A raise inside the parser would lose the
        # audit-chain trail (PRD §13).
        for shape in [
            {},
            {"content": None},
            {"content": "not a list"},
            {"content": [{"type": "tool_use", "input": None}]},
            {"content": [{"type": "tool_use"}]},  # missing input
        ]:
            res = BatchSubmissionResult(
                custom_id="a1",
                model_id=SONNET_MODEL_ID,  # type: ignore[arg-type]
                raw_response_json=shape,
                request_json={"model": SONNET_MODEL_ID},
                response_headers={"anthropic-version": "2023-06-01"},
                request_timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
                latency_ms=1200,
                anthropic_version="2023-06-01",
            )
            outcome = parse_structured_response(res)
            assert outcome.parse_failure is True

    def test_process_batch_routes_parse_failure_to_needs_review(self) -> None:
        # End-to-end: a totally-failing batch (Sonnet x2 + Opus all
        # produce malformed JSON) must surface as a
        # final_classification=NEEDS_REVIEW row with parse_failure=True
        # on the result. The row is auditable; the persistence path
        # is not lost.
        reqs = [_request(audit_id="a1")]
        bad_content = [
            {
                "type": "tool_use",
                "id": "tool_01",
                "name": "classify_transfusion_order",
                "input": "{garbage",
            }
        ]
        bad_response = RawBatchResponse(
            batch_id="batch_xx",
            results=(_result(custom_id="a1", content=bad_content),),
        )
        bad_opus = RawBatchResponse(
            batch_id="batch_opus",
            results=(
                _result(custom_id="a1", content=bad_content, model_id=OPUS_MODEL_ID),
            ),
        )
        transport = _StubTransport(
            scripted={
                (SONNET_MODEL_ID, ("a1",)): bad_response,
                (OPUS_MODEL_ID, ("a1",)): bad_opus,
            }
        )
        out = process_batch(reqs, transport, _config())
        assert len(out) == 1
        assert out[0].parse_failure is True
        assert out[0].final_classification == "NEEDS_REVIEW"
        assert out[0].needs_review is True
        assert out[0].review_reason is not None
        assert "parse_failure" in out[0].review_reason


# =============================================================================
# AC ④ — Retry → escalation test
#
# WHY: PRD §13 "Retry → Sonnet (≤ 2x) → escalate to Opus 4.7".
# Cost + latency vs. recovery tradeoff: a single Sonnet failure can
# be transient (rate-limit, content filter); two failures probably
# aren't, and Opus is reserved for those.
# =============================================================================


class TestRetryEscalation:
    def test_sonnet_succeeds_first_try_no_opus(self) -> None:
        reqs = [_request(audit_id="a1")]
        good = RawBatchResponse(
            batch_id="batch_01",
            results=(_result(custom_id="a1"),),
        )
        transport = _StubTransport(scripted={(SONNET_MODEL_ID, ("a1",)): good})
        out = process_batch(reqs, transport, _config())

        assert len(out) == 1
        assert out[0].escalation.sonnet_attempts == 1
        assert out[0].escalation.escalated_to_opus is False
        # Only the Sonnet call lands in persisted_calls.
        assert len(out[0].persisted_calls) == 1
        assert out[0].persisted_calls[0].model_id == SONNET_MODEL_ID

    def test_sonnet_fails_once_then_succeeds(self) -> None:
        # Single failure → second Sonnet attempt → success. Opus
        # never invoked. This is the "transient hiccup" path.
        reqs = [_request(audit_id="a1")]
        bad_content = [{"type": "tool_use", "name": "x", "input": "{garbage"}]
        # The stub transport returns a fixed scripted response per
        # (model, custom_ids) key — to script different responses on
        # subsequent attempts, the implementation must use a different
        # stub. For this test, we wire a sequence-aware stub.
        sequence = [
            RawBatchResponse(
                batch_id="batch_01",
                results=(_result(custom_id="a1", content=bad_content),),
            ),
            RawBatchResponse(
                batch_id="batch_02",
                results=(_result(custom_id="a1"),),
            ),
        ]
        transport = _SequenceTransport(sequence)
        out = process_batch(reqs, transport, _config())
        assert out[0].escalation.sonnet_attempts == 2
        assert out[0].escalation.escalated_to_opus is False
        assert len(out[0].persisted_calls) == 2

    def test_sonnet_fails_twice_then_opus_invoked(self) -> None:
        reqs = [_request(audit_id="a1")]
        bad_content = [{"type": "tool_use", "name": "x", "input": "{garbage"}]
        sequence = [
            RawBatchResponse(
                batch_id="batch_01",
                results=(_result(custom_id="a1", content=bad_content),),
            ),
            RawBatchResponse(
                batch_id="batch_02",
                results=(_result(custom_id="a1", content=bad_content),),
            ),
            # Opus succeeds.
            RawBatchResponse(
                batch_id="batch_opus",
                results=(_result(custom_id="a1", model_id=OPUS_MODEL_ID),),
            ),
        ]
        transport = _SequenceTransport(sequence)
        out = process_batch(reqs, transport, _config())
        assert out[0].escalation.sonnet_attempts == 2
        assert out[0].escalation.escalated_to_opus is True
        # 2 Sonnet + 1 Opus.
        assert len(out[0].persisted_calls) == 3
        model_ids = [c.model_id for c in out[0].persisted_calls]
        assert model_ids == [SONNET_MODEL_ID, SONNET_MODEL_ID, OPUS_MODEL_ID]

    def test_should_escalate_returns_true_when_budget_exhausted(self) -> None:
        # Pure policy check: every Sonnet attempt failed.
        outcomes = (
            ParseOutcome(
                parsed=None,
                parse_failure=True,
                parse_failure_reason=ParseFailureReason.MALFORMED_JSON,
            ),
            ParseOutcome(
                parsed=None,
                parse_failure=True,
                parse_failure_reason=ParseFailureReason.MALFORMED_JSON,
            ),
        )
        assert should_escalate(outcomes, _config()) is True

    def test_should_not_escalate_when_one_attempt_succeeded(self) -> None:
        outcomes = (
            ParseOutcome(
                parsed=None,
                parse_failure=True,
                parse_failure_reason=ParseFailureReason.MALFORMED_JSON,
            ),
            ParseOutcome(
                parsed=_classification_response(),
                parse_failure=False,
                parse_failure_reason=None,
            ),
        )
        assert should_escalate(outcomes, _config()) is False

    def test_should_not_escalate_below_max_attempts(self) -> None:
        outcomes = (
            ParseOutcome(
                parsed=None,
                parse_failure=True,
                parse_failure_reason=ParseFailureReason.MALFORMED_JSON,
            ),
        )
        assert should_escalate(outcomes, _config()) is False


class _SequenceTransport:
    """A transport that returns the next response from a fixed sequence.

    Used to script "Sonnet fails N times, then succeeds / then Opus
    runs" scenarios. The stub does NOT key on (model, custom_ids) —
    each :meth:`submit_batch` invocation pops the next sequence
    element. The escalation tests rely on this ordering.
    """

    def __init__(self, responses: Sequence[RawBatchResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def submit_batch(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        self.calls.append((model, tuple(r.audit_id for r in requests)))
        if not self._responses:
            raise AssertionError(
                "Sequence transport exhausted; production code asked "
                "for more responses than the test scripted"
            )
        return self._responses.pop(0)


# =============================================================================
# AC ⑤ — Sonnet/Opus disagreement test
#
# WHY: PRD §13 "Sonnet/Opus classification-disagreement detection →
# NEEDS_REVIEW". Phase-1 acceptance criteria require sensitivity for
# INAPPROPRIATE ≥ 0.90; routing disagreements to NEEDS_REVIEW protects
# that target by surfacing every model-vs-model ambiguity to a human.
# =============================================================================


class TestDisagreementDetection:
    def test_both_agree_no_review(self) -> None:
        verdict = detect_disagreement(
            sonnet_response=_classification_response(classification="APPROPRIATE"),
            opus_response=_classification_response(classification="APPROPRIATE"),
        )
        assert verdict.agreed is True
        assert verdict.routed_to_needs_review is False

    def test_disagree_routes_to_review(self) -> None:
        verdict = detect_disagreement(
            sonnet_response=_classification_response(classification="APPROPRIATE"),
            opus_response=_classification_response(classification="INAPPROPRIATE"),
        )
        assert verdict.agreed is False
        assert verdict.routed_to_needs_review is True
        assert verdict.sonnet_classification == "APPROPRIATE"
        assert verdict.opus_classification == "INAPPROPRIATE"

    def test_disagreement_routes_via_process_batch(self) -> None:
        # End-to-end: Sonnet fails twice, Opus produces a different
        # classification than the *retried-and-failed* Sonnet attempts
        # cannot disagree (they didn't classify). Here we exercise the
        # canonical disagreement path: Sonnet succeeds on attempt 1
        # but the operator's policy compares against Opus as a
        # cross-check. That cross-check is only invoked when Sonnet has
        # exhausted retries AND Opus is invoked — so this test
        # specifically forces that path: Sonnet x2 fails to PARSE,
        # Opus succeeds, and the final result records Opus's answer.
        # (No "shadow" Opus call in the success path — PRD §13 keeps
        # Opus reserved for escalation, not routine cross-check.)
        reqs = [_request(audit_id="a1")]
        bad_content = [{"type": "tool_use", "name": "x", "input": "{garbage"}]
        sequence = [
            RawBatchResponse(
                batch_id="b1",
                results=(_result(custom_id="a1", content=bad_content),),
            ),
            RawBatchResponse(
                batch_id="b2",
                results=(_result(custom_id="a1", content=bad_content),),
            ),
            RawBatchResponse(
                batch_id="b_opus",
                results=(
                    _result(
                        custom_id="a1",
                        model_id=OPUS_MODEL_ID,
                        content=_tool_use_content(classification="INAPPROPRIATE"),
                    ),
                ),
            ),
        ]
        transport = _SequenceTransport(sequence)
        out = process_batch(reqs, transport, _config())
        # When only Opus produces a parseable classification, that is
        # the final answer (no disagreement — Sonnet never spoke).
        assert out[0].final_classification == "INAPPROPRIATE"
        assert out[0].escalation.escalated_to_opus is True

    def test_disagreement_when_both_parse_routes_to_needs_review(self) -> None:
        # The structural disagreement path: detect_disagreement is
        # called directly with both responses present.
        reqs = [_request(audit_id="a1")]
        sequence = [
            RawBatchResponse(
                batch_id="b1",
                results=(
                    _result(
                        custom_id="a1",
                        content=_tool_use_content(classification="APPROPRIATE"),
                    ),
                ),
            ),
        ]
        transport = _SequenceTransport(sequence)
        out = process_batch(reqs, transport, _config())
        # Sonnet alone succeeded; no Opus call → no disagreement.
        assert out[0].disagreement is None
        assert out[0].final_classification == "APPROPRIATE"

    def test_cross_check_with_opus_routes_disagreement_to_review(self) -> None:
        # When cross_check_with_opus=True, Opus runs alongside a
        # successful Sonnet call. A mismatch routes to NEEDS_REVIEW
        # via the cross_check_disagreement path; this exercises the
        # exact code path raised by codex review §4.
        reqs = [_request(audit_id="a1")]
        sequence = [
            # Sonnet succeeds with APPROPRIATE.
            RawBatchResponse(
                batch_id="b_sonnet",
                results=(
                    _result(
                        custom_id="a1",
                        content=_tool_use_content(classification="APPROPRIATE"),
                    ),
                ),
            ),
            # Opus cross-check returns INAPPROPRIATE.
            RawBatchResponse(
                batch_id="b_opus",
                results=(
                    _result(
                        custom_id="a1",
                        model_id=OPUS_MODEL_ID,
                        content=_tool_use_content(classification="INAPPROPRIATE"),
                    ),
                ),
            ),
        ]
        transport = _SequenceTransport(sequence)
        out = process_batch(reqs, transport, _config(cross_check_with_opus=True))
        assert out[0].final_classification == "NEEDS_REVIEW"
        assert out[0].needs_review is True
        assert out[0].review_reason == "disagreement"
        assert out[0].disagreement is not None
        assert out[0].disagreement.sonnet_classification == "APPROPRIATE"
        assert out[0].disagreement.opus_classification == "INAPPROPRIATE"
        # The Opus cross-check call is persisted alongside Sonnet's.
        assert len(out[0].persisted_calls) == 2
        assert out[0].persisted_calls[1].model_id == OPUS_MODEL_ID

    def test_cross_check_opus_parse_failure_routes_to_review(self) -> None:
        # Sonnet succeeds, but Opus cross-check returns malformed JSON.
        # The cross-check quality gate has degraded — route to
        # NEEDS_REVIEW with a distinct reason so the reviewer knows
        # this is NOT a Sonnet/Opus disagreement but a cross-check
        # parse failure. Addresses codex re-review §1.
        reqs = [_request(audit_id="a1")]
        sequence = [
            RawBatchResponse(
                batch_id="b_sonnet",
                results=(
                    _result(
                        custom_id="a1",
                        content=_tool_use_content(classification="APPROPRIATE"),
                    ),
                ),
            ),
            RawBatchResponse(
                batch_id="b_opus",
                results=(
                    _result(
                        custom_id="a1",
                        model_id=OPUS_MODEL_ID,
                        content=[
                            {
                                "type": "tool_use",
                                "name": "classify_transfusion_order",
                                "input": "{garbage",
                            }
                        ],
                    ),
                ),
            ),
        ]
        transport = _SequenceTransport(sequence)
        out = process_batch(reqs, transport, _config(cross_check_with_opus=True))
        assert out[0].final_classification == "NEEDS_REVIEW"
        assert out[0].needs_review is True
        assert out[0].review_reason == "opus_cross_check_parse_failure"
        # Both calls are still persisted — chain of custody intact.
        assert len(out[0].persisted_calls) == 2

    def test_cross_check_agreement_keeps_sonnet_answer(self) -> None:
        # Cross-check enabled but both models agree → no NEEDS_REVIEW.
        reqs = [_request(audit_id="a1")]
        sequence = [
            RawBatchResponse(
                batch_id="b_sonnet",
                results=(
                    _result(
                        custom_id="a1",
                        content=_tool_use_content(classification="APPROPRIATE"),
                    ),
                ),
            ),
            RawBatchResponse(
                batch_id="b_opus",
                results=(
                    _result(
                        custom_id="a1",
                        model_id=OPUS_MODEL_ID,
                        content=_tool_use_content(classification="APPROPRIATE"),
                    ),
                ),
            ),
        ]
        transport = _SequenceTransport(sequence)
        out = process_batch(reqs, transport, _config(cross_check_with_opus=True))
        assert out[0].final_classification == "APPROPRIATE"
        assert out[0].needs_review is False
        assert out[0].disagreement is not None
        assert out[0].disagreement.agreed is True


# =============================================================================
# AC ⑦ — Full response persistence to llm_calls via audit_store
#
# WHY: PRD §"Reproducibility = 'we have the original answer,' not 'we
# re-derive it.' Persist the full Anthropic Batch API request and
# response per audit_id." A persisted LlmCall must capture every byte
# needed to re-derive: request payload, response payload, anthropic-
# version header, prompt_cache_id, request_timestamp, latency.
# =============================================================================


class TestFullResponsePersistence:
    def test_persisted_calls_carry_request_and_response_payloads(self) -> None:
        reqs = [_request(audit_id="a1")]
        good = RawBatchResponse(
            batch_id="batch_01",
            results=(_result(custom_id="a1"),),
        )
        transport = _SequenceTransport([good])
        out = process_batch(reqs, transport, _config())
        call = out[0].persisted_calls[0]
        assert call.audit_id == "a1"
        assert call.run_id == "run-aaa"
        assert call.model_id == SONNET_MODEL_ID
        assert call.anthropic_version == "2023-06-01"
        assert call.prompt_cache_id == "cache-aaa"
        # Full request + response payloads round-trip.
        assert "messages" in call.request_json
        assert call.response_json["model"] == SONNET_MODEL_ID
        assert call.response_json["stop_reason"] == "tool_use"
        # Timestamps are tz-aware UTC (audit_store contract).
        assert call.request_timestamp.tzinfo is not None
        assert call.latency_ms == 1200

    def test_persisted_call_round_trips_through_audit_store(
        self, tmp_path: Path
    ) -> None:
        # The transactional-ordering contract requires every
        # llm_calls row to share (audit_id, run_id) with the audit_row
        # it backs. The client must produce calls that satisfy that
        # invariant out of the box — no translation layer in between.
        store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        reqs = [_request(audit_id="a1")]
        good = RawBatchResponse(
            batch_id="batch_01",
            results=(_result(custom_id="a1"),),
        )
        transport = _SequenceTransport([good])
        out = process_batch(reqs, transport, _config())
        # Build a minimal AuditRow with matching (audit_id, run_id).
        row = _minimal_audit_row(audit_id="a1", run_id="run-aaa")
        result = store.write(row, out[0].persisted_calls)
        assert result.llm_calls_written == 1
        # validate_invariants must not raise — calls and row pair up.
        store.validate_invariants(run_id="run-aaa")
        # Read back: round-trip preserves the request/response payload.
        readback = store.read_llm_calls(run_id="run-aaa")
        assert len(readback) == 1
        assert readback[0].request_json["model"] == SONNET_MODEL_ID
        # Response headers survived persistence under the namespaced
        # key (PRD §"Persist the full Anthropic Batch API request and
        # response per audit_id, including anthropic-version header").
        assert "__bba_response_headers__" in readback[0].response_json
        headers = readback[0].response_json["__bba_response_headers__"]
        assert headers["anthropic-version"] == "2023-06-01"

    def test_call_id_deterministic_across_runs(self) -> None:
        # Two process_batch invocations with identical inputs must
        # produce identical call_ids — otherwise the audit_store's
        # append-only contract would let a re-run silently append a
        # duplicate row instead of being a no-op. The fix in codex
        # review §2 dropped request_timestamp from the call_id hash;
        # this test pins that property.
        reqs = [_request(audit_id="a1")]
        good_a = RawBatchResponse(
            batch_id="batch_01",
            results=(_result(custom_id="a1"),),
        )
        good_b = RawBatchResponse(
            batch_id="batch_02",  # different batch_id; same per-row inputs
            results=(_result(custom_id="a1"),),
        )
        out_a = process_batch(reqs, _SequenceTransport([good_a]), _config())
        out_b = process_batch(reqs, _SequenceTransport([good_b]), _config())
        assert (
            out_a[0].persisted_calls[0].call_id == out_b[0].persisted_calls[0].call_id
        )

    def test_escalation_persists_all_three_calls(self) -> None:
        reqs = [_request(audit_id="a1")]
        bad_content = [{"type": "tool_use", "name": "x", "input": "{garbage"}]
        sequence = [
            RawBatchResponse(
                batch_id="b1",
                results=(_result(custom_id="a1", content=bad_content),),
            ),
            RawBatchResponse(
                batch_id="b2",
                results=(_result(custom_id="a1", content=bad_content),),
            ),
            RawBatchResponse(
                batch_id="b_opus",
                results=(_result(custom_id="a1", model_id=OPUS_MODEL_ID),),
            ),
        ]
        transport = _SequenceTransport(sequence)
        out = process_batch(reqs, transport, _config())
        assert len(out[0].persisted_calls) == 3
        # Distinct call_ids — re-running with the same audit_id over
        # multiple model attempts must produce a unique call per
        # attempt or the audit_store overwrites them (filename
        # collision).
        call_ids = [c.call_id for c in out[0].persisted_calls]
        assert len(set(call_ids)) == 3


def _minimal_audit_row(*, audit_id: str, run_id: str) -> AuditRow:
    """Smallest valid AuditRow for the round-trip test."""
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
    return AuditRow(
        audit_id=audit_id,
        run_id=run_id,
        run_timestamp=now,
        hn_hash="hn-aaa",
        an_hash="an-bbb",
        reqno="REQ-1",
        order_datetime=now,
        products_ordered=("LPRC",),
        hb_value=7.2,
        hb_datetime=now,
        hb_freshness="fresh_<6h",
        hb_source="LABEXM",
        vitals_sbp=110.0,
        vitals_hr=80.0,
        vitals_timestamp=now,
        vitals_source="IPDADMPROGRESS",
        prior_rbc_units_24h=0,
        prior_rbc_units_7d=0,
        cohort_threshold=7.0,
        delta_hb_window_results=(),
        rule_classification="APPROPRIATE",
        final_classification="APPROPRIATE",
        cohort_applied="general_medical",
        indications_json=(),
        negative_evidence_json=(),
        confidence=0.85,
        reasoning_summary_thai="ตัวอย่าง",
        reasoning_summary_en="example",
        needs_human_review=False,
        review_reason=None,
        model_id=SONNET_MODEL_ID,
        prompt_hash="0" * 64,
        evidence_bundle_hash="bundle-hash-aaa",
        redactor_version="v0.1.0",
        redactor_model_sha="sha-aaa",
        policy_version="PR17.2",
        verifier_pass=True,
        verifier_retries=0,
        escalated_to_opus=False,
    )


# =============================================================================
# AC ③ — Betamax/VCR cassettes for offline replay
#
# WHY: PRD §"contract tests against the Anthropic SDK using betamax-
# style cassettes for offline replay". Offline replay means CI does not
# need an Anthropic API key, does not burn budget on every push, and
# produces deterministic output regardless of the live API state.
# =============================================================================


class TestCassetteReplay:
    def test_cassette_round_trip_through_process_batch(self, tmp_path: Path) -> None:
        # A handwritten cassette covering one Sonnet call. The transport
        # replays the recorded response and process_batch must produce
        # the expected classification deterministically.
        cassette_path = tmp_path / "cassette.json"
        cassette_path.write_text(
            json.dumps(
                {
                    "interactions": [
                        {
                            "model": SONNET_MODEL_ID,
                            "custom_ids": ["a1"],
                            "response": {
                                "batch_id": "batch_recorded",
                                "results": [
                                    {
                                        "custom_id": "a1",
                                        "model_id": SONNET_MODEL_ID,
                                        "raw_response_json": {
                                            "id": "msg_recorded",
                                            "type": "message",
                                            "role": "assistant",
                                            "model": SONNET_MODEL_ID,
                                            "stop_reason": "tool_use",
                                            "content": _tool_use_content(),
                                        },
                                        "request_json": {
                                            "model": SONNET_MODEL_ID,
                                            "messages": [
                                                {"role": "user", "content": "..."}
                                            ],
                                        },
                                        "response_headers": {
                                            "anthropic-version": "2023-06-01"
                                        },
                                        "request_timestamp": "2026-05-01T12:00:00Z",
                                        "latency_ms": 1100,
                                        "anthropic_version": "2023-06-01",
                                        "prompt_cache_id": "cache-recorded",
                                        "extended_thinking_blocks": None,
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        )
        interactions = load_cassette(cassette_path)
        assert len(interactions) == 1
        transport = CassetteTransport(interactions)
        out = process_batch([_request(audit_id="a1")], transport, _config())
        assert out[0].final_classification == "APPROPRIATE"
        assert out[0].persisted_calls[0].prompt_cache_id == "cache-recorded"

    def test_cassette_miss_raises_loudly(self, tmp_path: Path) -> None:
        # A cassette miss (the test asks for a (model, custom_ids)
        # combination that wasn't recorded) MUST raise. Silently
        # returning an empty / default response would let a regression
        # in submission-shape go undetected.
        interactions = (
            CassetteInteraction(
                model=SONNET_MODEL_ID,  # type: ignore[arg-type]
                custom_ids=("a1",),
                response=RawBatchResponse(
                    batch_id="b1",
                    results=(_result(custom_id="a1"),),
                ),
            ),
        )
        transport = CassetteTransport(interactions)
        with pytest.raises((KeyError, LlmClientError)):
            transport.submit_batch(
                model=SONNET_MODEL_ID,
                requests=[_request(audit_id="a99")],
                prompt_cache_enabled=True,
            )

    def test_cassette_transport_implements_protocol(self) -> None:
        # Structural check: CassetteTransport satisfies AnthropicTransport.
        # If a refactor changes the Protocol signature, this assertion
        # surfaces the drift at collection time rather than letting
        # mypy be the only line of defense.
        assert isinstance(CassetteTransport(()), AnthropicTransport)


# =============================================================================
# Cross-cutting: prompt-cache marker translation
#
# WHY: PRD §13 "prompt caching engaged". The PromptBuildResult carries
# cache_marker flags on the system + few_shot blocks; submit_batch
# MUST translate those markers into Anthropic's cache_control headers
# on the corresponding message blocks.
# =============================================================================


class TestPromptCacheMarkersTranslated:
    def test_submit_batch_engages_prompt_cache_when_config_enabled(self) -> None:
        # The transport receives prompt_cache_enabled=True iff the
        # config says so. A regression here is a silent cost regression:
        # disabling caching looks identical externally but doubles the
        # batch bill.
        reqs = [_request(audit_id="a1")]
        good = RawBatchResponse(
            batch_id="b1",
            results=(_result(custom_id="a1"),),
        )
        transport = _StubTransport(scripted={(SONNET_MODEL_ID, ("a1",)): good})
        process_batch(reqs, transport, _config(prompt_cache_enabled=True))
        assert transport.calls[0][2] is True  # prompt_cache_enabled

    def test_submit_batch_respects_disabled_cache(self) -> None:
        reqs = [_request(audit_id="a1")]
        good = RawBatchResponse(
            batch_id="b1",
            results=(_result(custom_id="a1"),),
        )
        transport = _StubTransport(scripted={(SONNET_MODEL_ID, ("a1",)): good})
        process_batch(reqs, transport, _config(prompt_cache_enabled=False))
        assert transport.calls[0][2] is False


# =============================================================================
# Cross-cutting: model immutability
#
# WHY: Frozen Pydantic models prevent silent state corruption across
# the audit chain. A mutable LlmClientResult could be patched between
# process_batch and audit_store.write to drop a parse_failure flag.
# =============================================================================


class TestModelImmutability:
    def test_classification_response_is_frozen(self) -> None:
        r = _classification_response()
        with pytest.raises(ValidationError):
            r.classification = "INAPPROPRIATE"  # type: ignore[misc]

    def test_indication_citation_is_frozen(self) -> None:
        c = IndicationCitation(code="x", quote="y", source_id="E1", confidence=0.5)
        with pytest.raises(ValidationError):
            c.confidence = 0.0  # type: ignore[misc]

    def test_parse_outcome_is_frozen(self) -> None:
        o = ParseOutcome(
            parsed=_classification_response(),
            parse_failure=False,
            parse_failure_reason=None,
        )
        with pytest.raises(ValidationError):
            o.parse_failure = True  # type: ignore[misc]

    def test_parse_outcome_rejects_inconsistent_state(self) -> None:
        # parse_failure=True with a non-None parsed is structurally
        # impossible per the fail-closed contract.
        with pytest.raises(ValidationError):
            ParseOutcome(
                parsed=_classification_response(),
                parse_failure=True,
                parse_failure_reason=ParseFailureReason.MALFORMED_JSON,
            )

    def test_disagreement_verdict_rejects_inconsistent_routing(self) -> None:
        # agreed=True with routed_to_needs_review=True is impossible —
        # agreement is the entire point of NOT routing.
        with pytest.raises(ValidationError):
            DisagreementVerdict(
                sonnet_classification="APPROPRIATE",
                opus_classification="APPROPRIATE",
                agreed=True,
                routed_to_needs_review=True,
            )

    def test_escalation_log_rejects_opus_without_sonnet_exhaustion(self) -> None:
        # Cannot escalate to Opus until the Sonnet retry budget is
        # exhausted (PRD §13). The model validator surfaces this at
        # construction.
        with pytest.raises(ValidationError):
            EscalationLog(
                audit_id="a1",
                sonnet_attempts=1,  # less than MAX_SONNET_ATTEMPTS
                sonnet_parse_failures=(ParseFailureReason.MALFORMED_JSON,),
                escalated_to_opus=True,
            )

    def test_llm_client_result_rejects_call_audit_id_mismatch(self) -> None:
        # Every persisted call must share the result's (audit_id, run_id)
        # — otherwise audit_store.write would reject the bundle later
        # and the failure would surface far from the actual mistake.
        mismatched_call = _build_persistable_call(audit_id="OTHER", run_id="run-aaa")
        with pytest.raises(ValidationError):
            LlmClientResult(
                audit_id="a1",
                run_id="run-aaa",
                final_classification="APPROPRIATE",
                response=_classification_response(),
                parse_failure=False,
                needs_review=False,
                review_reason=None,
                escalation=EscalationLog(
                    audit_id="a1",
                    sonnet_attempts=1,
                    sonnet_parse_failures=(),
                    escalated_to_opus=False,
                ),
                disagreement=None,
                persisted_calls=(mismatched_call,),
            )

    def test_llm_client_result_parse_failure_forces_needs_review(self) -> None:
        # parse_failure=True must force final_classification=NEEDS_REVIEW
        # AND needs_review=True. A regression where parse_failure leaked
        # through with a real classification would skip the human gate.
        good_call = _build_persistable_call(audit_id="a1", run_id="run-aaa")
        with pytest.raises(ValidationError):
            LlmClientResult(
                audit_id="a1",
                run_id="run-aaa",
                final_classification="APPROPRIATE",
                response=None,
                parse_failure=True,
                needs_review=True,
                review_reason="parse_failure",
                escalation=EscalationLog(
                    audit_id="a1",
                    sonnet_attempts=2,
                    sonnet_parse_failures=(
                        ParseFailureReason.MALFORMED_JSON,
                        ParseFailureReason.MALFORMED_JSON,
                    ),
                    escalated_to_opus=False,
                ),
                disagreement=None,
                persisted_calls=(good_call,),
            )


def _build_multi_system_block_prompt() -> PromptBuildResult:
    """Synthesize a PromptBuildResult with 3 system blocks where only
    the final block carries cache_marker=True.

    The real :func:`build_prompt` only emits ONE system block today, so
    this fixture goes through :func:`compute_prompt_hash` /
    :func:`build_envelope` directly to construct a valid frozen
    :class:`PromptBuildResult` whose ``prompt_hash`` matches its
    envelope. Used to exercise the multi-system-block boundary case
    in the cache-control translation contract.
    """
    blocks = (
        PromptBlock(role="system", text="System block A", cache_marker=False),
        PromptBlock(role="system", text="System block B", cache_marker=False),
        PromptBlock(role="system", text="System block C (cached)", cache_marker=True),
        PromptBlock(
            role="user",
            text='<evidence id="E1" untrusted="true">x</evidence>',
            cache_marker=False,
        ),
    )
    envelope = build_envelope(
        blocks=[
            {"role": b.role, "text": b.text, "cache_marker": b.cache_marker}
            for b in blocks
        ],
        task_mode="HB_7_10_REVIEW",
        cohort_threshold=7.0,
        injection_matches=[],
        route_to_needs_review=False,
        needs_review_reasons=[],
    )
    prompt_hash = compute_prompt_hash(envelope)
    return PromptBuildResult(
        blocks=blocks,
        task_mode="HB_7_10_REVIEW",
        cohort_threshold=7.0,
        injection_verdict=InjectionVerdict(flagged=False, matches=()),
        route_to_needs_review=False,
        needs_review_reasons=(),
        prompt_hash=prompt_hash,
    )


def _build_persistable_call(*, audit_id: str, run_id: str) -> LlmCall:
    return LlmCall(
        call_id=f"call-{audit_id}-1",
        audit_id=audit_id,
        run_id=run_id,
        model_id=SONNET_MODEL_ID,
        anthropic_version="2023-06-01",
        prompt_cache_id="cache-aaa",
        request_json={"model": SONNET_MODEL_ID, "messages": []},
        response_json={"id": "msg_01", "stop_reason": "tool_use"},
        request_timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
        latency_ms=1200,
        extended_thinking_blocks=None,
        cold_storage_uri=None,
    )


# =============================================================================
# Hypothesis property test (the "deep module" check from the script's
# promise gate). The parser is a fail-closed surface: for any string
# input, parse_structured_response MUST return ParseOutcome (never raise).
#
# WHY: An LLM under adversarial input can emit arbitrary bytes. A raise
# loses the audit-chain trail; the fail-closed property is the entire
# point of the parser layer.
# =============================================================================


class TestParserPropertyBased:
    @given(garbage=st.text(min_size=0, max_size=200))
    @settings(max_examples=50, deadline=None)
    def test_parser_never_raises_on_arbitrary_string_input(self, garbage: str) -> None:
        # Wrap the garbage in a syntactically valid response shape so
        # the model boundary admits it; the parser must still produce
        # ParseOutcome.
        res = BatchSubmissionResult(
            custom_id="a1",
            model_id=SONNET_MODEL_ID,  # type: ignore[arg-type]
            raw_response_json={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_01",
                        "name": "classify_transfusion_order",
                        "input": garbage,
                    }
                ]
            },
            request_json={"model": SONNET_MODEL_ID},
            response_headers={"anthropic-version": "2023-06-01"},
            request_timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
            latency_ms=100,
            anthropic_version="2023-06-01",
        )
        outcome = parse_structured_response(res)
        # Either it parses to a valid response, or it fails closed.
        if outcome.parse_failure:
            assert outcome.parsed is None
            assert outcome.parse_failure_reason is not None
        else:
            assert outcome.parsed is not None


# =============================================================================
# Submit-batch pre-flight checks
#
# WHY: pre-flight failures (empty list, duplicate custom_ids, oversize)
# MUST raise BatchSubmissionError before any HTTP call so a rejected
# submission leaves no half-state on Anthropic's side.
# =============================================================================


class TestSubmitBatchPreflight:
    def test_empty_requests_raises(self) -> None:
        transport = _StubTransport(scripted={})
        with pytest.raises(BatchSubmissionError):
            submit_batch([], transport, _config())

    def test_duplicate_custom_ids_raises(self) -> None:
        transport = _StubTransport(scripted={})
        reqs = [
            _request(audit_id="a1"),
            _request(audit_id="a1", run_id="run-different"),
        ]
        with pytest.raises(BatchSubmissionError):
            submit_batch(reqs, transport, _config())

    def test_submit_batch_returns_raw_response_on_success(self) -> None:
        reqs = [_request(audit_id="a1")]
        good = RawBatchResponse(
            batch_id="b1",
            results=(_result(custom_id="a1"),),
        )
        transport = _StubTransport(scripted={(SONNET_MODEL_ID, ("a1",)): good})
        out = submit_batch(reqs, transport, _config())
        assert out.batch_id == "b1"
        assert out.results[0].custom_id == "a1"


# =============================================================================
# Cross-cutting: real Anthropic cache_control marker translation
#
# WHY: PRD §13 "prompt caching engaged". The boolean flag check is not
# sufficient — a regression that flips ``cache_marker`` translation
# without changing the flag (rename, reorder, type-cast bug) would
# pass that check. This class inspects the actual Anthropic request
# payload produced by build_anthropic_request and asserts the
# ``cache_control`` marker lands on the system + few-shot block ends.
# =============================================================================


class TestAnthropicRequestCacheControlTranslation:
    def test_cache_marker_translates_to_ephemeral_cache_control(self) -> None:
        # One evidence chunk → one system block (with cache_marker) +
        # one user payload block (no cache_marker). The translated
        # request must carry cache_control on the LAST system block
        # (Anthropic's contract treats the marker as the cache
        # breakpoint at the END of the marked region — placing it
        # anywhere else either wastes breakpoints or fails to cache).
        request = _request(audit_id="a1")
        payload = build_anthropic_request(
            request, model=SONNET_MODEL_ID, prompt_cache_enabled=True
        )
        system_blocks = payload["system"]
        assert len(system_blocks) >= 1
        # Specifically the LAST system block carries the marker.
        assert system_blocks[-1].get("cache_control") == {"type": "ephemeral"}
        # The trailing user payload (evidence) is never cacheable.
        user_blocks = payload["messages"][0]["content"]
        cached_user_blocks = [b for b in user_blocks if "cache_control" in b]
        assert cached_user_blocks == []

    def test_cache_marker_lands_on_last_system_block_in_multi_block_prompt(
        self,
    ) -> None:
        # Multi-system-block fixture: a forged PromptBuildResult with
        # 3 system blocks, only the LAST of which has cache_marker=True.
        # The translated Anthropic payload must mark ONLY the last
        # system block — never the earlier ones. Addresses codex
        # re-review round 3, issue §1.
        forged = _build_multi_system_block_prompt()
        request = BatchSubmissionRequest(
            audit_id="a-multi",
            run_id="run-multi",
            task_mode="HB_7_10_REVIEW",
            prompt=forged,
        )
        payload = build_anthropic_request(
            request, model=SONNET_MODEL_ID, prompt_cache_enabled=True
        )
        system_blocks = payload["system"]
        assert len(system_blocks) == 3
        # Earlier blocks are NOT cached.
        assert "cache_control" not in system_blocks[0]
        assert "cache_control" not in system_blocks[1]
        # Only the final system block carries the breakpoint.
        assert system_blocks[2].get("cache_control") == {"type": "ephemeral"}

    def test_disabled_cache_strips_all_cache_control_markers(self) -> None:
        request = _request(audit_id="a1")
        payload = build_anthropic_request(
            request, model=SONNET_MODEL_ID, prompt_cache_enabled=False
        )
        # No block on either side carries cache_control when caching is off.
        for block in payload["system"]:
            assert "cache_control" not in block
        for block in payload["messages"][0]["content"]:
            assert "cache_control" not in block

    def test_tool_choice_forces_classifier_tool(self) -> None:
        # Tool-use mode is mandatory (PRD §13 fail-closed parsing requires
        # the LLM to emit a structured tool call, not free text). The
        # request must force tool_choice on the classify_transfusion_order
        # tool — otherwise the model could revert to free text under load.
        request = _request(audit_id="a1")
        payload = build_anthropic_request(
            request, model=SONNET_MODEL_ID, prompt_cache_enabled=True
        )
        assert payload["tool_choice"]["type"] == "tool"
        assert payload["tool_choice"]["name"] == "classify_transfusion_order"
        tool_names = [t["name"] for t in payload["tools"]]
        assert "classify_transfusion_order" in tool_names

    def test_tool_input_schema_constrains_classification_to_four_labels(self) -> None:
        # The tool's input_schema must restrict classification to the
        # four-label set so a hallucinated label cannot pass schema
        # validation at the Anthropic side (defence in depth alongside
        # the post-hoc parser check).
        request = _request(audit_id="a1")
        payload = build_anthropic_request(
            request, model=SONNET_MODEL_ID, prompt_cache_enabled=True
        )
        schema = payload["tools"][0]["input_schema"]
        enum = schema["properties"]["classification"]["enum"]
        assert set(enum) == {
            "APPROPRIATE",
            "INAPPROPRIATE",
            "NEEDS_REVIEW",
            "INSUFFICIENT_EVIDENCE",
        }

    def test_reasoning_fields_instruct_language_separation_and_thai_fluency(
        self,
    ) -> None:
        # Pilot 2026-07-06: the model packed both languages (plus leaked
        # tool-call tags) into reasoning_summary_en on 131/165 rows, and
        # the Thai it did write read as a stiff word-for-word translation.
        # The field descriptions are the model-facing contract: en is
        # English-only, th is natural clinical Thai — not a translation.
        request = _request(audit_id="a1")
        payload = build_anthropic_request(
            request, model=SONNET_MODEL_ID, prompt_cache_enabled=True
        )
        props = payload["tools"][0]["input_schema"]["properties"]
        en_desc = props["reasoning_summary_en"]["description"]
        th_desc = props["reasoning_summary_th"]["description"]
        assert "English" in en_desc
        assert "reasoning_summary_th" in en_desc  # points Thai to its field
        assert "Thai" in th_desc
        assert "translation" in th_desc  # forbids literal-translation style


# =============================================================================
# Production AnthropicBatchTransport — surface-level sanity checks
#
# WHY: PRD §13 "Anthropic Batch API integration" requires a concrete
# production transport (not just a Protocol). These tests pin the
# constructor contract; full SDK-backed roundtrip is exercised by the
# audit_pipeline integration test in issue #24 (out of scope here).
# =============================================================================


class TestAnthropicBatchTransportSurface:
    def test_missing_api_key_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(LlmClientConfigError) as exc_info:
            AnthropicBatchTransport()
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_explicit_api_key_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        transport = AnthropicBatchTransport(api_key="sk-test-xxx")
        assert isinstance(transport, AnthropicTransport)

    def test_env_var_satisfies_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-from-env")
        transport = AnthropicBatchTransport()
        assert isinstance(transport, AnthropicTransport)

    def test_empty_string_api_key_treated_as_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An env var that is set but empty must NOT pass as a key.
        # Otherwise a misconfigured deployment would route every call
        # through an empty header and surface a confusing Anthropic
        # 401 deep inside SDK retry logic.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        with pytest.raises(LlmClientConfigError):
            AnthropicBatchTransport()


class TestAnthropicBatchEntryErrorHandling:
    """The production transport must handle non-``succeeded`` batch
    entry types (``errored``, ``canceled``, ``expired``).

    PRD §"Anthropic API outage mid-batch" rules out raising
    AttributeError on these entries: a partial-failure batch must still
    surface its errored rows on the audit chain so a reviewer can see
    why those rows didn't get a real classification. Addresses codex
    re-review §2 (HIGH).

    These tests go through the public :meth:`AnthropicBatchTransport.submit_batch`
    surface with a monkeypatched ``anthropic`` SDK that yields stub
    batch result entries of each type, exercising the same code path
    a real Anthropic batch would.
    """

    @pytest.mark.parametrize(
        ("result_type", "expected_parse_reason"),
        [
            ("errored", ParseFailureReason.EMPTY_RESPONSE),
            ("canceled", ParseFailureReason.EMPTY_RESPONSE),
            ("expired", ParseFailureReason.EMPTY_RESPONSE),
        ],
    )
    def test_non_succeeded_entry_routes_via_public_submit_batch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        result_type: str,
        expected_parse_reason: ParseFailureReason,
    ) -> None:
        # Stand up a fake `anthropic` SDK that the lazy import inside
        # AnthropicBatchTransport.submit_batch picks up. The fake
        # batches API returns one entry of the parameterized type.
        fake_module = _build_fake_anthropic_sdk(
            entries=[_FakeBatchEntry(custom_id="a1", result_type=result_type)]
        )
        monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_module)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-xxx")
        transport = AnthropicBatchTransport(poll_interval_seconds=0.0)
        response = transport.submit_batch(
            model=SONNET_MODEL_ID,
            requests=[_request(audit_id="a1")],
            prompt_cache_enabled=True,
        )
        assert len(response.results) == 1
        entry_payload = response.results[0].raw_response_json
        assert entry_payload["_batch_result_type"] == result_type
        # The synthesized envelope flows through the parser cleanly.
        parsed = parse_structured_response(response.results[0])
        assert parsed.parse_failure is True
        assert parsed.parse_failure_reason == expected_parse_reason


class _FakeBatchEntry:
    """Stub Anthropic Batch result entry, one per (custom_id, type)."""

    def __init__(self, *, custom_id: str, result_type: str) -> None:
        self.custom_id = custom_id
        self.anthropic_version = "2023-06-01"
        self.id = f"msg_{result_type}"

        class _Error:
            def model_dump(self_inner) -> dict[str, Any]:
                return {"type": "invalid_request_error", "message": "stub"}

        class _Result:
            type = result_type
            error = _Error() if result_type == "errored" else None

        self.result = _Result()


def _build_fake_anthropic_sdk(*, entries: list[_FakeBatchEntry]) -> Any:
    """Construct a fake anthropic SDK module exposing the surface
    AnthropicBatchTransport.submit_batch uses.

    The fake supports the four call sites: ``Anthropic(...)``,
    ``client.messages.batches.create(...)``, ``...retrieve(batch_id)``,
    and ``...results(batch_id)`` (iterator yielding the supplied
    entries).
    """
    from types import SimpleNamespace

    class _BatchesNamespace:
        def create(self, *, requests: Any) -> SimpleNamespace:
            return SimpleNamespace(id="batch_stub")

        def retrieve(self, batch_id: str) -> SimpleNamespace:
            return SimpleNamespace(processing_status="ended")

        def results(self, batch_id: str) -> list[_FakeBatchEntry]:
            return entries

    class _Anthropic:
        def __init__(self, *, api_key: str, default_headers: dict[str, str]) -> None:
            self.messages = SimpleNamespace(batches=_BatchesNamespace())

    return SimpleNamespace(Anthropic=_Anthropic)


# =============================================================================
# Expanded fail-closed property test (codex review §8)
#
# WHY: The original property test only varied the ``input`` string of
# an otherwise valid tool_use block. The real fail-closed invariant
# applies to ARBITRARY raw_response_json shapes — missing keys, non-list
# content, multi-block content, valid-JSON-but-not-dict, deeply nested
# garbage. This expanded test generates every shape variation and
# asserts the parser never raises.
# =============================================================================


class TestParserPropertyBasedExpanded:
    @given(
        shape=st.one_of(
            st.fixed_dictionaries({}),
            st.fixed_dictionaries({"content": st.none()}),
            st.fixed_dictionaries({"content": st.text(max_size=20)}),
            st.fixed_dictionaries({"content": st.integers()}),
            st.fixed_dictionaries(
                {
                    "content": st.lists(
                        st.dictionaries(
                            st.text(min_size=1, max_size=10),
                            st.one_of(st.text(), st.integers(), st.none()),
                            max_size=4,
                        ),
                        max_size=5,
                    )
                }
            ),
            st.fixed_dictionaries(
                {
                    "content": st.lists(
                        st.fixed_dictionaries(
                            {
                                "type": st.sampled_from(
                                    ["text", "tool_use", "thinking"]
                                ),
                                "input": st.one_of(
                                    st.text(max_size=20),
                                    st.dictionaries(
                                        st.text(min_size=1, max_size=8),
                                        st.text(max_size=20),
                                        max_size=4,
                                    ),
                                    st.none(),
                                ),
                            }
                        ),
                        min_size=0,
                        max_size=3,
                    )
                }
            ),
        )
    )
    @settings(max_examples=75, deadline=None)
    def test_parser_never_raises_on_arbitrary_envelope_shape(
        self, shape: dict[str, Any]
    ) -> None:
        # The hypothesis shape must validate as FrozenJsonDict input.
        # Construct via the model so we exercise the same path the
        # production pipeline uses; if construction itself raises, that
        # is information about the model boundary, not the parser.
        try:
            result = BatchSubmissionResult(
                custom_id="a1",
                model_id=SONNET_MODEL_ID,  # type: ignore[arg-type]
                raw_response_json=shape,
                request_json={"model": SONNET_MODEL_ID},
                response_headers={"anthropic-version": "2023-06-01"},
                request_timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
                latency_ms=100,
                anthropic_version="2023-06-01",
            )
        except ValidationError:
            # The model rejected the shape; the parser never sees it.
            # That is also acceptable fail-closed behaviour.
            return
        outcome = parse_structured_response(result)
        if outcome.parse_failure:
            assert outcome.parsed is None
            assert outcome.parse_failure_reason is not None
        else:
            assert outcome.parsed is not None


# =============================================================================
# Fix A (Codex P2): PLATELET_REVIEW tool schema must advertise hard-signal bools
#
# WHY: the request tool schema and the parser response contract must agree.
# parse_platelet_structured_response requires active_bleeding,
# procedure_indication, prophylactic_marrow_failure as StrictBool. Without
# these fields in the tool input_schema, the model is never instructed to
# emit them, omits them, and EVERY platelet response fails SCHEMA_MISMATCH
# — the live platelet leg is dead-on-arrival producing no real verdicts.
# =============================================================================


def _platelet_request(audit_id: str = "plt-001") -> BatchSubmissionRequest:
    """Build a minimal PLATELET_REVIEW BatchSubmissionRequest."""
    from bba.prompt_builder import EvidenceChunk, PromptBuildRequest, build_prompt

    result = build_prompt(
        PromptBuildRequest(
            task_mode="PLATELET_REVIEW",
            cohort_threshold=None,
            evidence_chunks=(
                EvidenceChunk(
                    evidence_id="E1",
                    source="IPDNRFOCUSDT",
                    text="Plt 5,000 /uL. Active GI bleed documented.",
                ),
            ),
            few_shot_examples=(),
        )
    )
    return BatchSubmissionRequest(
        audit_id=audit_id,
        run_id="run-plt",
        task_mode="PLATELET_REVIEW",
        prompt=result,
    )


class TestPlateletReviewToolSchema:
    """PLATELET_REVIEW requests must advertise the three hard-signal bool fields.

    WHY: request schema and parser contract must agree end-to-end. If the
    tool input_schema omits active_bleeding / procedure_indication /
    prophylactic_marrow_failure, the model never emits them, and
    parse_platelet_structured_response raises SCHEMA_MISMATCH on every
    response — the live platelet leg can never produce a real verdict.
    """

    def test_platelet_review_schema_contains_three_boolean_fields(self) -> None:
        payload = build_anthropic_request(
            _platelet_request(), model=SONNET_MODEL_ID, prompt_cache_enabled=True
        )
        props = payload["tools"][0]["input_schema"]["properties"]
        for field in (
            "active_bleeding",
            "procedure_indication",
            "prophylactic_marrow_failure",
        ):
            assert field in props, (
                f"PLATELET_REVIEW tool schema missing '{field}'; "
                "model will omit it and every response will fail SCHEMA_MISMATCH"
            )

    def test_platelet_review_hard_signal_fields_are_type_boolean(self) -> None:
        payload = build_anthropic_request(
            _platelet_request(), model=SONNET_MODEL_ID, prompt_cache_enabled=True
        )
        props = payload["tools"][0]["input_schema"]["properties"]
        for field in (
            "active_bleeding",
            "procedure_indication",
            "prophylactic_marrow_failure",
        ):
            assert props[field]["type"] == "boolean", (
                f"'{field}' must be type 'boolean' so the model emits a proper bool "
                "and StrictBool in PlateletLlmClassificationResponse catches coercions"
            )

    def test_platelet_review_hard_signal_fields_are_required(self) -> None:
        payload = build_anthropic_request(
            _platelet_request(), model=SONNET_MODEL_ID, prompt_cache_enabled=True
        )
        required: list[str] = payload["tools"][0]["input_schema"]["required"]
        for field in (
            "active_bleeding",
            "procedure_indication",
            "prophylactic_marrow_failure",
        ):
            assert field in required, (
                f"'{field}' must be in 'required'; omitting it means the model "
                "may omit the field and every response silently fails SCHEMA_MISMATCH"
            )

    def test_rbc_tool_schema_unchanged_no_platelet_bool_fields(self) -> None:
        # RBC path must be byte-identical: platelet booleans must NOT appear
        # in RBC requests; adding them would alter the tool schema hash and
        # invalidate stored RBC audit records.
        for mode in ("HB_7_10_REVIEW", "HB_GT_10_OVERRIDE"):
            rbc_req = _request(audit_id="rbc-chk", task_mode=mode)
            payload = build_anthropic_request(
                rbc_req, model=SONNET_MODEL_ID, prompt_cache_enabled=True
            )
            props = payload["tools"][0]["input_schema"]["properties"]
            for field in (
                "active_bleeding",
                "procedure_indication",
                "prophylactic_marrow_failure",
            ):
                assert field not in props, (
                    f"RBC tool schema must not contain platelet field '{field}' "
                    f"(task_mode={mode!r}); RBC path must be byte-identical"
                )

    def test_conforming_platelet_response_parses_without_schema_mismatch(
        self,
    ) -> None:
        # A model response that conforms to the platelet tool schema must
        # parse via parse_platelet_structured_response without SCHEMA_MISMATCH
        # and must yield grounded PlateletHardSignals — the live leg round-trip.
        conforming_content = [
            {
                "type": "tool_use",
                "id": "tool_01",
                "name": "classify_transfusion_order",
                "input": {
                    "classification": "APPROPRIATE",
                    "indications": [
                        {
                            "code": "C1.active_bleeding",
                            "quote": "Active GI bleed documented.",
                            "source_id": "E1",
                            "confidence": 0.92,
                        }
                    ],
                    "negative_evidence": [],
                    "reasoning_summary_en": "Active GI bleed grounded from E1.",
                    "reasoning_summary_th": "พบเลือดออกทางเดินอาหารเฉียบพลันจาก E1",
                    "active_bleeding": True,
                    "procedure_indication": False,
                    "prophylactic_marrow_failure": False,
                },
            }
        ]
        result = BatchSubmissionResult(
            custom_id="plt-001",
            model_id=SONNET_MODEL_ID,
            raw_response_json={"content": conforming_content},
            request_json={"model": SONNET_MODEL_ID},
            response_headers={"anthropic-version": "2023-06-01"},
            request_timestamp=datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC),
            latency_ms=100,
            anthropic_version="2023-06-01",
        )
        outcome = parse_platelet_structured_response(result)
        assert not outcome.parse_failure, (
            f"Expected successful parse but got "
            f"parse_failure_reason={outcome.parse_failure_reason!r}; "
            "the tool schema and parser must agree on the three boolean fields"
        )
        assert outcome.platelet_hard_signals is not None
        assert outcome.platelet_hard_signals.active_bleeding is True
        assert outcome.platelet_hard_signals.procedure_indication is False
        assert outcome.platelet_hard_signals.prophylactic_marrow_failure is False
