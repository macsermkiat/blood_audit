"""bba.llm_client — Anthropic Batch API wrapper with retry + escalation.

See issue #22 for acceptance criteria. PRD §13 (Implementation Decisions)
defines the contract:

* Snapshot-pinned model IDs (Sonnet 4.6, Opus 4.7) — drift is detected
  by quarterly golden-set re-run.
* ``custom_id == audit_id`` assertion on every Batch API result — never
  positional zip. Mismatch aborts the batch with an explicit error.
* Anthropic prompt-caching engaged on the system + few-shot blocks
  (translated from :class:`bba.prompt_builder.PromptBlock.cache_marker`).
* Structured-output (tool-use) JSON shape with fail-closed parsing:
  malformed output -> :class:`ParseOutcome.parse_failure = True` and
  routes to ``NEEDS_REVIEW`` with the ``parse_failure`` flag.
* Retry policy: up to 2 Sonnet attempts; further failures escalate to
  Opus 4.7. If Opus still fails (parse / grounding), the row routes to
  ``NEEDS_REVIEW``.
* Sonnet/Opus classification-disagreement detection: when both succeed
  with different classifications, route to ``NEEDS_REVIEW`` with the
  ``disagreement`` reason.
* Persists FULL response object (request payload, response payload,
  response headers, extended-thinking blocks, prompt_cache_id) to the
  ``llm_calls`` parquet via :mod:`bba.audit_store`.

This module sits between :mod:`bba.prompt_builder` (issue #21) and
:mod:`bba.audit_store` (issue #19) in the audit pipeline. The HTTP
boundary is injected as :class:`AnthropicTransport` so tests replay
recorded cassettes offline (Betamax/VCR pattern; PRD §"contract tests
against the Anthropic SDK").
"""

from bba.llm_client.cassette import CassetteTransport, load_cassette
from bba.llm_client.client import process_batch, submit_batch
from bba.llm_client.custom_id import assert_custom_ids_match
from bba.llm_client.disagreement import detect_disagreement
from bba.llm_client.escalation import (
    escalate_to_opus,
    run_with_escalation,
    should_escalate,
)
from bba.llm_client.exceptions import (
    AnthropicAPIError,
    BatchSubmissionError,
    CustomIdMismatchError,
    LlmClientConfigError,
    LlmClientError,
)
from bba.llm_client.models import (
    ALLOWED_MODELS,
    ANTHROPIC_BETA_HEADER,
    MAX_SONNET_ATTEMPTS,
    OPUS_MODEL_ID,
    SONNET_MODEL_ID,
    AnthropicTransport,
    BatchSubmissionRequest,
    BatchSubmissionResult,
    CassetteInteraction,
    Classification,
    DisagreementVerdict,
    EscalationLog,
    IndicationCitation,
    LlmClassificationResponse,
    LlmClientConfig,
    LlmClientResult,
    Model,
    ParseFailureReason,
    ParseOutcome,
    RawBatchResponse,
    StructuredToolInput,
)
from bba.llm_client.parser import parse_structured_response

__all__ = [
    "ALLOWED_MODELS",
    "ANTHROPIC_BETA_HEADER",
    "AnthropicAPIError",
    "AnthropicTransport",
    "BatchSubmissionError",
    "BatchSubmissionRequest",
    "BatchSubmissionResult",
    "CassetteInteraction",
    "CassetteTransport",
    "Classification",
    "CustomIdMismatchError",
    "DisagreementVerdict",
    "EscalationLog",
    "IndicationCitation",
    "LlmClassificationResponse",
    "LlmClientConfig",
    "LlmClientConfigError",
    "LlmClientError",
    "LlmClientResult",
    "MAX_SONNET_ATTEMPTS",
    "Model",
    "OPUS_MODEL_ID",
    "ParseFailureReason",
    "ParseOutcome",
    "RawBatchResponse",
    "SONNET_MODEL_ID",
    "StructuredToolInput",
    "assert_custom_ids_match",
    "detect_disagreement",
    "escalate_to_opus",
    "load_cassette",
    "parse_structured_response",
    "process_batch",
    "run_with_escalation",
    "should_escalate",
    "submit_batch",
]
