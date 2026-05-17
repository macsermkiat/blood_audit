"""Production :class:`AnthropicTransport` — Anthropic Message Batches API.

PRD §13 "Anthropic Batch API integration". This module is the only
place in :mod:`bba.llm_client` that imports the live Anthropic SDK; the
rest of the package speaks to the :class:`AnthropicTransport` Protocol
so unit tests can substitute :class:`CassetteTransport` and run offline.

The transport's job:

1. Translate :class:`BatchSubmissionRequest` into the Anthropic
   Messages API request shape, honouring
   :attr:`bba.prompt_builder.PromptBlock.cache_marker` by emitting
   ``cache_control={"type": "ephemeral"}`` on the corresponding block.
2. Submit + poll the Batch endpoint, persisting the
   ``anthropic-version`` header and ``prompt_cache_id`` on every
   result for the audit chain.
3. Surface response headers verbatim so :class:`BatchSubmissionResult`
   carries them into ``llm_calls`` persistence.

The SDK import is deferred to method bodies (lazy) so importing
``bba.llm_client`` in offline / CI contexts does not require the
``anthropic`` extra. A caller that instantiates
:class:`AnthropicBatchTransport` without the SDK installed receives a
clean :class:`LlmClientConfigError` instead of an obscure
``ImportError``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final

from bba.llm_client.exceptions import (
    AnthropicAPIError,
    LlmClientConfigError,
)
from bba.llm_client.models import (
    ANTHROPIC_BETA_HEADER,
    AnthropicTransport,
    BatchSubmissionRequest,
    BatchSubmissionResult,
    RawBatchResponse,
)


_TOOL_NAME: Final[str] = "classify_transfusion_order"
_TOOL_DESCRIPTION: Final[str] = (
    "Return the structured RBC transfusion audit classification for "
    "the supplied evidence. Mandatory tool; no free-form text answers."
)

MAX_OUTPUT_TOKENS: Final[int] = 4096
"""Max tokens reserved for the LLM's tool-call output.

Sized for the structured-output envelope (classification + up to ~5
indications with verbatim quotes + reasoning summaries in EN + TH).
The audit pipeline never emits free-form responses; the tool-use shape
caps verbosity structurally regardless of this limit, but Anthropic
requires an explicit ``max_tokens`` on every request."""


def build_anthropic_request(
    request: BatchSubmissionRequest,
    *,
    model: str,
    prompt_cache_enabled: bool,
) -> dict[str, Any]:
    """Translate one :class:`BatchSubmissionRequest` to Anthropic's
    Messages-API request payload.

    Emits one ``cache_control={"type": "ephemeral"}`` marker per
    :class:`bba.prompt_builder.PromptBlock` with ``cache_marker=True``,
    on the LAST content element of that block. Anthropic's contract
    treats the marker as the cache breakpoint at the END of the marked
    content; placing it on every cached element would burn breakpoints
    unnecessarily (the SDK caps the count at 4 per request).
    """
    system_blocks: list[dict[str, Any]] = []
    user_message_blocks: list[dict[str, Any]] = []

    for prompt_block in request.prompt.blocks:
        block_payload: dict[str, Any] = {
            "type": "text",
            "text": prompt_block.text,
        }
        if prompt_cache_enabled and prompt_block.cache_marker:
            block_payload["cache_control"] = {"type": "ephemeral"}

        if prompt_block.role == "system":
            system_blocks.append(block_payload)
        else:
            user_message_blocks.append(block_payload)

    return {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": system_blocks,
        "messages": [
            {"role": "user", "content": user_message_blocks},
        ],
        "tools": [
            {
                "name": _TOOL_NAME,
                "description": _TOOL_DESCRIPTION,
                "input_schema": _TOOL_INPUT_SCHEMA,
            },
        ],
        "tool_choice": {"type": "tool", "name": _TOOL_NAME},
    }


_TOOL_INPUT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": [
                "APPROPRIATE",
                "INAPPROPRIATE",
                "NEEDS_REVIEW",
                "INSUFFICIENT_EVIDENCE",
            ],
        },
        "indications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "quote": {"type": "string"},
                    "source_id": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["code", "quote", "source_id", "confidence"],
            },
        },
        "negative_evidence": {"type": "array", "items": {"type": "string"}},
        "reasoning_summary_en": {"type": "string"},
        "reasoning_summary_th": {"type": "string"},
    },
    "required": [
        "classification",
        "indications",
        "negative_evidence",
        "reasoning_summary_en",
        "reasoning_summary_th",
    ],
}


class AnthropicBatchTransport:
    """Production :class:`AnthropicTransport` backed by the official
    ``anthropic`` SDK.

    Construction validates the ANTHROPIC_API_KEY environment variable
    is set (or supplied to the constructor) — a missing key raises
    :class:`LlmClientConfigError` at instantiation time, not deep
    inside the SDK on first call.

    Poll interval defaults to 30 seconds; Anthropic recommends
    "minutes, not seconds" but the audit pipeline drives this transport
    from a single-threaded loop so 30 s minimises latency without
    burning the rate budget.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        poll_interval_seconds: float = 30.0,
        max_wait_seconds: float = 86_400.0,
    ) -> None:
        resolved_key = (
            api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        )
        if not resolved_key:
            raise LlmClientConfigError(
                "ANTHROPIC_API_KEY not configured; set the env var or "
                "supply api_key= to AnthropicBatchTransport(...)"
            )
        self._api_key = resolved_key
        self._poll_interval = poll_interval_seconds
        self._max_wait = max_wait_seconds

    def submit_batch_only(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> str:
        """Create the remote batch and return its ``batch_id`` immediately.

        Does NOT poll. Callers that need to persist the ``batch_id``
        before waiting for results (the audit pipeline's row-level
        checkpoint) call this first, persist, then invoke
        :meth:`fetch_batch_results`.
        """
        client = self._client()
        per_row_requests = [
            {
                "custom_id": req.audit_id,
                "params": build_anthropic_request(
                    req,
                    model=model,
                    prompt_cache_enabled=prompt_cache_enabled,
                ),
            }
            for req in requests
        ]
        try:
            batch = client.messages.batches.create(requests=per_row_requests)
        except Exception as exc:  # SDK error surface intentionally broad
            raise AnthropicAPIError(f"Message Batches create failed: {exc!r}") from exc
        batch_id: str = batch.id
        return batch_id

    def fetch_batch_results(
        self,
        batch_id: str,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        """Poll ``batch_id`` until completion and return the results.

        Raises :class:`AnthropicAPIError` on timeout or
        ``custom_id`` drift. ``requests`` is the original submission
        set; each result's ``request_json`` is rebuilt from the
        matching :class:`BatchSubmissionRequest`."""
        client = self._client()
        deadline = time.monotonic() + self._max_wait
        while True:
            status = client.messages.batches.retrieve(batch_id)
            if status.processing_status == "ended":
                break
            if time.monotonic() > deadline:
                raise AnthropicAPIError(
                    f"batch {batch_id!r} did not complete within {self._max_wait}s"
                )
            time.sleep(self._poll_interval)

        results: list[BatchSubmissionResult] = []
        request_by_id = {req.audit_id: req for req in requests}
        for entry in client.messages.batches.results(batch_id):
            custom_id = entry.custom_id
            outer_request = request_by_id.get(custom_id)
            if outer_request is None:
                raise AnthropicAPIError(
                    f"Batch result custom_id={custom_id!r} not in submitted set"
                )
            results.append(
                _result_from_batch_entry(
                    entry,
                    model=model,
                    request_payload=build_anthropic_request(
                        outer_request,
                        model=model,
                        prompt_cache_enabled=prompt_cache_enabled,
                    ),
                )
            )
        return RawBatchResponse(batch_id=batch_id, results=tuple(results))

    def submit_batch(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        """Convenience wrapper: submit + poll.

        Preserved for backward compatibility with callers that do not
        need split-phase checkpointing. The audit_pipeline orchestrator
        calls :meth:`submit_batch_only` and :meth:`fetch_batch_results`
        directly so it can persist the batch_id between create and
        poll (PRD §15 row-level checkpoint).
        """
        batch_id = self.submit_batch_only(
            model=model,
            requests=requests,
            prompt_cache_enabled=prompt_cache_enabled,
        )
        return self.fetch_batch_results(
            batch_id,
            model=model,
            requests=requests,
            prompt_cache_enabled=prompt_cache_enabled,
        )

    def _client(self) -> Any:
        """Construct the SDK client. Lazy-imported so the SDK isn't a
        hard install dependency."""
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise LlmClientConfigError(
                "anthropic SDK not installed; "
                "`uv add anthropic` to use AnthropicBatchTransport"
            ) from exc
        return anthropic.Anthropic(
            api_key=self._api_key,
            default_headers={"anthropic-beta": ANTHROPIC_BETA_HEADER},
        )


_SUCCEEDED_RESULT_TYPE: Final[str] = "succeeded"


def _result_from_batch_entry(
    entry: Any,
    *,
    model: str,
    request_payload: dict[str, Any],
) -> BatchSubmissionResult:
    """Translate one SDK batch-result entry into our frozen model.

    Anthropic Batch API result entries carry a ``type`` discriminator:
    ``succeeded``, ``errored``, ``canceled``, or ``expired``. Only
    ``succeeded`` entries expose ``result.message``; the other three
    surface ``result.error`` and would raise ``AttributeError`` if we
    blindly read ``message``. For every non-succeeded type we return a
    structured error envelope (raw_response_json carries the error
    detail) so the row still reaches the audit chain and routes to
    NEEDS_REVIEW via the parser's ``EMPTY_RESPONSE``/``TOOL_USE_MISSING``
    paths — never silently dropped.
    """
    result_type = getattr(entry.result, "type", None)
    headers = {
        "anthropic-version": getattr(entry, "anthropic_version", "2023-06-01"),
    }
    response_dict: dict[str, Any]
    prompt_cache_id: str | None = None

    if result_type == _SUCCEEDED_RESULT_TYPE:
        message = entry.result.message
        response_dict = (
            message.model_dump() if hasattr(message, "model_dump") else dict(message)
        )
        usage = (
            response_dict.get("usage", {}) if isinstance(response_dict, dict) else {}
        )
        if isinstance(usage, dict) and usage.get("cache_read_input_tokens"):
            prompt_cache_id = "cache-hit"
    else:
        error_detail = getattr(entry.result, "error", None)
        error_dict: dict[str, Any]
        if error_detail is None:
            error_dict = {"message": "no error detail returned by Anthropic"}
        elif hasattr(error_detail, "model_dump"):
            error_dict = error_detail.model_dump()
        else:
            error_dict = (
                dict(error_detail)
                if isinstance(error_detail, dict)
                else {"detail": str(error_detail)}
            )
        # Synthesize a response envelope that parses as EMPTY_RESPONSE
        # (no content key) — the audit row routes to NEEDS_REVIEW with
        # the structured error preserved for the reviewer to inspect.
        response_dict = {
            "_batch_result_type": result_type or "unknown",
            "_batch_error": error_dict,
            "id": getattr(entry, "id", ""),
            "type": "message",
            "role": "assistant",
            "content": [],
            "stop_reason": "batch_error",
        }

    return BatchSubmissionResult(
        custom_id=entry.custom_id,
        model_id=model,
        raw_response_json=response_dict,
        request_json=request_payload,
        response_headers=headers,
        request_timestamp=datetime.now(UTC),
        latency_ms=0,  # Anthropic Batch API does not expose per-row latency
        anthropic_version=headers["anthropic-version"],
        prompt_cache_id=prompt_cache_id,
        extended_thinking_blocks=None,
    )


__all__: Sequence[str] = (
    "AnthropicBatchTransport",
    "build_anthropic_request",
)


# Static check: AnthropicBatchTransport satisfies the Protocol.
_PROTOCOL_CHECK: type[AnthropicTransport] = AnthropicBatchTransport
