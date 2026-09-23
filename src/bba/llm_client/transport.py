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
from importlib import import_module
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

MAX_OUTPUT_TOKENS: Final[int] = 8192
"""Max tokens reserved for the LLM's tool-call output.

Raised from 4096 (2026-09-21): 3 of 107 sandbox answers stopped at the old
limit, and because every schema asks for ``classification`` LAST a cut-off
answer carries no label at all (schema_mismatch / NEEDS_REVIEW).

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
                "input_schema": (
                    _PLATELET_TOOL_INPUT_SCHEMA
                    if request.task_mode == "PLATELET_REVIEW"
                    else _RESERVE_AHEAD_TOOL_INPUT_SCHEMA
                    if request.task_mode == "RESERVE_AHEAD_REVIEW"
                    else _TOOL_INPUT_SCHEMA
                ),
            },
        ],
        "tool_choice": {"type": "tool", "name": _TOOL_NAME},
    }


# Field order is load-bearing in every schema below (issues #239, #242): a
# forced tool call is generated in schema order, so ``classification`` comes
# LAST, after the reasoning and every other judgment. With the label first the
# model committed to it before reasoning: 22 of 297 real-data platelet answers,
# and RBC answers 68023033 / 68042514 / 68055542, carried APPROPRIATE while
# their own reasoning concluded INAPPROPRIATE or NEEDS_REVIEW. The model can
# still ignore property order, so the field states the rule in words too.
_LABEL_FIELD: Final[str] = "classification"


def _label_property(written_after: str) -> dict[str, Any]:
    return {
        "type": "string",
        "enum": [
            "APPROPRIATE",
            "INAPPROPRIATE",
            "NEEDS_REVIEW",
            "INSUFFICIENT_EVIDENCE",
        ],
        "description": (
            f"Write this field LAST, after {written_after}. It must be the "
            "class that reasoning_summary_en concludes: if your reasoning ends "
            "at INAPPROPRIATE or NEEDS_REVIEW, this field says the same."
        ),
    }


def _without_label(names: Any) -> list[str]:
    return [name for name in names if name != _LABEL_FIELD]


_TOOL_INPUT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
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
        "reasoning_summary_en": {
            "type": "string",
            "description": (
                "Reasoning summary in English ONLY. Do not include the "
                "Thai summary or any markup tags in this field; the Thai "
                "summary belongs in reasoning_summary_th."
            ),
        },
        "reasoning_summary_th": {
            "type": "string",
            "description": (
                "Reasoning summary in natural, fluent clinical Thai, as a "
                "Thai transfusion-committee reviewer would write it - NOT "
                "a word-for-word translation of the English summary. Keep "
                "standard clinical terms (Hb, ACS, MTP, peri-operative, "
                "gray-zone) in English, as Thai clinicians do."
            ),
        },
        _LABEL_FIELD: _label_property("both reasoning summaries"),
    },
    "required": [
        "indications",
        "negative_evidence",
        "reasoning_summary_en",
        "reasoning_summary_th",
        _LABEL_FIELD,
    ],
}

_SIGNAL_ORDER_RULE: Final[str] = (
    " Write this boolean after both reasoning summaries; it must agree with "
    "what reasoning_summary_en concludes about this indication."
)
"""Appended to every platelet hard-signal description. Once the label was
pinned last (#239) the model moved the four booleans to the FRONT and set
prophylactic_marrow_failure true before reasoning that the count was above the
threshold (68042732, 68051598): the same commit-before-reasoning defect."""

# Platelet-specific extension: adds the three hard-signal booleans that
# parse_platelet_structured_response requires. Without these fields the
# model never emits them, every platelet response fails SCHEMA_MISMATCH,
# and the live platelet leg can produce no real verdicts.
# Used ONLY for PLATELET_REVIEW requests. The hard signals sit between the
# reasoning and the label (see the field-order note above).
_PLATELET_TOOL_INPUT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        **{
            name: _TOOL_INPUT_SCHEMA["properties"][name]
            for name in _without_label(_TOOL_INPUT_SCHEMA["properties"])
        },
        "active_bleeding": {
            "type": "boolean",
            "description": (
                (
                    "True iff the evidence explicitly grounds documented active, "
                    "life-threatening, or clinically significant bleeding AND no "
                    "exclusion population applies. Set False for bare low count alone."
                )
                + _SIGNAL_ORDER_RULE
            ),
        },
        "procedure_indication": {
            "type": "boolean",
            "description": (
                (
                    "True iff the evidence grounds an invasive procedure or surgery "
                    "within the audit window whose policy threshold the count sits "
                    "below (LP <50-80k /uL; CVC, thoracocentesis, arthrocentesis, "
                    "dental extraction <50k /uL; major surgery <80-100k /uL)."
                )
                + _SIGNAL_ORDER_RULE
            ),
        },
        "prophylactic_marrow_failure": {
            "type": "boolean",
            "description": (
                (
                    "True iff the evidence grounds chemo/HSCT/consumptive "
                    "thrombocytopenia with count <10,000 /uL (or expected <10,000 "
                    "/uL within 24 hours) AND no exclusion population applies. "
                    "Aplastic anaemia on active therapy belongs under "
                    "aplastic_active_therapy_indication, not here."
                )
                + _SIGNAL_ORDER_RULE
            ),
        },
        "intracranial_bleed_indication": {
            "type": "boolean",
            "description": (
                (
                    "True iff the evidence grounds an intracranial bleed with the "
                    "count below its threshold: <100,000 /uL when acute, "
                    "acute-on-chronic, growing, or neurosurgery is planned; <50,000 "
                    "/uL when stable and non-operative (including a stable chronic "
                    "SDH). Chronic SDH expansion belongs here, not under "
                    "active_bleeding."
                )
                + _SIGNAL_ORDER_RULE
            ),
        },
        "aplastic_active_therapy_indication": {
            "type": "boolean",
            "description": (
                (
                    "True iff the notes document aplastic anaemia on active therapy "
                    "aimed at reversing the thrombocytopenia (ATG, cyclosporine, "
                    "eltrombopag, transplant work-up) with the count below its "
                    "indication 8 threshold: <10,000 /uL (or expected <10,000 /uL "
                    "within 24 hours), or <20,000 /uL during the ATG course or with "
                    "sepsis. Chronic, stable or untreated aplastic anaemia is an "
                    "exclusion population, not this signal."
                )
                + _SIGNAL_ORDER_RULE
            ),
        },
        _LABEL_FIELD: _label_property(
            "both reasoning summaries and the four hard-signal booleans"
        ),
    },
    "required": [
        *_without_label(_TOOL_INPUT_SCHEMA["required"]),
        "active_bleeding",
        "procedure_indication",
        "prophylactic_marrow_failure",
        "intracranial_bleed_indication",
        "aplastic_active_therapy_indication",
        _LABEL_FIELD,
    ],
}


_RESERVE_AHEAD_TOOL_INPUT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        **{
            name: _TOOL_INPUT_SCHEMA["properties"][name]
            for name in _without_label(_TOOL_INPUT_SCHEMA["properties"])
        },
        "administration_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "source_id": {"type": "string"},
                    "marker_type": {
                        "type": "string",
                        "enum": [
                            "gave_blood",
                            "component_given",
                            "unit_numbers",
                            "intraop_transfusion",
                            "post_transfusion_check",
                        ],
                    },
                },
                "required": ["quote", "source_id", "marker_type"],
            },
        },
        "administration_claimed": {"type": "boolean"},
        "reservation_assessment": {
            "type": "string",
            "enum": ["APPROPRIATE", "INAPPROPRIATE", "INSUFFICIENT_EVIDENCE"],
        },
        _LABEL_FIELD: _label_property(
            "both reasoning summaries, the administration fields and "
            "reservation_assessment"
        ),
    },
    "required": [
        *_without_label(_TOOL_INPUT_SCHEMA["required"]),
        "administration_evidence",
        "administration_claimed",
        "reservation_assessment",
        _LABEL_FIELD,
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
            anthropic = import_module("anthropic")
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
