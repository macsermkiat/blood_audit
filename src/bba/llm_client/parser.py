"""Fail-closed structured-output parser.

PRD §13: "structured-output (tool-use) JSON shape with fail-closed
parsing (malformed → NEEDS_REVIEW with parse_failure flag)".

The parser returns :class:`ParseOutcome` for every input. It NEVER
raises on bad LLM output — a raise would lose the audit-chain trail
because the caller's persistence layer would never see the failure.
Instead, every failure mode lands as ``parse_failure=True`` with a
:class:`ParseFailureReason` so the row routes to ``NEEDS_REVIEW``
with the structured reason persisted on the audit row.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from bba.llm_client.models import (
    BatchSubmissionResult,
    LlmClassificationResponse,
    ParseFailureReason,
    ParseOutcome,
    _ALLOWED_CLASSIFICATIONS,
)


def parse_structured_response(result: BatchSubmissionResult) -> ParseOutcome:
    """Parse the tool-use JSON payload out of one Batch result row.

    Fails closed: every failure mode lands as ``parse_failure=True``
    with a structured :class:`ParseFailureReason`. The parser never
    raises.
    """
    raw_text_fallback = ""

    try:
        content = _extract_content(result.raw_response_json)
    except _ParseGuard as guard:
        return _failure(guard.reason, raw_text_fallback)

    tool_block = _find_tool_use(content)
    if tool_block is None:
        return _failure(
            ParseFailureReason.TOOL_USE_MISSING,
            _stringify_content(content),
        )

    tool_input = tool_block.get("input")
    parsed_dict, malformed = _coerce_to_dict(tool_input)
    if malformed:
        return _failure(
            ParseFailureReason.MALFORMED_JSON,
            _stringify(tool_input),
        )
    if parsed_dict is None:
        return _failure(
            ParseFailureReason.SCHEMA_MISMATCH,
            _stringify(tool_input),
        )

    classification = parsed_dict.get("classification")
    if isinstance(classification, str) and classification not in _ALLOWED_CLASSIFICATIONS:
        return _failure(
            ParseFailureReason.CLASSIFICATION_OUT_OF_SET,
            _stringify(parsed_dict),
        )

    try:
        response = LlmClassificationResponse.model_validate(parsed_dict)
    except ValidationError:
        return _failure(
            ParseFailureReason.SCHEMA_MISMATCH,
            _stringify(parsed_dict),
        )

    return ParseOutcome(
        parsed=response,
        parse_failure=False,
        parse_failure_reason=None,
        raw_text=_stringify(parsed_dict),
    )


class _ParseGuard(Exception):
    """Internal control flow: a structural shape check failed."""

    def __init__(self, reason: ParseFailureReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


def _extract_content(raw: Any) -> Sequence[Any]:
    """Pull the ``content`` array out of an Anthropic response envelope.

    The envelope may arrive as a plain ``dict`` (live API) or as a
    deeply-frozen :class:`MappingProxyType` (after construction via
    :class:`FrozenJsonDict`); both are :class:`Mapping`. The content
    array similarly arrives as ``list`` or ``tuple``; both are
    :class:`Sequence` (but ``str``/``bytes`` are explicitly excluded —
    those are scalar payloads, not arrays).
    """
    if not isinstance(raw, Mapping):
        raise _ParseGuard(ParseFailureReason.EMPTY_RESPONSE)
    content = raw.get("content")
    if content is None:
        raise _ParseGuard(ParseFailureReason.EMPTY_RESPONSE)
    if isinstance(content, str | bytes) or not isinstance(content, Sequence):
        raise _ParseGuard(ParseFailureReason.SCHEMA_MISMATCH)
    if len(content) == 0:
        raise _ParseGuard(ParseFailureReason.EMPTY_RESPONSE)
    return content


def _find_tool_use(content: Sequence[Any]) -> Mapping[str, Any] | None:
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "tool_use":
            return block
    return None


def _coerce_to_dict(value: Any) -> tuple[Mapping[str, Any] | None, bool]:
    """Return (dict_or_none, malformed_json_flag).

    A string ``value`` is treated as a JSON-encoded payload (legacy SDK
    shape passthrough). Anything else is either a mapping or a schema
    mismatch.
    """
    if isinstance(value, Mapping):
        return value, False
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None, True
        if isinstance(decoded, dict):
            return decoded, False
        return None, False
    return None, False


def _failure(reason: ParseFailureReason, raw_text: str) -> ParseOutcome:
    return ParseOutcome(
        parsed=None,
        parse_failure=True,
        parse_failure_reason=reason,
        raw_text=raw_text,
    )


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    except (TypeError, ValueError):
        return repr(value)


def _stringify_content(content: Sequence[Any]) -> str:
    return _stringify(content)


def _json_default(value: Any) -> Any:
    """Coerce frozen / non-JSON-serializable values to JSON for logging.

    :class:`MappingProxyType` and ``tuple`` are JSON-serializable via
    ``dict(...)`` and ``list(...)`` respectively. Everything else
    falls back to ``str()``.
    """
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str | bytes):
        return value
    if isinstance(value, Sequence):
        return list(value)
    return str(value)
