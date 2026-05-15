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

from bba.llm_client.models import BatchSubmissionResult, ParseOutcome


def parse_structured_response(result: BatchSubmissionResult) -> ParseOutcome:
    """Parse the tool-use JSON payload out of one Batch result row.

    Expected Anthropic response shape (tool-use mode)::

        result.raw_response_json["content"] is a list. The structured
        output is the first ``tool_use`` block; its ``input`` field is
        the parsed JSON object matching
        :class:`LlmClassificationResponse`.

    Failure-to-reason mapping:

    * No ``content`` array, or empty array → ``EMPTY_RESPONSE``.
    * No ``tool_use`` block in ``content`` →  ``TOOL_USE_MISSING``.
    * ``tool_use.input`` is not a dict / missing required keys →
      ``SCHEMA_MISMATCH``.
    * ``classification`` not in the four allowed labels →
      ``CLASSIFICATION_OUT_OF_SET``.
    * ``tool_use.input`` is a string that fails ``json.loads`` →
      ``MALFORMED_JSON`` (legacy SDK shape passthrough).
    """
    raise NotImplementedError("RED-phase scaffold; see issue #22")
