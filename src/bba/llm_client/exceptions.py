"""Custom exceptions for :mod:`bba.llm_client`.

The client distinguishes contract-violation errors (raised — never silently
swallowed) from routing decisions (returned on :class:`LlmClientResult`).
Routing decisions like "parse failure" or "Sonnet/Opus disagreement" are
not exceptions because the audit pipeline must persist them with the row
and route to ``NEEDS_REVIEW`` — mirrors the prompt_builder convention
(PRD §"fail-closed parsing").
"""

from __future__ import annotations


class LlmClientError(Exception):
    """Base class for any failure inside :mod:`bba.llm_client`.

    Concrete subclasses are raised; ``LlmClientError`` itself is the
    ``except`` target for callers that need to handle every failure
    family without distinguishing them.
    """


class CustomIdMismatchError(LlmClientError):
    """A Batch API result's ``custom_id`` does not match the submitted
    request, OR the result set is missing one or more submitted
    ``custom_id``\\s.

    PRD §13 makes this load-bearing: "``custom_id == audit_id`` assertion
    on every result — never positional zip". A positional zip silently
    swaps audit rows under partial-failure or out-of-order delivery; the
    explicit assertion makes the failure mode loud.

    The error message names the offending ``custom_id``\\s so the
    operator can quarantine the batch without re-running with extra
    logging.
    """


class AnthropicAPIError(LlmClientError):
    """The Anthropic Batch API surfaced a non-recoverable error.

    Recoverable errors (timeouts, rate limits) are handled by the retry
    layer inside :func:`bba.llm_client.client.submit_batch` and never
    bubble up as :class:`AnthropicAPIError`. This class is reserved for
    structural failures — auth, malformed request shape, missing model
    — that no amount of retrying will fix.
    """


class BatchSubmissionError(LlmClientError):
    """The batch could not be submitted at all (pre-flight failure).

    Includes: empty request list, duplicate ``custom_id``\\s in the
    submission, request payload exceeding Anthropic's per-batch byte
    cap. Raised before any HTTP call so a rejected submission leaves no
    half-state on Anthropic's side.
    """


class LlmClientConfigError(LlmClientError):
    """The :class:`LlmClientConfig` is malformed.

    Examples: model_id not in :data:`ALLOWED_MODELS` (snapshot pin
    violated), negative ``max_sonnet_attempts``, missing API-key
    environment variable when ``transport`` is the production
    Anthropic transport.
    """
