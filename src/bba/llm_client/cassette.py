"""Betamax/VCR-style JSON cassette transport for offline replay.

PRD §"contract tests against the Anthropic SDK using betamax-style
cassettes for offline replay" + issue #22 AC: "Betamax/VCR cassettes
for offline replay of Anthropic API".

The cassette format is a small subset of VCR's: a JSON file with one
``interactions`` array. Each interaction has a request key
(``model`` + sorted ``custom_ids``) and a response payload that
matches :class:`RawBatchResponse`. The :class:`CassetteTransport`
implements :class:`AnthropicTransport` and looks up the recorded
response on each :meth:`submit_batch` call — a miss raises so the
test fails loudly instead of silently substituting a default.

Recording is deliberately out of scope here: cassettes are hand-
authored in ``tests/fixtures/llm_client/`` and the production
recording path is a separate concern (issue #24's audit_pipeline
integration test).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from bba.llm_client.exceptions import LlmClientError
from bba.llm_client.models import (
    AnthropicTransport,
    BatchSubmissionRequest,
    CassetteInteraction,
    RawBatchResponse,
)


def load_cassette(path: Path) -> tuple[CassetteInteraction, ...]:
    """Load a JSON cassette from ``path``.

    Raises :class:`FileNotFoundError` for a missing path or
    :class:`pydantic.ValidationError` for a malformed shape — both
    surfacing loudly so a typo in fixture authoring doesn't silently
    cause cassette misses at test run time.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    interactions = payload.get("interactions")
    if not isinstance(interactions, list):
        raise LlmClientError(
            f"cassette at {path} missing 'interactions' list "
            f"(got {type(interactions).__name__})"
        )
    return tuple(
        CassetteInteraction.model_validate(entry) for entry in interactions
    )


class CassetteTransport:
    """An :class:`AnthropicTransport` backed by a recorded cassette.

    The cassette is indexed at construction by
    ``(model, sorted_tuple(custom_ids))``; subsequent
    :meth:`submit_batch` calls look up the response by the same key.
    A miss raises :class:`KeyError` so the test fails loudly.

    Phase 1 runs the batch loop single-threaded (PRD §10 DuckDB
    single-writer rule), so the cassette is not promised thread-safe.

    The transport implements the split-phase contract
    (:meth:`submit_batch_only` + :meth:`fetch_batch_results`) so the
    audit_pipeline's row-level checkpoint can exercise the same code
    path the production transport uses. ``submit_batch_only`` records
    the planned ``batch_id`` keyed on the interaction's
    ``RawBatchResponse.batch_id``; ``fetch_batch_results`` retrieves
    by either ``batch_id`` (the resume path) or the
    ``(model, sorted_custom_ids)`` key (the submit-then-fetch path).
    """

    def __init__(self, interactions: Sequence[CassetteInteraction]) -> None:
        self._table: dict[tuple[str, tuple[str, ...]], RawBatchResponse] = {}
        self._by_batch_id: dict[str, RawBatchResponse] = {}
        for interaction in interactions:
            key = (interaction.model, tuple(sorted(interaction.custom_ids)))
            if key in self._table:
                raise LlmClientError(
                    f"duplicate cassette interaction for key {key!r}"
                )
            self._table[key] = interaction.response
            self._by_batch_id[interaction.response.batch_id] = interaction.response

    def submit_batch_only(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> str:
        """Return the recorded ``batch_id`` without polling.

        Same lookup key as :meth:`submit_batch`; the cassette must
        already carry an interaction for this submission. The recorded
        :class:`RawBatchResponse.batch_id` is returned so the caller
        can persist it before invoking :meth:`fetch_batch_results`.
        """
        _ = prompt_cache_enabled
        response = self._lookup_by_request_key(model, requests)
        return response.batch_id

    def fetch_batch_results(
        self,
        batch_id: str,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        """Return the recorded :class:`RawBatchResponse` for ``batch_id``.

        Falls back to the ``(model, sorted_custom_ids)`` lookup when the
        ``batch_id`` was not seen at construction time — this supports
        the resume path that polls a batch_id read off the
        ``batch_runs`` table without having to re-record the cassette
        per resume run."""
        _ = prompt_cache_enabled
        if batch_id in self._by_batch_id:
            return self._by_batch_id[batch_id]
        return self._lookup_by_request_key(model, requests)

    def submit_batch(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        """Backward-compatible: submit + fetch in one call."""
        _ = prompt_cache_enabled  # cassettes do not key on cache mode
        return self._lookup_by_request_key(model, requests)

    def _lookup_by_request_key(
        self, model: str, requests: Sequence[BatchSubmissionRequest]
    ) -> RawBatchResponse:
        key = (model, tuple(sorted(r.audit_id for r in requests)))
        if key not in self._table:
            raise KeyError(
                f"no recorded interaction for {key!r}; "
                f"known: {sorted(self._table)}"
            )
        return self._table[key]


__all__: Sequence[str] = ("CassetteTransport", "load_cassette")


# Static check: CassetteTransport satisfies the AnthropicTransport
# protocol. Catches signature drift in CI.
_PROTOCOL_CHECK: type[AnthropicTransport] = CassetteTransport
