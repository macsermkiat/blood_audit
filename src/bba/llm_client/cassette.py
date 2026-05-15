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
    """

    def __init__(self, interactions: Sequence[CassetteInteraction]) -> None:
        self._table: dict[tuple[str, tuple[str, ...]], RawBatchResponse] = {}
        for interaction in interactions:
            key = (interaction.model, tuple(sorted(interaction.custom_ids)))
            if key in self._table:
                raise LlmClientError(
                    f"duplicate cassette interaction for key {key!r}"
                )
            self._table[key] = interaction.response

    def submit_batch(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        """Look up the cassette by ``(model, sorted_custom_ids)``."""
        _ = prompt_cache_enabled  # cassettes do not key on cache mode
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
