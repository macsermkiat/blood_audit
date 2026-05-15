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

Recording is deliberately out of scope here: the cassettes are
hand-authored in ``tests/fixtures/llm_client/`` and the production
recording path is a separate concern (issue #24's audit_pipeline
integration test).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from bba.llm_client.models import (
    AnthropicTransport,
    BatchSubmissionRequest,
    CassetteInteraction,
    RawBatchResponse,
)


def load_cassette(path: Path) -> tuple[CassetteInteraction, ...]:
    """Load a JSON cassette from ``path``.

    The file shape is::

        {
          "interactions": [
            {
              "model": "claude-sonnet-4-6-20251018",
              "custom_ids": ["audit-001", "audit-002"],
              "response": { ... RawBatchResponse JSON ... }
            },
            ...
          ]
        }

    Raises :class:`FileNotFoundError` on missing path, or
    :class:`pydantic.ValidationError` on malformed shape — both
    surfacing loudly so a typo in fixture authoring doesn't silently
    cause cassette misses at test run time.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #22")


class CassetteTransport:
    """An :class:`AnthropicTransport` backed by a recorded cassette.

    The cassette is loaded once at construction; subsequent
    :meth:`submit_batch` calls look up the response by
    ``(model, sorted_tuple(audit_ids))``. A miss raises
    :class:`KeyError` so the test fails loudly.

    Class-level docstring deliberately not promising thread safety:
    Phase 1 runs the batch loop single-threaded, mirroring the
    DuckDB single-writer contention rule (PRD §10).
    """

    def __init__(self, interactions: Sequence[CassetteInteraction]) -> None:
        raise NotImplementedError("RED-phase scaffold; see issue #22")

    def submit_batch(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        """Look up the cassette by ``(model, sorted_custom_ids)``.

        ``prompt_cache_enabled`` is accepted but does not participate
        in the lookup — cassettes are model + request-set keyed; cache
        marker correctness is exercised at the request-build layer, not
        the replay layer.
        """
        raise NotImplementedError("RED-phase scaffold; see issue #22")


__all__: Sequence[str] = ("CassetteTransport", "load_cassette")


# Static check: CassetteTransport satisfies the AnthropicTransport
# protocol. This isn't strictly necessary (Protocol matching is
# structural), but the explicit assertion catches signature drift in CI.
_PROTOCOL_CHECK: type[AnthropicTransport] = CassetteTransport  # type: ignore[assignment]
