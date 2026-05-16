"""Cost guard: refuse the live Anthropic transport in tests.

User constraint #10: "The live Anthropic API SHOULD NOT be called
during ralph-loop iteration. If you see live API calls in test output,
that's a bug — the cassette setup from #22 is being bypassed. Fix the
cassette before proceeding."

The guard is a single function call near the pipeline entrypoint. It
inspects the transport's class identity (not duck-typed structural
match — we want to detect the production class specifically) and
raises :class:`bba.audit_pipeline.LiveAnthropicApiError` if the live
transport is detected. Tests opt in by invoking the guard before
:func:`bba.audit_pipeline.run_pipeline`; production code skips it.
"""

from __future__ import annotations

from bba.audit_pipeline.exceptions import LiveAnthropicApiError
from bba.llm_client import AnthropicBatchTransport, AnthropicTransport


def assert_test_safe_transport(transport: AnthropicTransport) -> None:
    """Raise :class:`LiveAnthropicApiError` if ``transport`` is the live
    Anthropic Batch API client.

    The check is class-identity-based, not duck-typed: a wrapper that
    inherits from :class:`bba.llm_client.AnthropicBatchTransport` is
    still considered live. The cassette transport
    (:class:`bba.llm_client.CassetteTransport`) passes through silently.
    """
    if isinstance(transport, AnthropicBatchTransport):
        raise LiveAnthropicApiError(
            f"live Anthropic transport {type(transport).__name__!r} detected "
            "in a test context; tests MUST inject CassetteTransport. "
            "If you see this in test output, the cassette setup from "
            "issue #22 is being bypassed — fix the cassette, not the guard."
        )


__all__ = ["assert_test_safe_transport"]
