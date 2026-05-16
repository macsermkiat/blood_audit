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

from bba.audit_pipeline.exceptions import LiveAnthropicApiError  # noqa: F401  used by GREEN
from bba.llm_client import AnthropicBatchTransport, AnthropicTransport  # noqa: F401  used by GREEN


def assert_test_safe_transport(transport: AnthropicTransport) -> None:
    """Raise :class:`LiveAnthropicApiError` if ``transport`` is the live
    Anthropic Batch API client.

    The check is class-identity-based, not duck-typed: a wrapper that
    inherits from :class:`bba.llm_client.AnthropicBatchTransport` is
    still considered live. The cassette transport
    (:class:`bba.llm_client.CassetteTransport`) passes through silently.

    The implementation lives in GREEN (issue #24).
    """
    _ = transport
    raise NotImplementedError("RED-phase scaffold; see issue #24")


__all__ = ["assert_test_safe_transport"]
