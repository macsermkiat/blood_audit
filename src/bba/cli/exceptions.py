"""Typed errors for ``bba.cli``.

The CLI fails loud — every error here exits the process with a non-zero
status. Retry/backoff lives in :mod:`bba.llm_client` (issue #22), never in
the CLI layer.
"""

from __future__ import annotations


class CliError(Exception):
    """Base class for ``bba.cli`` errors.

    Subclasses signal a specific failure mode and are caught at the click
    top-level to map to exit codes / structured-log events; ``CliError``
    itself is never raised directly.
    """


class IdempotencyError(CliError):
    """An idempotency invariant was violated.

    Raised when the on-disk / store state contradicts what the CLI expects:
    e.g. a run marked complete but missing audit rows, or a partial run
    re-entered without ``--force``.
    """


class RunNotFoundError(CliError):
    """A ``--run-id`` was supplied but no such run exists in the store."""


class MutuallyExclusiveOptionError(CliError):
    """``bba audit`` was invoked with both ``--input`` and ``--run-id``.

    Click reports option-level conflicts via :class:`click.UsageError`;
    this subclass is for the post-parse semantic check that one (and only
    one) of the two is provided.
    """
