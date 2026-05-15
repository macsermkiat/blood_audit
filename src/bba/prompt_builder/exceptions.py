"""Custom exceptions for the :mod:`bba.prompt_builder` module.

The prompt builder is a *pure-function assembly layer* — it never calls the
LLM and never reads from disk. Failure modes are limited to:

* :class:`PromptBuilderError` — base class for any builder failure.
* :class:`UnknownTaskModeError` — caller passed a ``task_mode`` outside the
  declared :data:`TaskMode` literal. Caught at the model boundary by
  Pydantic validation, but raised explicitly by the system-prompt
  selector if a future code path bypasses that boundary.
* :class:`UnsupportedCohortThresholdError` — ``cohort_threshold`` outside
  the allowed ``{7.0, 7.5, 8.0}`` set. Mirrors :data:`UnknownTaskModeError`.
* :class:`HashMismatchError` — the recomputed
  :class:`PromptBuildResult.prompt_hash` does not match the canonical
  envelope; asserts the audit-chain replay invariant (mirrors
  :mod:`bba.evidence_bundle_builder` and :mod:`bba.deid_redactor`).

Routing decisions (injection-detected, empty-evidence) are NOT exceptions
— they are values on :class:`PromptBuildResult.needs_review_reasons` so
the audit pipeline can persist them. Quality-gate failures must travel
through the audit chain so reviewers can audit them, not raise mid-
pipeline and lose the bundle (mirrors the deid_redactor contract).
"""

from __future__ import annotations


class PromptBuilderError(Exception):
    """Base class for failures inside the prompt_builder assembly layer."""


class UnknownTaskModeError(PromptBuilderError):
    """``task_mode`` is outside the declared :data:`TaskMode` literal set.

    Normally caught by Pydantic at :class:`PromptBuildRequest` construction;
    raised explicitly by the system-prompt selector as defense in depth
    for future internal callers that bypass the model boundary.
    """


class UnsupportedCohortThresholdError(PromptBuilderError):
    """``cohort_threshold`` is outside the allowed ``{7.0, 7.5, 8.0}`` set.

    PRD §"Cohort detection is deterministic, not LLM-judged" — the LLM
    receives ``cohort_threshold`` as a hard input; the prompt builder
    refuses to inject anything outside the three deterministic values so a
    malformed value cannot leak past the assembly boundary as a free-form
    string.
    """


class HashMismatchError(PromptBuilderError):
    """The recomputed :class:`PromptBuildResult.prompt_hash` does not match.

    Asserted by the result model validator: same canonical-envelope bytes
    must yield the same SHA-256 hex. A mismatch means a downstream caller
    forged a hash or the canonical-JSON serializer drifted under a
    refactor — either way, the audit-chain replay invariant is broken
    (mirrors the :mod:`bba.evidence_bundle_builder` and
    :mod:`bba.deid_redactor` hash contracts).
    """
