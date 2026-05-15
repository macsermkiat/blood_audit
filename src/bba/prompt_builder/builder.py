"""Top-level :func:`build_prompt` orchestration.

Composes the per-step transforms into one pure function:

1. Run :func:`bba.prompt_builder.injection.scan_chunks` over every
   evidence chunk. Collect the verdict.
2. Build the system prompt for ``request.task_mode`` /
   ``request.cohort_threshold`` (:func:`system_prompt_for`).
3. Build the few-shot block (:func:`build_few_shot_block`).
4. Wrap each evidence chunk in ``<evidence>`` envelopes
   (:func:`wrap_evidence_chunks`).
5. Assemble the prompt blocks:
   * system block, cache_marker = ``True`` iff no few-shot block follows
   * few-shot block (if any), cache_marker = ``True``
   * user payload block, cache_marker = ``False``
6. Compute the canonical envelope hash via
   :func:`bba.prompt_builder.canonical.compute_prompt_hash`.
7. Decide routing:
   * injection match -> ``INJECTION_DETECTED``
   * empty evidence -> ``EMPTY_EVIDENCE``
   * route_to_needs_review = ``bool(reasons)``
8. Return the frozen :class:`PromptBuildResult`.

The function NEVER raises on a routing decision (injection / empty
evidence): those route via the result. It DOES raise on contract
violations — :class:`UnknownTaskModeError` if the system-prompt
selector receives an unknown mode, :class:`UnsupportedCohortThresholdError`
if the cohort threshold is out of range. Both should be caught by the
:class:`PromptBuildRequest` validator before reaching the orchestrator;
they are defense-in-depth raises for internal callers.

RED-phase scaffold: :func:`build_prompt` raises
:class:`NotImplementedError`.
"""

from __future__ import annotations

from bba.prompt_builder.models import (
    PromptBuildRequest,
    PromptBuildResult,
)


def build_prompt(request: PromptBuildRequest) -> PromptBuildResult:
    """Assemble the system + few-shot + user-payload prompt for ``request``."""
    raise NotImplementedError("RED-phase scaffold; see issue #21")
