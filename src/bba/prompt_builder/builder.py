"""Top-level :func:`build_prompt` orchestration.

Composes the per-step transforms into one pure function:

1. Run :func:`bba.prompt_builder.injection.scan_chunks` over every
   evidence chunk.
2. Build the system prompt (:func:`system_prompt_for`).
3. Build the few-shot block (:func:`build_few_shot_block`).
4. Wrap each evidence chunk in ``<evidence>`` envelopes
   (:func:`wrap_evidence_chunks`).
5. Assemble prompt blocks with cache markers (system end + few-shot end).
6. Compute the canonical envelope hash.
7. Decide routing: injection match -> ``INJECTION_DETECTED``; empty
   evidence -> ``EMPTY_EVIDENCE``; ``route_to_needs_review`` =
   ``bool(reasons)``.
8. Return the frozen :class:`PromptBuildResult`.

The function NEVER raises on a routing decision: those route via the
result. The audit pipeline (#24) branches on
:attr:`PromptBuildResult.route_to_needs_review`.
"""

from __future__ import annotations

from bba.prompt_builder.canonical import build_envelope, compute_prompt_hash
from bba.prompt_builder.envelope import wrap_evidence_chunks
from bba.prompt_builder.few_shot import build_few_shot_block
from bba.prompt_builder.injection import scan_chunks
from bba.prompt_builder.models import (
    InjectionVerdict,
    NeedsReviewReason,
    PromptBlock,
    PromptBuildRequest,
    PromptBuildResult,
)
from bba.prompt_builder.system_prompt import platelet_system_prompt, system_prompt_for


_EMPTY_USER_PAYLOAD: str = "<no_evidence/>"
"""Placeholder for the trailing user block when evidence is empty.

A :class:`PromptBlock` requires ``text`` of length >= 1, and the
:class:`PromptBuildResult` envelope is part of the prompt hash. Using a
fixed sentinel keeps the hash deterministic for the empty-evidence
case (which always routes to NEEDS_REVIEW anyway, so the LLM never
sees this payload).
"""


def build_prompt(request: PromptBuildRequest) -> PromptBuildResult:
    """Assemble the system + few-shot + user-payload prompt for ``request``."""
    injection_matches = scan_chunks(request.evidence_chunks)

    if request.task_mode == "PLATELET_REVIEW":
        # Platelet mode has no cohort threshold; use the dedicated function
        # so system_prompt_for is never called with cohort_threshold=None.
        system_text = platelet_system_prompt()
    else:
        # For RBC modes cohort_threshold is guaranteed non-None by the
        # PromptBuildRequest._cohort_threshold_mode_consistent validator.
        cohort_threshold = request.cohort_threshold
        assert cohort_threshold is not None, (
            "cohort_threshold must be set for RBC task modes; "
            "PromptBuildRequest validator should have caught this"
        )
        system_text = system_prompt_for(
            task_mode=request.task_mode,
            cohort_threshold=cohort_threshold,
        )
    few_shot_text = build_few_shot_block(request.few_shot_examples)
    user_payload = wrap_evidence_chunks(request.evidence_chunks)

    # Both system end and few-shot end are cache breakpoints — Anthropic's
    # prompt-cache contract supports up to 4 breakpoints; we use 2 when
    # few-shot is present so partial-prefix cache hits work (system
    # alone OR system + few-shot). The trailing user-payload block is
    # never cacheable: it changes per audit row.
    has_few_shot = bool(few_shot_text)
    blocks: list[PromptBlock] = [
        PromptBlock(role="system", text=system_text, cache_marker=True)
    ]
    if has_few_shot:
        blocks.append(PromptBlock(role="user", text=few_shot_text, cache_marker=True))
    blocks.append(
        PromptBlock(
            role="user",
            text=user_payload or _EMPTY_USER_PAYLOAD,
            cache_marker=False,
        )
    )

    reasons: list[NeedsReviewReason] = []
    if injection_matches.flagged:
        reasons.append(NeedsReviewReason.INJECTION_DETECTED)
    if not request.evidence_chunks:
        reasons.append(NeedsReviewReason.EMPTY_EVIDENCE)
    route = bool(reasons)

    envelope = build_envelope(
        blocks=[
            {"role": b.role, "text": b.text, "cache_marker": b.cache_marker}
            for b in blocks
        ],
        task_mode=request.task_mode,
        cohort_threshold=request.cohort_threshold,
        injection_matches=[
            {
                "category": m.category.value,
                "pattern_id": m.pattern_id,
                "evidence_id": m.evidence_id,
                "span_text": m.span_text,
                "start": m.start,
                "end": m.end,
            }
            for m in injection_matches.matches
        ],
        route_to_needs_review=route,
        needs_review_reasons=[r.value for r in reasons],
    )
    prompt_hash = compute_prompt_hash(envelope)

    return PromptBuildResult(
        blocks=tuple(blocks),
        task_mode=request.task_mode,
        cohort_threshold=request.cohort_threshold,
        injection_verdict=InjectionVerdict(
            flagged=injection_matches.flagged,
            matches=injection_matches.matches,
        ),
        route_to_needs_review=route,
        needs_review_reasons=tuple(reasons),
        prompt_hash=prompt_hash,
    )
