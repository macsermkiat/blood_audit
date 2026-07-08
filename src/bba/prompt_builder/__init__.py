"""bba.prompt_builder — system + few-shot + per-row payload assembly.

See issue #21 for acceptance criteria. PRD §12 (Implementation Decisions)
defines the assembly contract:

* System prompt + few-shot block + per-row user payload, with a
  ``task_mode`` switch (``HB_7_10_REVIEW`` / ``HB_GT_10_OVERRIDE``).
* ``cohort_threshold`` injection as a hard numeric input (PRD §"Cohort
  detection is deterministic, not LLM-judged").
* ``<evidence id="E1" untrusted="true">…</evidence>`` envelope for
  every redacted evidence chunk (PRD §38).
* Pre-LLM injection scanner (>=20 adversarial patterns covering
  imperative verbs, fake guidelines, bilingual jailbreaks); flagged
  rows route to ``NEEDS_REVIEW`` without an LLM call.
* Anthropic prompt-cache markers on the system + few-shot blocks; the
  trailing per-row user payload is never cacheable.

This module is a *pure function* (no I/O, zero deps on the Anthropic
SDK). The LLM client (#22) consumes :class:`PromptBuildResult` and
translates its blocks into the Anthropic Messages-API shape at that
boundary. Mirrors the Protocol pattern from :mod:`bba.quote_grounder`
and :mod:`bba.deid_redactor`.
"""

from bba.prompt_builder.builder import build_prompt
from bba.prompt_builder.canonical import (
    build_envelope,
    canonical_serialize,
    compute_prompt_hash,
)
from bba.prompt_builder.envelope import (
    wrap_evidence,
    wrap_evidence_chunks,
)
from bba.prompt_builder.exceptions import (
    HashMismatchError,
    PromptBuilderError,
    UnknownTaskModeError,
    UnsupportedCohortThresholdError,
)
from bba.prompt_builder.few_shot import build_few_shot_block
from bba.prompt_builder.injection import (
    INJECTION_PATTERNS,
    InjectionPattern,
    scan_chunks,
    scan_injection,
)
from bba.prompt_builder.models import (
    ALLOWED_COHORT_THRESHOLDS,
    EVIDENCE_TAG_CLOSE,
    EVIDENCE_TAG_OPEN_TEMPLATE,
    MIN_REQUIRED_INJECTION_PATTERNS,
    TASK_MODES,
    CohortThreshold,
    EvidenceChunk,
    FewShotExample,
    InjectionCategory,
    InjectionMatch,
    InjectionVerdict,
    NeedsReviewReason,
    PromptBlock,
    PromptBuildRequest,
    PromptBuildResult,
    PromptRole,
    TaskMode,
)
from bba.prompt_builder.system_prompt import platelet_system_prompt, system_prompt_for

__all__ = [
    "ALLOWED_COHORT_THRESHOLDS",
    "CohortThreshold",
    "EVIDENCE_TAG_CLOSE",
    "EVIDENCE_TAG_OPEN_TEMPLATE",
    "EvidenceChunk",
    "FewShotExample",
    "HashMismatchError",
    "INJECTION_PATTERNS",
    "InjectionCategory",
    "InjectionMatch",
    "InjectionPattern",
    "InjectionVerdict",
    "MIN_REQUIRED_INJECTION_PATTERNS",
    "NeedsReviewReason",
    "PromptBlock",
    "PromptBuildRequest",
    "PromptBuildResult",
    "PromptBuilderError",
    "PromptRole",
    "TASK_MODES",
    "TaskMode",
    "UnknownTaskModeError",
    "UnsupportedCohortThresholdError",
    "build_envelope",
    "build_few_shot_block",
    "build_prompt",
    "canonical_serialize",
    "compute_prompt_hash",
    "platelet_system_prompt",
    "scan_chunks",
    "scan_injection",
    "system_prompt_for",
    "wrap_evidence",
    "wrap_evidence_chunks",
]
