"""Pydantic v2 models, enums, and type aliases for :mod:`bba.prompt_builder`.

The module surface mirrors the convention established by
:mod:`bba.evidence_bundle_builder`, :mod:`bba.deid_redactor`, and
:mod:`bba.quote_grounder`: all public models are frozen; the assembly is a
pure function; the canonical envelope underwrites a SHA-256 ``prompt_hash``
the audit row persists (PRD §"Output schema" — ``prompt_hash`` reproducible).

Task modes correspond directly to the LLM-eligible audit branches in
the PRD's deterministic-engine output:

* ``HB_7_10_REVIEW`` — gray-zone case (Hb 7-10 g/dL, or Hb < cohort_threshold
  for the patient's cohort). The LLM reads the ±24-h note window for Tier-1
  indications + Tier-2 supportive context.
* ``HB_GT_10_OVERRIDE`` — Hb > 10 case escalated for the LLM to look for a
  Tier-1 override from the fixed hard-indication vocabulary (ACTIVE_BLEEDING,
  HEMODYNAMIC_INSTABILITY, ACS, PERIOPERATIVE, MTP) that would justify
  ordering despite the deterministic ``POTENTIALLY_INAPPROPRIATE``
  pre-classification. Under the clear-cut policy (#92) the prompt treats
  soft/prose grounds — a specialist or neuro Hb target, symptomatic-anemia
  prose — as non-clearing and instructs the model to return INAPPROPRIATE when
  no hard override is documented. This entry describes the prompt contract
  only; structural enforcement of that verdict — extending the over-clear
  guardrail to the Hb>=10 path and adding the quantified/life-threatening
  bleed exception — lands in #94 and is not yet wired.
* ``PLATELET_REVIEW`` — platelet transfusion order, reviewed against the
  Chula DRAFT policy (AABB/ICTMG 2025). Has no Hb cohort threshold;
  ``cohort_threshold`` is None for this mode.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# =============================================================================
# Task-mode + cohort-threshold contract
# =============================================================================


TaskMode = Literal["HB_7_10_REVIEW", "HB_GT_10_OVERRIDE", "PLATELET_REVIEW"]
"""The LLM-eligible audit branches.

Mirrors the deterministic-engine outputs that route into the LLM stage
(PRD §12 — "task_mode switch ``HB_7_10_REVIEW`` / ``HB_GT_10_OVERRIDE``").
``PLATELET_REVIEW`` is the Phase-2 platelet branch (Stage C1 prompt +
response contract; Stage C2 pipeline wiring).
"""


TASK_MODES: frozenset[str] = frozenset(
    {"HB_7_10_REVIEW", "HB_GT_10_OVERRIDE", "PLATELET_REVIEW"}
)
"""Runtime-introspectable copy of :data:`TaskMode` for defensive checks
outside Pydantic's literal validation (e.g. the
:func:`bba.prompt_builder.system_prompt.system_prompt_for` selector)."""


ALLOWED_COHORT_THRESHOLDS: frozenset[float] = frozenset({7.0, 7.5, 8.0})
"""The three deterministic cohort thresholds the LLM receives as a hard input.

PRD §"Cohort detection is deterministic, not LLM-judged": ``7.0`` is the
default and ESRD-on-EPO floor, ``7.5`` is the cardiac-surgery cohort, and
``8.0`` is the orthopedic-surgery / cardiopulmonary-comorbidity floor.
Free-form floats here would let a caller silently push a non-clinical
threshold past the assembly layer into the LLM prompt.
"""


def _ensure_cohort_threshold(value: float) -> float:
    """Reject values outside :data:`ALLOWED_COHORT_THRESHOLDS`.

    Free-form floats would silently leak non-clinical thresholds into the
    LLM's hard-input slot. The audit chain depends on the threshold being
    one of three deterministic clinical decisions, not a typo or a
    calibration mistake.
    """
    if value not in ALLOWED_COHORT_THRESHOLDS:
        allowed = sorted(ALLOWED_COHORT_THRESHOLDS)
        raise ValueError(
            f"cohort_threshold must be one of {allowed} g/dL (got {value!r}); "
            "deviation breaks the deterministic-engine contract "
            "(PRD §'Cohort detection is deterministic, not LLM-judged')"
        )
    return value


CohortThreshold = Annotated[float, AfterValidator(_ensure_cohort_threshold)]


# =============================================================================
# Injection-scanner vocabulary
# =============================================================================


class InjectionCategory(StrEnum):
    """Mutually-exclusive category tag for a detected injection pattern.

    Surfaced on :class:`InjectionMatch.category` so the audit row can group
    matches for monitoring — e.g. "Thai jailbreak rate spike on ward 5".
    The categories map 1:1 to the ticket-scope items: imperative verbs,
    fake-guideline patterns, bilingual Thai/EN jailbreaks, plus
    system-prompt-extraction attempts (a closely related family).
    """

    IMPERATIVE_VERB_EN = "imperative_verb_en"
    IMPERATIVE_VERB_TH = "imperative_verb_th"
    FAKE_GUIDELINE = "fake_guideline"
    JAILBREAK_EN = "jailbreak_en"
    JAILBREAK_TH = "jailbreak_th"
    SYSTEM_PROMPT_EXFIL = "system_prompt_exfil"
    ROLE_PRETEND = "role_pretend"
    ENVELOPE_ESCAPE = "envelope_escape"


class InjectionMatch(BaseModel):
    """One detected injection-pattern hit inside an evidence chunk.

    ``category`` groups the hit for monitoring; ``pattern_id`` is the
    stable identifier of the specific regex (e.g. ``"ignore_prior_v1"``)
    so a corpus drift can be diagnosed by counting hits per pattern_id
    over time. ``span_text`` is the matched substring (post-NFC) — kept
    short because the upstream redactor already ran; the span is only the
    suspicious fragment, not the full note.
    """

    model_config = ConfigDict(frozen=True)

    category: InjectionCategory
    pattern_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    span_text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        if self.end <= self.start:
            raise ValueError(
                f"InjectionMatch.end ({self.end}) must be > start ({self.start})"
            )
        return self


class InjectionVerdict(BaseModel):
    """Aggregated injection-scanner outcome over all evidence chunks.

    ``flagged`` is the routing signal — when ``True`` the audit pipeline
    must short-circuit to ``NEEDS_REVIEW`` without an LLM call (PRD §38).
    ``matches`` is the full list of hits across all chunks so reviewers
    can see exactly which patterns fired without re-running the scanner.
    """

    model_config = ConfigDict(frozen=True)

    flagged: bool
    matches: tuple[InjectionMatch, ...] = ()

    @model_validator(mode="after")
    def _flag_matches_matches(self) -> Self:
        # Internal consistency: ``flagged`` must be ``True`` iff ``matches``
        # is non-empty. The two fields exist together (rather than computing
        # ``flagged`` from ``matches`` on the fly) so callers can
        # short-circuit on the boolean without unpacking the tuple — but
        # a desync between the two would silently change routing.
        if self.flagged != bool(self.matches):
            raise ValueError(
                "InjectionVerdict.flagged must be True iff matches is "
                f"non-empty (flagged={self.flagged}, n_matches={len(self.matches)})"
            )
        return self


# =============================================================================
# Inputs
# =============================================================================


class EvidenceChunk(BaseModel):
    """One redacted evidence item ready for ``<evidence>`` wrapping.

    ``evidence_id`` is the stable bundle ID (``E1``, ``E2``, ...) assigned
    by :mod:`bba.evidence_bundle_builder` and preserved through
    :mod:`bba.deid_redactor`. ``source`` echoes the bundle's source family
    (Diagnosis / IPDADMPROGRESS / IPDNRFOCUSDT / MED / Lab / Vitals) for
    surface-level grouping in the user payload. ``text`` is the post-
    redaction text — the prompt builder treats it as opaque and does NOT
    re-redact.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(min_length=1, pattern=r"^E\d+$")
    source: str = Field(min_length=1)
    text: str

    @field_validator("text")
    @classmethod
    def _text_must_be_non_blank(cls, v: str) -> str:
        # A whitespace-only chunk renders to ``<evidence id="..."
        # untrusted="true"></evidence>`` after wrapping — there is
        # nothing for the LLM to cite and nothing for
        # :mod:`bba.quote_grounder` to ground against. If upstream has
        # no real content for a slot, omit the chunk entirely; the
        # builder routes zero-chunks to ``EMPTY_EVIDENCE`` /
        # NEEDS_REVIEW. Codex review #21 round 4 P2.
        if not v.strip():
            raise ValueError(
                "EvidenceChunk.text must contain non-whitespace content "
                "(omit the chunk if upstream has no content for this slot; "
                "the builder routes zero-chunks to NEEDS_REVIEW)"
            )
        return v


class FewShotExample(BaseModel):
    """One vetted (input, expected output) example for the few-shot block.

    PRD §"Stack" — clinical team provides 3-5 few-shot examples for
    clinical sign-off; these are committee-approved and ship inside the
    cacheable few-shot block (Anthropic prompt-cache marker boundary).
    ``name`` is a stable identifier (e.g. ``"benign_hb_8.2_active_bleed"``)
    so a corpus update can target a single example by name. ``user_payload``
    is the example evidence content (already in the canonical envelope
    form). ``assistant_output`` is the expected LLM response (typically
    a JSON object — kept as a string here to avoid coupling the prompt-
    builder to the LLM's structured-output shape, which lives in #22).
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    user_payload: str = Field(min_length=1)
    assistant_output: str = Field(min_length=1)


_RBC_TASK_MODES: frozenset[str] = frozenset({"HB_7_10_REVIEW", "HB_GT_10_OVERRIDE"})
"""The two RBC audit branches that require a ``cohort_threshold``."""


class PromptBuildRequest(BaseModel):
    """Top-level input to :func:`bba.prompt_builder.build_prompt`.

    ``task_mode`` selects the system-prompt template (PRD §12). The
    ``cohort_threshold`` lands as a hard numeric input in the system
    prompt for RBC modes — never inferred by the LLM. For
    ``PLATELET_REVIEW``, ``cohort_threshold`` must be ``None`` (platelet
    transfusion has no Hb cohort threshold). ``evidence_chunks`` is the
    per-row payload (the redacted bundle's items, in canonical emission
    order). ``few_shot_examples`` is the vetted exemplar block — empty
    tuple is acceptable for early-pilot builds before the clinical-team
    sign-off lands.
    """

    model_config = ConfigDict(frozen=True)

    task_mode: TaskMode
    cohort_threshold: CohortThreshold | None = None
    evidence_chunks: tuple[EvidenceChunk, ...]
    few_shot_examples: tuple[FewShotExample, ...] = ()

    @model_validator(mode="after")
    def _cohort_threshold_mode_consistent(self) -> Self:
        # RBC modes require a validated cohort threshold; PLATELET_REVIEW
        # must NOT carry one (platelet policy has no Hb cohort). Enforcing
        # here keeps the downstream system-prompt selector simple: it can
        # trust cohort_threshold is non-None for RBC and None for platelet.
        if self.task_mode in _RBC_TASK_MODES:
            if self.cohort_threshold is None:
                raise ValueError(
                    f"cohort_threshold is required for task_mode "
                    f"{self.task_mode!r} (must be one of "
                    f"{sorted(ALLOWED_COHORT_THRESHOLDS)} g/dL)"
                )
        elif self.task_mode == "PLATELET_REVIEW":
            if self.cohort_threshold is not None:
                raise ValueError(
                    "cohort_threshold must be None for PLATELET_REVIEW "
                    "(platelet transfusion has no Hb cohort threshold; "
                    "pass cohort_threshold=None or omit the argument)"
                )
        return self

    @model_validator(mode="after")
    def _evidence_ids_unique(self) -> Self:
        # Downstream grounding (:mod:`bba.quote_grounder`) treats a
        # non-unique ``cited_id`` as ``CITED_ID_NOT_FOUND`` — duplicate
        # evidence IDs in a single prompt would silently fail all
        # citations against that ID, even legitimate ones. The builder
        # emits one ``<evidence id="...">`` envelope per chunk, so the
        # contract must be enforced at request boundary (codex review
        # #21 round 2 P2-b).
        ids = [c.evidence_id for c in self.evidence_chunks]
        if len(ids) != len(set(ids)):
            from collections import Counter

            duplicates = sorted(eid for eid, n in Counter(ids).items() if n > 1)
            raise ValueError(
                f"evidence_chunks must have unique evidence_id values; "
                f"duplicates: {duplicates}"
            )
        return self


# =============================================================================
# Output
# =============================================================================


class NeedsReviewReason(StrEnum):
    """Mutually-exclusive routing tags surfaced on :class:`PromptBuildResult`.

    OR-of-reasons: multiple reasons may co-fire and all are persisted on
    :attr:`PromptBuildResult.needs_review_reasons` so the audit pipeline
    (#24) reads them straight into the row-level ``review_reason`` field.
    """

    INJECTION_DETECTED = "injection_detected"
    EMPTY_EVIDENCE = "empty_evidence"


PromptRole = Literal["system", "user"]
"""The two prompt-block roles the prompt builder emits.

The builder NEVER emits ``assistant`` blocks — few-shot examples are
embedded inside the leading ``user`` block as a single text envelope so
the cache marker can land cleanly on the cacheable boundary. The LLM
client (#22) translates the user block into Anthropic API ``messages``
turns at its boundary if needed.
"""


class PromptBlock(BaseModel):
    """One segment of the assembled prompt.

    ``role`` is ``"system"`` or ``"user"``. ``text`` is the segment body
    (NFC-normalized at construction). ``cache_marker`` is ``True`` when
    this block is the LAST cacheable block — Anthropic's prompt-cache
    contract marks the cache breakpoint at the END of a marked block, so
    only one cache_marker per logical region.

    The builder emits at most two cache markers: one at the end of the
    system prompt and one at the end of the few-shot block (or, when
    no few-shot examples are supplied, only at the end of the system
    prompt). The trailing per-row user payload is NEVER cacheable
    (changes per audit row).
    """

    model_config = ConfigDict(frozen=True)

    role: PromptRole
    text: str = Field(min_length=1)
    cache_marker: bool


class PromptBuildResult(BaseModel):
    """Top-level output of :func:`bba.prompt_builder.build_prompt`.

    Carries the assembled prompt segments and every metadata field the
    audit row needs: the task mode and cohort threshold that produced the
    prompt, the injection-scanner verdict, the NEEDS_REVIEW routing, and
    the canonical-bytes hash for prompt-hash stability.

    The ``prompt_hash`` is computed over a canonical-JSON envelope of the
    result. Same input + same task_mode + same cohort_threshold + same
    redacted evidence + same few-shot block -> same canonical bytes ->
    same hash (issue #21 AC: prompt-cache marker correctness verified +
    audit-chain reproducibility). The model validator recomputes the
    hash at construction and rejects mismatches — mirrors the
    :class:`bba.evidence_bundle_builder.EvidenceBundle` and
    :class:`bba.deid_redactor.RedactionResult` audit-chain invariants.
    """

    model_config = ConfigDict(frozen=True)

    blocks: tuple[PromptBlock, ...]
    task_mode: TaskMode
    cohort_threshold: CohortThreshold | None
    injection_verdict: InjectionVerdict
    route_to_needs_review: bool
    needs_review_reasons: tuple[NeedsReviewReason, ...]
    prompt_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _hash_must_match_envelope(self) -> Self:
        # Lazy import to avoid a circular import: canonical depends on
        # nothing in models at import time, but importing canonical at
        # module load would create a top-level cycle if canonical ever
        # learned about models (mirrors the deid_redactor / bundle
        # builder pattern).
        from bba.prompt_builder.canonical import (
            build_envelope,
            compute_prompt_hash,
        )

        if not all(c in "0123456789abcdef" for c in self.prompt_hash):
            raise ValueError(
                f"prompt_hash must be lowercase hex (got {self.prompt_hash!r})"
            )

        # The route_to_needs_review boolean must be the OR of the reasons —
        # any drift between the two would silently change LLM-call routing
        # in the audit pipeline.
        expected_route = bool(self.needs_review_reasons)
        if self.route_to_needs_review != expected_route:
            raise ValueError(
                f"route_to_needs_review ({self.route_to_needs_review}) must be "
                f"True iff needs_review_reasons is non-empty "
                f"(n_reasons={len(self.needs_review_reasons)})"
            )

        # The scanner verdict is the mandatory short-circuit (PRD §38):
        # ``flagged`` is the routing signal, ``INJECTION_DETECTED`` is the
        # persisted reason. Allowing one without the other would let a
        # caller reconstruct an inconsistent result whose hash is
        # self-consistent but whose routing diverges from what the audit
        # pipeline reads (codex review #21 round 2 P2-a).
        injection_in_reasons = (
            NeedsReviewReason.INJECTION_DETECTED in self.needs_review_reasons
        )
        if self.injection_verdict.flagged != injection_in_reasons:
            raise ValueError(
                "injection_verdict.flagged must agree with "
                "INJECTION_DETECTED membership in needs_review_reasons "
                f"(flagged={self.injection_verdict.flagged}, "
                f"in_reasons={injection_in_reasons})"
            )

        envelope = build_envelope(
            blocks=[
                {"role": b.role, "text": b.text, "cache_marker": b.cache_marker}
                for b in self.blocks
            ],
            task_mode=self.task_mode,
            cohort_threshold=self.cohort_threshold,
            injection_matches=[
                {
                    "category": m.category.value,
                    "pattern_id": m.pattern_id,
                    "evidence_id": m.evidence_id,
                    "span_text": m.span_text,
                    "start": m.start,
                    "end": m.end,
                }
                for m in self.injection_verdict.matches
            ],
            route_to_needs_review=self.route_to_needs_review,
            needs_review_reasons=[r.value for r in self.needs_review_reasons],
        )
        expected = compute_prompt_hash(envelope)
        if self.prompt_hash != expected:
            raise ValueError(
                f"prompt_hash ({self.prompt_hash}) does not match "
                f"sha256(canonical envelope) ({expected}); construct via "
                "build_prompt() to maintain the audit-chain invariant"
            )
        return self


# =============================================================================
# Module-level constants
# =============================================================================


EVIDENCE_TAG_OPEN_TEMPLATE: str = '<evidence id="{evidence_id}" untrusted="true">'
"""Opening tag for :func:`bba.prompt_builder.envelope.wrap_evidence`.

PRD §38: "every evidence chunk wrapped in
``<evidence id="E1" untrusted="true">…</evidence>``". The
``untrusted="true"`` attribute is a fixed string — there is no
trusted-evidence case in Phase 1 (all evidence has crossed the redactor
boundary and is treated as adversarial to the LLM's role)."""


EVIDENCE_TAG_CLOSE: str = "</evidence>"
"""Closing tag for :func:`bba.prompt_builder.envelope.wrap_evidence`."""


MIN_REQUIRED_INJECTION_PATTERNS: int = 20
"""Issue #21 AC: ``Injection scanner test set: ≥20 known adversarial patterns``.

The scanner ships ``≥20`` patterns covering imperative verbs (EN + TH),
fake guidelines, bilingual jailbreaks, and system-prompt exfiltration.
The threshold is asserted as a structural invariant in the test suite —
dropping below 20 patterns is a contract violation regardless of any
single test's outcome."""


__all__: Sequence[str] = (
    "ALLOWED_COHORT_THRESHOLDS",
    "CohortThreshold",
    "EVIDENCE_TAG_CLOSE",
    "EVIDENCE_TAG_OPEN_TEMPLATE",
    "EvidenceChunk",
    "FewShotExample",
    "InjectionCategory",
    "InjectionMatch",
    "InjectionVerdict",
    "MIN_REQUIRED_INJECTION_PATTERNS",
    "NeedsReviewReason",
    "PromptBlock",
    "PromptBuildRequest",
    "PromptBuildResult",
    "PromptRole",
    "TASK_MODES",
    "TaskMode",
)
