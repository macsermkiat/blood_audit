"""Pydantic v2 models for the platelet_classifier module.

All public types are immutable (``frozen=True``). The classifier is a pure
function from a :class:`PlateletClassifierInputs` snapshot to a
:class:`PlateletClassifierResult`, mirroring
:mod:`bba.deterministic_classifier` for the RBC path.

Public-surface invariants (docs plan §5.1, §8/CR-C1, CR-M2):

* ``PlateletClassifierResult.classification`` is one of the canonical
  :data:`bba.audit_store.Classification` values.

* With :data:`bba.feature_flags.PLATELET_PROPHYLAXIS_AUTOCLEAR_ENABLED` off
  (the default) the platelet classifier NEVER emits ``"APPROPRIATE"`` or
  ``"INAPPROPRIATE"``. With it on, the single ``APPROPRIATE`` path is the
  cohort-gated prophylaxis clear (fresh count < 10, heme/chemo diagnosis, no
  withhold population); ``"INAPPROPRIATE"`` is still never emitted. Otherwise, in v1, Platelet appropriateness is context-dependent (indication, procedure,
  and 6+ exclusion populations such as TTP/HIT/ITP/dengue where transfusion is
  withheld or harmful) and cannot be safely auto-cleared or auto-condemned by
  the deterministic layer. Everything with a count routes onward
  (POTENTIALLY_INAPPROPRIATE / NEEDS_REVIEW → LLM/review), so no present-count
  verdict is ever deterministic-final. See
  :data:`bba.audit_pipeline.pipeline._DETERMINISTIC_FINAL_CLASSIFICATIONS`.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from bba.audit_store import Classification


class PlateletClassifierInputs(BaseModel):
    """Per-audit platelet-classifier inputs.

    * ``platelet_count`` is the most-recent pre-anchor count in ×10³/µL
      (LABEXM 290078), or ``None`` when no valid count was found — distinct
      from a real low count. The count is produced upstream by the platelet
      lookup + :func:`bba.platelet_lookup.parse_platelet_count`.

    * ``enable_missing_platelet_defer`` mirrors the RBC
      ``enable_missing_hb_positive_evidence`` opt-in (defaults ``False``):
      when a count is missing, ``False`` yields a terminal
      ``INSUFFICIENT_EVIDENCE`` documentation gap (original-spec behaviour),
      while ``True`` defers the missing-count case to the LLM
      (``NEEDS_REVIEW``) rather than dead-ending. A SEED pending clinical
      sign-off; set per-row only after QI committee approval.

    * ``diagnosis_codes`` are the admission's ICD-10 codes (dotted or dotless,
      any case), ``platelet_freshness`` the lookup's freshness bucket and
      ``has_recent_chemo_med`` whether a chemotherapy drug was prescribed
      within the cohort detector's lookback (lets MDS qualify). All three
      are consulted ONLY by the cohort-gated prophylaxis auto-clear
      (``enable_prophylaxis_autoclear``, default ``False``); with the flag off
      they are inert and the gate is byte-identical to v1.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: str
    platelet_count: float | None
    enable_missing_platelet_defer: bool = False
    diagnosis_codes: tuple[str, ...] = ()
    platelet_freshness: str | None = None
    has_recent_chemo_med: bool = False
    enable_prophylaxis_autoclear: bool = False


class PlateletClassifierResult(BaseModel):
    """Outcome of :func:`bba.platelet_classifier.classify_platelet`.

    ``classification`` is the canonical :data:`bba.audit_store.Classification`.
    ``review_ceiling`` echoes the count ceiling actually applied (×10³/µL) so
    the audit trail is reproducible without re-reading the policy; ``None`` on
    the missing-count paths where no ceiling was evaluated.

    ``rationale`` is a short slug: ``"plt_ge_100"`` (at/above the review
    ceiling), ``"plt_defer_llm"`` (below the ceiling → LLM), ``"plt_missing"``
    (no count, defer flag off → terminal documentation gap), or
    ``"plt_missing_defer_llm"`` (no count, defer flag on → routed to LLM).
    There is deliberately no bare ``plt_lt_10`` slug: the unconditional
    very-low-count auto-clear was removed as a patient-safety defect
    (§8/CR-C1). The flag-gated ``"plt_lt_10_heme_prophylaxis"`` clear requires
    a heme/chemo diagnosis AND the absence of every withhold population.
    """

    model_config = ConfigDict(frozen=True)

    classification: Classification
    review_ceiling: float | None
    rationale: str


__all__: Sequence[str] = (
    "PlateletClassifierInputs",
    "PlateletClassifierResult",
)
