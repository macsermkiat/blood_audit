"""Pydantic v2 models for the deterministic_classifier module.

All public types are immutable (``frozen=True``). The classifier is a pure
function from a :class:`ClassifierInputs` snapshot to a :class:`ClassifierResult`;
downstream :mod:`bba.audit_store` persists ``classification`` (canonical
:data:`bba.audit_store.Classification`) and ``bypass_reason`` (the structured
:class:`BypassReason` enum) verbatim.

Public-surface invariants (issue #8 acceptance criteria):

* ``ClassifierResult.classification`` MUST be one of the canonical
  :data:`bba.audit_store.Classification` Literal values. The deterministic
  layer NEVER emits ``"INAPPROPRIATE"`` — that label requires positive
  evidence reasoning only the LLM stage can do (Round 1 B2).

* ``ClassifierResult.bypass_reason`` is a :class:`BypassReason` member —
  never a free-form string. Each bypass path sets exactly one reason;
  non-bypass classifications carry :attr:`BypassReason.NONE`.

* ``ClassifierResult.cohort_threshold`` echoes the threshold actually
  applied at decision time (``None`` for non-threshold cohorts: MTP,
  HEME_MALIGNANCY_ACTIVE, UNKNOWN). This makes the audit trail
  reproducible without re-running the cohort detector.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from bba.audit_store import Classification
from bba.cohort_detector import CohortAssignment
from bba.declared_use import DeclaredUseLabel
from bba.hb_lookup import HbLookupResult


class BypassReason(StrEnum):
    """Why a deterministic bypass fired (or NONE if no bypass applied).

    Per PRD §"Implementation Decisions §6" and user constraint #5:
    bypass-reason is a structured enum, not a free string. Each bypass
    pathway in :func:`bba.deterministic_classifier.classify` sets exactly
    one of these values.

    * :attr:`DELTA_HB` — Hb dropped at-or-above the tiered threshold inside
      a recent window (PRD §3 + Round 2 E3).
    * :attr:`PERI_PROCEDURAL_6H` — an operative procedure was performed in
      the 6 h before the order anchor.
    * :attr:`PRE_OP_CROSSMATCH` — an operative procedure is scheduled soon
      after the order anchor, consistent with a pre-op crossmatch.
    * :attr:`PERIOP_EVIDENCE` — Hb is missing but a HARD peri-operative
      signal (an intra-op transfusion, or estimated blood loss at or above
      :data:`bba.deterministic_classifier.PERIOP_MIN_EBL_ML`) was recovered
      from the notes. Only fires on the missing-Hb positive-evidence
      pre-pass (gated behind ``enable_missing_hb_positive_evidence``);
      the documented surgical haemorrhage stands in for the absent Hb.
    * :attr:`MTP` — cohort detector flagged the massive-transfusion-protocol
      cluster (≥4 RBC units in 1 h, or RBC + FFP + platelets co-ordered).
    * :attr:`HEMODILUTION_FLAGGED` — after the global Hb < 7.0 rule, Hb is
      still sub-threshold but ≥2 L of crystalloid in the prior 4 h marks the
      reading as hemodilution-suspect; route to
      :data:`bba.audit_store.Classification` ``NEEDS_REVIEW`` rather than
      auto-APPROPRIATE (Round 1 B5).
    * :attr:`NONE` — no bypass fired; classification reflects the plain
      Hb-tier rule.
    """

    DELTA_HB = "delta_hb"
    PERI_PROCEDURAL_6H = "peri_procedural_6h"
    PRE_OP_CROSSMATCH = "pre_op_crossmatch"
    PERIOP_EVIDENCE = "periop_evidence"
    MTP = "mtp"
    HEMODILUTION_FLAGGED = "hemodilution_flagged"
    RETURNED_NOT_TRANSFUSED = "returned_not_transfused"
    PERIOP_TRANSFUSION_EXEMPT = "periop_transfusion_exempt"
    NONE = "none"


class ClassifierInputs(BaseModel):
    """Per-audit deterministic-classifier inputs.

    Composed at the orchestrator layer from the upstream module outputs:
    :class:`bba.hb_lookup.HbLookupResult`,
    :class:`bba.cohort_detector.CohortAssignment`, the procedure-proximity
    derived from IPTSUMOPRT joins, and the 4-h crystalloid total derived
    from :func:`bba.deterministic_classifier.crystalloid.total_crystalloid_liters`.

    Field semantics:

    * ``procedure_proximity_hours`` is the elapsed hours from the most
      recent operative event to the order anchor. ``None`` means "no
      operative event found in the lookback window" — distinct from
      ``0.0`` (procedure at the same instant). Procedures in the future
      relative to the order ARE ignored upstream (the orchestrator filters).

    * ``upcoming_procedure_hours`` is the elapsed hours from the order
      anchor to the next operative event. ``None`` means no future
      operative event was found in the caller's lookahead window.

    * ``crystalloid_liters_prior_4h`` is the 4-h totaling output. ``0.0``
      means "no crystalloid in window OR data unavailable but the caller
      asserts the 4-h check is N/A". The hemodilution rule only fires
      strictly ≥ :data:`bba.deterministic_classifier.HEMODILUTION_CRYSTALLOID_LITERS`.

    * ``audit_id`` is a SafeId-shaped opaque string; this module does not
      revalidate it (the orchestrator already passed it through
      :class:`bba.audit_store.AuditRow`).

    * ``enable_missing_hb_positive_evidence`` is the operator-supplied
      kill-switch for the missing-Hb positive-evidence pre-pass
      (MTP / peri-procedural / hard peri-op evidence auto-APPROPRIATE on
      no documented Hb, with everything else deferred to the LLM rather
      than dead-ended). Defaults to ``False`` because the policy is "SEED
      pending clinical sign-off" (see
      :mod:`bba.deterministic_classifier.classifier` and docs/CONTEXT.md
      §"Missing-Hb positive-evidence pre-check"). When ``False`` the
      classifier ignores every missing-Hb bypass branch and returns
      ``INSUFFICIENT_EVIDENCE`` for missing Hb, preserving the original
      PRD spec. When ``True`` the missing-Hb terminal verdict becomes
      ``NEEDS_REVIEW`` (rationale ``hb_missing_defer_llm``) so the router
      routes the case to the LLM instead of terminating. Set to ``True``
      per-row only after the QI committee has signed off.

    * ``periop_blood_loss_ml`` / ``periop_intraop_transfusion`` /
      ``periop_surgical_context`` are the HARD/soft peri-operative signals
      recovered from free-text notes by
      :func:`bba.vitals_extractor.periop.scan_periop` (mirroring the three
      fields of :class:`bba.vitals_extractor.PeriopSummary`). They feed the
      missing-Hb pre-pass ONLY:
        - ``periop_intraop_transfusion`` (a charted intra-op blood
          component) and ``periop_blood_loss_ml`` at or above
          :data:`bba.deterministic_classifier.PERIOP_MIN_EBL_ML` are the
          HARD signals that auto-classify ``APPROPRIATE`` on missing Hb.
        - ``periop_surgical_context`` is carried for traceability/audit
          only; "a surgery is documented" is the WEAK signal the design
          deliberately refuses to rubber-stamp, so it never gates a verdict
          (those cases defer to the LLM). ``None``/``False`` defaults leave
          the pre-pass inert, so a caller that never ran ``scan_periop``
          gets the unchanged Hb-present behaviour.

    * ``enable_declared_use_preop_exemption`` selects the order-level pre-op
      rule. When ``True``, declared ``surgery`` and ``type_screen`` orders are
      exempt from the Hb-appropriateness judgment regardless of ledger state;
      factual all-returned/incompatible disposition still has precedence.
      The peri-op note/proximity envelope is ignored. It defaults to ``False``
      so direct callers that do not opt in preserve the legacy transfused plus
      peri-op-envelope behavior byte-for-byte.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: str
    hb_result: HbLookupResult
    cohort_assignment: CohortAssignment
    order_datetime: AwareDatetime
    procedure_proximity_hours: float | None
    upcoming_procedure_hours: float | None = None
    crystalloid_liters_prior_4h: float = Field(ge=0.0)
    enable_missing_hb_positive_evidence: bool = False
    periop_blood_loss_ml: int | None = None
    periop_intraop_transfusion: bool = False
    periop_surgical_context: bool = False
    returns_disposition: Literal["not_transfused", "transfused", "inconclusive"] = (
        "inconclusive"
    )
    returns_periop_context: bool = False
    declared_use: DeclaredUseLabel | None = None
    enable_declared_use_preop_exemption: bool = False


class ClassifierResult(BaseModel):
    """Outcome of :func:`bba.deterministic_classifier.classify`.

    ``classification`` is the canonical :data:`bba.audit_store.Classification`
    Literal — re-imported here so the audit_store row can persist this
    field directly with no further interpretation.

    ``bypass_reason`` is :attr:`BypassReason.NONE` whenever the result was
    produced by the plain Hb-tier rule (Hb < threshold, threshold ≤ Hb <
    10, Hb ≥ 10, or Hb missing). Otherwise it identifies which of the five
    bypass pathways fired.

    ``cohort_threshold`` carries the threshold actually used; ``None``
    when the cohort is non-threshold-driven (MTP, HEME_MALIGNANCY_ACTIVE,
    UNKNOWN) so a reviewer can distinguish "threshold 7.0 default applied"
    from "no threshold meaningful for this cohort".

    ``rationale`` is a short slug identifying which rule fired
    (``"hb_lt_7_universal"``, ``"hb_lt_threshold"``, ``"hb_7_to_10"``,
    ``"hb_ge_10"``, ``"hb_missing"``, ``"hb_missing_defer_llm"``,
    ``"bypass_delta_hb"``, ``"bypass_peri_procedural"``,
    ``"bypass_pre_op_crossmatch"``, ``"preop_defer_llm"``,
    ``"preop_defer_llm_declared"``, ``"bypass_mtp"``,
    ``"bypass_mtp_hb_missing"``, ``"bypass_peri_procedural_hb_missing"``,
    ``"bypass_periop_evidence_hb_missing"``,
    ``"bypass_hemodilution"``, ``"cohort_unknown"``). The
    ``*_hb_missing`` slugs mark the missing-Hb positive-evidence pre-pass
    (approved with no documented Hb) so the QI committee can count them;
    ``hb_missing_defer_llm`` marks a missing-Hb case routed to the LLM
    (no hard evidence, flag on) rather than dead-ended.
    Free-form prose summaries are produced by the LLM stage, not here.
    """

    model_config = ConfigDict(frozen=True)

    classification: Classification
    bypass_reason: BypassReason
    cohort_threshold: float | None
    rationale: str


__all__: Sequence[str] = (
    "BypassReason",
    "ClassifierInputs",
    "ClassifierResult",
)
