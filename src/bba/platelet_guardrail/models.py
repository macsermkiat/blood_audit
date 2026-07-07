"""Structured platelet hard-signals for the over-clear guardrail.

Phase 2 counterpart to the RBC B1 guardrail's hard-signal set
(:func:`bba.audit_pipeline.replay._has_structured_hard_signal`), which is
entirely Hb-shaped (Hb<7, MTP, peri-op, SBP/HR). Per the user ruling
(2026-07-08, "ADD hard signals"), the platelet LLM leg MAY finalize
``APPROPRIATE`` below the review ceiling only when one of these positive
indications is grounded from the notes; absent any, the over-clear guardrail
floors the clear to human review.

Derived from the Chula DRAFT platelet policy (AABB/ICTMG 2025):

* ``active_bleeding`` — documented active / life-threatening / clinically
  significant bleeding. The strongest indication for a low-count transfusion.
* ``procedure_indication`` — an invasive procedure or surgery in-window whose
  policy threshold the count sits below (LP <50-80k, CVC <50k, major surgery
  <80-100k).
* ``prophylactic_marrow_failure`` — chemo / HSCT / consumptive thrombocytopenia
  with count <10k (or expected <10k within 24h), AND no exclusion population.

Deliberately NOT a blanket ``count < 10`` exemption (§8/CR-C2): a plt<10
dengue-no-bleed / TTP / HIT / ITP / aplastic-no-bleed patient is exactly the
population the policy WITHHOLDS platelets from, so a low count alone must never
exempt an LLM clear. The LLM must ground a POSITIVE indication and confirm no
exclusion applies before setting any of these True.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict


class PlateletHardSignals(BaseModel):
    """The structured positive-indication signals the LLM grounds from notes.

    All default ``False`` so a signals object built from an empty extraction
    leaves the guardrail maximally protective (any LLM clear without a grounded
    indication is floored). The LLM sets a field True only when it can cite a
    positive indication AND has ruled out the exclusion populations.
    """

    model_config = ConfigDict(frozen=True)

    active_bleeding: bool = False
    procedure_indication: bool = False
    prophylactic_marrow_failure: bool = False

    def any_signal(self) -> bool:
        """True iff at least one positive platelet indication is grounded."""
        return (
            self.active_bleeding
            or self.procedure_indication
            or self.prophylactic_marrow_failure
        )


__all__: Sequence[str] = ("PlateletHardSignals",)
