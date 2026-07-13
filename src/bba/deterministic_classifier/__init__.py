"""bba.deterministic_classifier — Hb-tier rule engine (issue #8).

The deterministic rule engine. Given an audit row + Hb result + vitals + cohort
+ delta-Hb + procedure-proximity + crystalloid balance, returns a
:class:`ClassifierResult` whose ``classification`` is the canonical
:data:`bba.audit_store.Classification` literal and whose ``bypass_reason`` is
the structured :class:`BypassReason` enum (PRD §"Implementation Decisions §6").

The engine NEVER returns ``INAPPROPRIATE`` — that label is reserved for the
LLM stage which can apply positive-evidence reasoning (Round 1 B2). Missing
or empty notes therefore cannot bypass this invariant at the deterministic
layer; the worst-case Hb-tier output is ``POTENTIALLY_INAPPROPRIATE`` and
the LLM may downgrade it later.

Public entry point: :func:`classify`. See :mod:`bba.deterministic_classifier.models`
for the input / output schemas and :mod:`bba.deterministic_classifier.crystalloid`
for the thin MED-table totaling helper used to derive the
``crystalloid_liters_prior_4h`` input.
"""

from bba.deterministic_classifier.classifier import (
    HB_GT_10_THRESHOLD,
    HEMODILUTION_CRYSTALLOID_LITERS,
    PERI_PROCEDURAL_WINDOW_HOURS,
    PERIOP_MIN_EBL_ML,
    PRE_OP_CROSSMATCH_WINDOW_HOURS,
    UNIVERSAL_LOW_HB_APPROPRIATE_THRESHOLD,
    classify,
    periop_envelope,
)
from bba.deterministic_classifier.crystalloid import total_crystalloid_liters
from bba.deterministic_classifier.models import (
    BypassReason,
    ClassifierInputs,
    ClassifierResult,
)
from bba.deterministic_classifier.procedure_filter import (
    NON_BLOOD_PROCEDURE_ICD9,
    is_blood_requiring_procedure,
)

__all__ = [
    "HB_GT_10_THRESHOLD",
    "HEMODILUTION_CRYSTALLOID_LITERS",
    "NON_BLOOD_PROCEDURE_ICD9",
    "PERI_PROCEDURAL_WINDOW_HOURS",
    "PERIOP_MIN_EBL_ML",
    "PRE_OP_CROSSMATCH_WINDOW_HOURS",
    "UNIVERSAL_LOW_HB_APPROPRIATE_THRESHOLD",
    "BypassReason",
    "ClassifierInputs",
    "ClassifierResult",
    "classify",
    "is_blood_requiring_procedure",
    "periop_envelope",
    "total_crystalloid_liters",
]
