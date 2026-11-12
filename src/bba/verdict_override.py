"""Say when a final verdict differs from the model's own conclusion.

Clinician review 2026-11-09: an RBC order's reasoning read "appropriate" while
its verdict said "inappropriate". The replay guardrails had overridden the
model, but reviewers saw the model's reasoning under the guardrail's verdict
with only a provenance code to connect them. The pilot review page and the
reviewer dashboard both use this module so they explain an override the same
way.
"""

from __future__ import annotations

from dataclasses import dataclass

from bba.llm_client.conclusion import closing_classification

OVERRIDE_EXPLANATIONS: dict[str, str] = {
    "llm_overclear_asserted_inappropriate": (
        "The over-clear guardrail accepts an LLM clear of a withheld order only "
        "on a structured signal (Hb at or below 7.0 g/dL, an MTP, a recorded "
        "intra-operative transfusion or major operative blood loss) or a cited "
        "bleed the model is confident in (0.8 or more) from the current episode "
        "that is over 300 mL (a quote that mentions melena cannot use the "
        "volume test) or marked life-threatening. A confidently cited ACS, "
        "instability with bleeding or fluid-refractory shock, or a "
        "SUB_THRESHOLD_HB citation whose Hb really is at or below the cohort "
        "floor sends the order to human review instead. This order had none of "
        "these"
    ),
    "llm_native_review_asserted_inappropriate": (
        "The model found no hard indication and no qualifying bleed, and the "
        "audit closes such orders as INAPPROPRIATE rather than leaving them open"
    ),
}
"""Plain-language reasons for the guardrails that replace the model's verdict
with a firm one. Keyed by ``review_reason``; other reasons have their own
labels where they are displayed."""


@dataclass(frozen=True)
class VerdictOverride:
    """The model concluded ``model_concluded``; the audit recorded ``final``."""

    model_concluded: str
    final: str
    explanation: str | None


def verdict_override(
    reasoning_en: str, final: str, review_reason: str | None
) -> VerdictOverride | None:
    """The override behind ``final``, or None when the model's verdict stands.

    Reads only the closing "Final classification: X" sentence the prompt
    requires. A reasoning without that sentence yields None rather than a
    guessed conclusion, because a wrong "model concluded" line would mislead
    the reviewer more than a missing one.
    """
    concluded = closing_classification(reasoning_en)
    if concluded is None or concluded == final.upper():
        return None
    return VerdictOverride(
        model_concluded=concluded,
        final=final.upper(),
        explanation=OVERRIDE_EXPLANATIONS.get(review_reason or ""),
    )


__all__ = ("OVERRIDE_EXPLANATIONS", "VerdictOverride", "verdict_override")
