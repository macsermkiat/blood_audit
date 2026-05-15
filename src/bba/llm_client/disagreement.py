"""Sonnet/Opus classification-disagreement detection.

PRD §13: "Sonnet/Opus classification-disagreement detection →
NEEDS_REVIEW". When escalation succeeds (both Sonnet and Opus produce
parseable responses) but the two classifications differ, route to
``NEEDS_REVIEW`` with the ``disagreement`` reason instead of silently
picking one.

Same classification + different reasoning is NOT a disagreement — the
classification is the load-bearing field; reasoning summaries are
adjunct.
"""

from __future__ import annotations

from bba.llm_client.models import (
    DisagreementVerdict,
    LlmClassificationResponse,
)


def detect_disagreement(
    sonnet_response: LlmClassificationResponse | None,
    opus_response: LlmClassificationResponse | None,
) -> DisagreementVerdict:
    """Compare the two parsed classifications.

    Routing rules:

    * Both ``None`` → no comparison possible; routing handled by the
      parse-failure path.
    * One ``None`` → no disagreement to surface; the parseable side's
      classification is the recorded answer.
    * Both present and equal → agreement; do not route to review.
    * Both present and different → route to ``NEEDS_REVIEW``.
    """
    sonnet_cls = sonnet_response.classification if sonnet_response else None
    opus_cls = opus_response.classification if opus_response else None

    both_present = sonnet_cls is not None and opus_cls is not None
    if both_present:
        agreed = sonnet_cls == opus_cls
        return DisagreementVerdict(
            sonnet_classification=sonnet_cls,
            opus_classification=opus_cls,
            agreed=agreed,
            routed_to_needs_review=not agreed,
        )

    # Exactly one side is None: not a disagreement event.
    return DisagreementVerdict(
        sonnet_classification=sonnet_cls,
        opus_classification=opus_cls,
        agreed=False,
        routed_to_needs_review=False,
    )
