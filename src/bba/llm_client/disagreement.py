"""Sonnet/Opus classification-disagreement detection.

PRD §13: "Sonnet/Opus classification-disagreement detection →
NEEDS_REVIEW". When escalation succeeds (both Sonnet and Opus produce
parseable responses) but the two classifications differ, we route the
row to ``NEEDS_REVIEW`` with the ``disagreement`` reason instead of
silently picking one.

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

    Cases:

    * Both ``None`` → ``agreed=False, routed_to_needs_review=False``;
      escalation already routed this row to ``NEEDS_REVIEW`` via
      ``parse_failure``, no disagreement to surface.
    * One ``None`` → ``agreed=False``, but ``routed_to_needs_review=False``;
      the parseable side is accepted (the escalation pipeline already
      logged the failed side).
    * Both present and equal → ``agreed=True,
      routed_to_needs_review=False``.
    * Both present and different → ``agreed=False,
      routed_to_needs_review=True``.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #22")
