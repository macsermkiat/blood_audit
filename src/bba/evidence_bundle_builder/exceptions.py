"""Exceptions raised by the evidence_bundle_builder pipeline."""

from __future__ import annotations


class EvidenceBundleTooLargeError(ValueError):
    """Raised when :func:`build_evidence_bundle` cannot satisfy ``char_cap``.

    Truncation drops sections (in reverse :data:`SECTION_PRIORITY`) and then
    whole items (in reverse emission order) until the bundle fits the cap.
    When even the anchor envelope alone — with the items list emptied —
    still exceeds the cap, no further reduction is possible: the cap is
    structurally unsatisfiable.

    The pipeline fails loud at construction time rather than returning an
    over-budget bundle, because the AC ("bundles never exceed token-budget
    proxy") would otherwise be silently violated and the prompt_builder /
    llm_client would see an oversized prompt with no signal to route to a
    longer-context tier or split the anchor.
    """
