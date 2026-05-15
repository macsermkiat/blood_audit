"""``custom_id == audit_id`` assertion.

PRD §13 makes this load-bearing: "``custom_id == audit_id`` assertion on
every result, never positional zip". A positional zip silently swaps
audit rows under partial failure or out-of-order delivery. The explicit
assertion makes the failure mode loud — :class:`CustomIdMismatchError`
names the offending IDs so the operator can quarantine without rerunning
with extra logging.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from bba.llm_client.exceptions import CustomIdMismatchError
from bba.llm_client.models import BatchSubmissionRequest, BatchSubmissionResult


def assert_custom_ids_match(
    requests: Sequence[BatchSubmissionRequest],
    results: Sequence[BatchSubmissionResult],
) -> Mapping[str, BatchSubmissionResult]:
    """Return a ``{audit_id: result}`` mapping if every submitted
    ``audit_id`` has a matching ``custom_id`` in ``results`` (and vice
    versa).

    Raises :class:`CustomIdMismatchError` with the offending IDs listed
    when:

    * a submitted ``audit_id`` has no matching ``custom_id`` in the
      response (missing result);
    * a response carries a ``custom_id`` that was never submitted
      (extra / mis-attributed result);
    * any duplicate ``custom_id`` appears in the response (Anthropic's
      Batch API does not promise dedup, so the client enforces it).
    """
    raise NotImplementedError("RED-phase scaffold; see issue #22")
