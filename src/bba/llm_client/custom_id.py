"""``custom_id == audit_id`` assertion.

PRD §13 makes this load-bearing: "``custom_id == audit_id`` assertion on
every result, never positional zip". A positional zip silently swaps
audit rows under partial failure or out-of-order delivery. The explicit
assertion makes the failure mode loud — :class:`CustomIdMismatchError`
names the offending IDs so the operator can quarantine without rerunning
with extra logging.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from bba.llm_client.exceptions import CustomIdMismatchError
from bba.llm_client.models import BatchSubmissionRequest, BatchSubmissionResult


def assert_custom_ids_match(
    requests: Sequence[BatchSubmissionRequest],
    results: Sequence[BatchSubmissionResult],
) -> Mapping[str, BatchSubmissionResult]:
    """Return a ``{audit_id: result}`` mapping if every submitted
    ``audit_id`` has exactly one matching ``custom_id`` in ``results``
    (and vice versa).

    Raises :class:`CustomIdMismatchError` with the offending IDs listed
    when the request/result custom_id sets diverge, or when the
    response contains duplicates.
    """
    submitted_ids = [r.audit_id for r in requests]
    result_ids = [r.custom_id for r in results]

    duplicate_results = sorted(cid for cid, n in Counter(result_ids).items() if n > 1)
    if duplicate_results:
        raise CustomIdMismatchError(
            f"duplicate custom_id(s) in results: {duplicate_results}"
        )

    submitted_set = set(submitted_ids)
    result_set = set(result_ids)
    missing = sorted(submitted_set - result_set)
    extra = sorted(result_set - submitted_set)
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing custom_id(s) in results: {missing}")
        if extra:
            parts.append(f"unexpected custom_id(s) in results: {extra}")
        raise CustomIdMismatchError("; ".join(parts))

    return {res.custom_id: res for res in results}
