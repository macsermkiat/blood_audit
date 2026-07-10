"""Re-apply cached pilot LLM responses with the current audit transform.

This driver rebuilds pilot contexts with the current cohorts, dispatch, and
guardrails, then replaces the live Anthropic transport before invoking
``run_llm_leg.main()``. No Anthropic call is made: the resume batch id skips
submission, and the replacement transport only replays audit-store calls.

The replayed responses retain their original ``request_json`` and therefore
their old prompts. This is deliberate: the driver validates the response-to-
audit-row transform, not the prompt rewrite, which still requires a live batch.

Outputs, including ``llm_report.json``, land under ``BBA_PILOT_WORK_DIR`` and
follow the same PHI handling rules as the pilot README. The report is
overwritten; snapshot the previous file first when a diff is wanted.

Environment variables:

* ``BBA_PILOT_SOURCE_RUN_ID`` — run id containing cached ``llm_calls``
  (default: ``pilot-mini``).
* ``BBA_PILOT_RUN_ID`` — required new run id for re-applied audit rows; it must
  differ from the source run id.
* ``BBA_PILOT_WORK_DIR`` — pilot bundle, audit-store, and report directory
  (default inherited from ``run_llm_leg.py``: ``/tmp/bba_mini``).
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    source_run_id = os.environ.get("BBA_PILOT_SOURCE_RUN_ID", "pilot-mini")
    target_run_id = os.environ.get("BBA_PILOT_RUN_ID")
    if not target_run_id:
        sys.exit(
            "BBA_PILOT_RUN_ID is required and must name the new offline re-apply run"
        )
    if target_run_id == source_run_id:
        sys.exit(
            "BBA_PILOT_RUN_ID must differ from BBA_PILOT_SOURCE_RUN_ID; "
            "reusing the source run id would make every audit-store write "
            "an idempotent no-op"
        )

    os.environ.setdefault("ANTHROPIC_API_KEY", "offline-reapply-no-spend")
    os.environ["BBA_PILOT_BATCH_ID"] = "offline-reapply"

    import run_llm_leg
    from bba.audit_pipeline.resume import _result_from_cached_call
    from bba.audit_store import AuditStore, AuditStoreConfig, LlmCall
    from bba.llm_client.models import (
        BatchSubmissionRequest,
        BatchSubmissionResult,
        RawBatchResponse,
    )

    store = AuditStore(
        AuditStoreConfig(
            root_dir=run_llm_leg.AUDIT_STORE_ROOT,
            code_version=run_llm_leg.CODE_VERSION,
        )
    )
    calls_by_audit_id: dict[str, list[LlmCall]] = {}
    for call in store.read_llm_calls(run_id=source_run_id):
        calls_by_audit_id.setdefault(call.audit_id, []).append(call)
    if not calls_by_audit_id:
        sys.exit(
            f"no cached llm_calls found under BBA_PILOT_SOURCE_RUN_ID={source_run_id!r}"
        )

    class OfflineReplayTransport:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def fetch_batch_results(
            self,
            batch_id: str,
            *,
            model: str,
            requests: list[BatchSubmissionRequest],
            prompt_cache_enabled: bool,
        ) -> RawBatchResponse:
            del model, prompt_cache_enabled
            results: list[BatchSubmissionResult] = []
            requested_audit_ids: set[str] = set()
            replayed_audit_ids: set[str] = set()
            missing_audit_ids: list[str] = []

            for request in requests:
                audit_id = request.audit_id
                requested_audit_ids.add(audit_id)
                cached_calls = calls_by_audit_id.get(audit_id)
                if not cached_calls:
                    missing_audit_ids.append(audit_id)
                    continue
                replayed_audit_ids.add(audit_id)
                for call in cached_calls:
                    # Deliberately reuse the resume reconciler's byte-faithful
                    # reconstruction; it strips the writer-folded
                    # __bba_response_headers__ envelope key.
                    results.append(_result_from_cached_call(call))

            if missing_audit_ids:
                print(f"  awaiting live batch: {', '.join(missing_audit_ids)}")
            unrequested_count = len(calls_by_audit_id.keys() - requested_audit_ids)
            print(f"  cached audit_ids not requested: {unrequested_count}")
            print(
                f"  offline replay: requested {len(requests)}, replayed "
                f"{len(results)} results for {len(replayed_audit_ids)} audit_ids, "
                f"missing {len(missing_audit_ids)}"
            )
            return RawBatchResponse(batch_id=batch_id, results=tuple(results))

        def submit_batch_only(self, **kw: object) -> str:
            del kw
            raise RuntimeError("offline replay never submits")

        def submit_batch(self, **kw: object) -> RawBatchResponse:
            del kw
            raise RuntimeError("offline replay never submits")

    run_llm_leg.RealAnthropicTransport = OfflineReplayTransport
    run_llm_leg.main()


if __name__ == "__main__":
    main()
