"""Immutable request and evidence provenance for batch recovery."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bba import feature_flags
from bba.audit_pipeline.exceptions import ResumeReconciliationError
from bba.audit_pipeline.models import AuditPipelineConfig, BatchRun, PipelineRowContext
from bba.audit_store import AuditStore
from bba.llm_client.models import BatchSubmissionRequest, LlmClientConfig
from bba.llm_client.request_identity import canonical_bytes, canonical_request_manifest
from bba.secure_files import read_private_bytes, write_private_bytes


def _bytes(value: Any) -> bytes:
    return canonical_bytes(value)


def context_hash(
    context: PipelineRowContext, *, enable_missing_platelet_defer: bool = False
) -> str:
    return hashlib.sha256(
        canonical_bytes(
            {
                "context": json.loads(context.model_dump_json()),
                "feature_flags": _feature_flags(),
                "enable_missing_platelet_defer": enable_missing_platelet_defer,
            }
        )
    ).hexdigest()


def _path(store: AuditStore, batch_id: str) -> Path:
    return (
        store.config.root_dir
        / "_batch_manifests"
        / (hashlib.sha256(batch_id.encode()).hexdigest() + ".json")
    )


def has_manifest(audit_store: AuditStore, batch_id: str) -> bool:
    return _path(audit_store, batch_id).exists()


def _feature_flags() -> dict[str, bool]:
    return {
        name: value
        for name, value in vars(feature_flags).items()
        if name.isupper() and isinstance(value, bool)
    }


def persist_run_contexts(
    *,
    audit_store: AuditStore,
    run_id: str,
    contexts: Sequence[PipelineRowContext],
    admission_required: bool = False,
    pipeline_config: AuditPipelineConfig | None = None,
) -> None:
    """Reject a repeated run whose clinical inputs or policy switches changed."""
    digest = hashlib.sha256(
        f"{run_id}|{audit_store.config.code_version}".encode()
    ).hexdigest()
    path = audit_store.config.root_dir / "_run_contexts" / f"{digest}.json"
    payload = _bytes(
        {
            "admission_required": admission_required,
            "pipeline_policy": pipeline_config.model_dump(
                mode="json", exclude={"db_url", "poll_interval_seconds"}
            )
            if pipeline_config is not None
            else None,
            "context_hashes": {c.order.audit_id: context_hash(c) for c in contexts},
            "feature_flags": _feature_flags(),
        }
    )
    try:
        write_private_bytes(path, payload)
    except FileExistsError:
        if read_private_bytes(path) != payload:
            raise ResumeReconciliationError(
                "run context or policy identity changed; use a new run ID"
            ) from None


def persist_manifest(
    *,
    run: BatchRun,
    audit_store: AuditStore,
    contexts: Sequence[PipelineRowContext],
    requests: Sequence[BatchSubmissionRequest],
    llm_config: LlmClientConfig,
    enable_missing_platelet_defer: bool,
) -> None:
    request_by_id = {request.audit_id: request for request in requests}
    contexts = tuple(
        context.model_copy(
            update={
                "prompt_hash": request_by_id[context.order.audit_id].prompt.prompt_hash
            }
        )
        for context in contexts
    )
    manifest = {
        "format_version": 1,
        "feature_flags": _feature_flags(),
        "batch_id": run.batch_id,
        "run_id": run.run_id,
        "code_version": run.code_version,
        "audit_ids": list(run.audit_ids),
        "context_hashes": {c.order.audit_id: context_hash(c) for c in contexts},
        "requests": [r.model_dump(mode="json") for r in requests],
        "request_identity": canonical_request_manifest(
            requests=requests,
            model=llm_config.sonnet_model_id,
            prompt_cache_enabled=llm_config.prompt_cache_enabled,
        ),
        "llm_config": llm_config.model_dump(mode="json"),
        "enable_missing_platelet_defer": enable_missing_platelet_defer,
    }
    path = _path(audit_store, run.batch_id)
    payload = _bytes(manifest)
    try:
        write_private_bytes(path, payload)
    except FileExistsError:
        if read_private_bytes(path) != payload:
            raise ResumeReconciliationError(
                f"batch {run.batch_id!r} request or evidence identity changed"
            ) from None


def load_manifest(
    *,
    run: BatchRun,
    audit_store: AuditStore,
    contexts: Mapping[str, PipelineRowContext],
    llm_config: LlmClientConfig | None = None,
    enable_missing_platelet_defer: bool = False,
) -> tuple[list[BatchSubmissionRequest], LlmClientConfig]:
    try:
        manifest = json.loads(read_private_bytes(_path(audit_store, run.batch_id)))
        if manifest["format_version"] != 1 or any(
            manifest[key] != value
            for key, value in {
                "feature_flags": _feature_flags(),
                "batch_id": run.batch_id,
                "run_id": run.run_id,
                "code_version": run.code_version,
                "audit_ids": list(run.audit_ids),
                "context_hashes": {
                    aid: context_hash(contexts[aid]) for aid in run.audit_ids
                },
                "enable_missing_platelet_defer": enable_missing_platelet_defer,
            }.items()
        ):
            raise ValueError("batch or evidence identity changed")
        config = LlmClientConfig.model_validate(manifest["llm_config"])
        if llm_config is not None and config != llm_config:
            raise ValueError("model configuration changed")
        requests = [
            BatchSubmissionRequest.model_validate(r) for r in manifest["requests"]
        ]
        if tuple(r.audit_id for r in requests) != run.audit_ids or any(
            r.run_id != run.run_id for r in requests
        ):
            raise ValueError("request membership changed")
        identity = canonical_request_manifest(
            requests=requests,
            model=config.sonnet_model_id,
            prompt_cache_enabled=config.prompt_cache_enabled,
        )
        if _bytes(identity) != _bytes(manifest["request_identity"]):
            raise ValueError("request bytes changed")
        return requests, config
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ResumeReconciliationError(
            f"batch {run.batch_id!r} lacks matching immutable request provenance: {exc}"
        ) from exc
