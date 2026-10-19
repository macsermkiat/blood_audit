"""Supported ingestion, context assembly, privacy admission and persistence."""

from __future__ import annotations

import os
import fcntl
from pathlib import Path

from bba.application.config import ApplicationConfig
from bba.application.deferred import DeferredBatchStore, DeferredTransport
from bba.audit_pipeline.cost_guard import assert_test_safe_transport
from bba.application.contexts import build_contexts
from bba.application.hosxp import read_tables
from bba.application.source_snapshot import snapshot_source, verify_snapshot
from bba.audit_pipeline import (
    AuditPipelineConfig,
    InMemoryBatchRunStore,
    PostgresBatchRunStore,
    run_pipeline,
)
from bba.audit_pipeline.store import BatchRunStore
from bba.audit_store import AuditStore
from bba.deid_redactor.admission import (
    PrivacyAdmissionError,
    PrivacyAdmissionPolicy,
    load_privacy_artifacts,
)
from bba.ingest import IngestConfig, ingest
from bba.llm_client import (
    AnthropicBatchTransport,
    CassetteTransport,
    LlmClientConfig,
    load_cassette,
)
from bba.llm_client.models import AnthropicTransport, BatchSubmissionRequest


def _reject_submission(request: BatchSubmissionRequest) -> None:
    raise PrivacyAdmissionError("local mode requires review, without model submission")


def _identity_key(data_dir: Path) -> bytes:
    key_path = data_dir / "identity.key"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "identity.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(os.urandom(32))
                stream.flush()
                os.fsync(stream.fileno())
    if key_path.stat().st_mode & 0o077:
        raise ValueError("identity.key must allow only its owner access")
    key = key_path.read_bytes()
    if len(key) < 32:
        raise ValueError("identity.key must contain at least 32 bytes")
    return key


def _batch_store(dsn: str) -> BatchRunStore:
    return PostgresBatchRunStore(dsn)


def audit_bundle(
    *,
    input_csv: Path,
    data_dir: Path,
    version: str,
    run_id: str,
    audit_store: AuditStore,
    config: ApplicationConfig,
) -> tuple[str, ...]:
    explicit_config = bool(os.environ.get("BBA_APPLICATION_CONFIG"))
    source_dir = snapshot_source(
        input_dir=input_csv.parent,
        data_dir=data_dir,
        version=version,
        run_id=run_id,
        config=config,
        explicit_config=explicit_config,
    )
    ingest(
        IngestConfig(
            input_dir=source_dir,
            output_dir=data_dir / "ingest",
            code_version=version,
        )
    )
    artifact_path = os.environ.get("BBA_PRIVACY_ARTIFACTS")
    artifacts = load_privacy_artifacts(Path(artifact_path)) if artifact_path else {}
    contexts = build_contexts(
        read_tables(source_dir),
        config=config,
        version=version,
        identity_key=_identity_key(data_dir),
        artifacts=artifacts,
    )
    verify_snapshot(
        source_dir,
        run_id=run_id,
        version=version,
        config_json=config.model_dump_json() if explicit_config else None,
    )
    if not contexts:
        raise ValueError(
            "bundle contains no eligible audit orders; no audit completion was recorded"
        )
    transport: AnthropicTransport = CassetteTransport(())
    admission = _reject_submission
    db_store = None
    raw_policy = os.environ.get("BBA_PRIVACY_POLICY")
    if config.transport != "local" and raw_policy:
        policy = PrivacyAdmissionPolicy.from_file(Path(raw_policy))
        admission = policy.verify

        def configured_store() -> BatchRunStore:
            dsn = os.environ.get("BBA_DB_URL")
            if not dsn:
                raise ValueError(
                    "BBA_DB_URL is required for durable model batch checkpoints"
                )
            return _batch_store(dsn)

        def configured_transport() -> AnthropicTransport:
            if config.transport == "cassette":
                assert config.cassette is not None
                return CassetteTransport(load_cassette(config.cassette))
            live = AnthropicBatchTransport(
                privacy_policy=policy, manifest_root=data_dir / "request_manifests"
            )
            assert_test_safe_transport(live, allow_live_api=config.allow_live_api)
            return live

        db_store = DeferredBatchStore(configured_store)
        transport = DeferredTransport(configured_transport)
    try:
        result = run_pipeline(
            contexts,
            transport=transport,
            audit_store=audit_store,
            batch_run_store=db_store
            if db_store is not None
            else InMemoryBatchRunStore(),
            llm_config=LlmClientConfig(code_version=version),
            pipeline_config=AuditPipelineConfig(
                db_url=os.environ.get("BBA_DB_URL", "local-no-submission"),
                code_version=version,
                allow_live_api=config.allow_live_api,
                enable_missing_platelet_defer=config.enable_missing_platelet_defer,
            ),
            run_id=run_id,
            admission_verifier=admission,
        )
    finally:
        if db_store is not None:
            db_store.close()
    if result.orphan_audit_ids:
        raise ValueError(
            "audit left uncommitted rows; resume the recorded batch before completion"
        )
    audit_store.validate_invariants(run_id)
    return tuple(
        row.audit_id
        for row in audit_store.read_audit_results(run_id=run_id, code_version=version)
    )
