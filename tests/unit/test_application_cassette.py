"""Offline raw-export journey through the same privacy and pipeline boundary."""

from datetime import UTC, date, datetime
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from click.testing import CliRunner
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bba.application.operations import MonitoringManifest
from bba.audit_orders import build_audit_id
from bba.audit_pipeline import InMemoryBatchRunStore, PostgresBatchRunStore
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.cli.main import _resolve_run_id, cli
from bba.cli.models import AuditCommandInput
from bba.deid_redactor import (
    BackendRedactionResult,
    NoteInput,
    QuasiIdentifiers,
    RedactionRequest,
    RedactorVersion,
    redact_bundle,
)
from bba.deid_redactor.admission import PrivacyArtifact, attest_redaction
from bba.llm_client.models import (
    BatchSubmissionRequest,
    BatchSubmissionResult,
    CassetteInteraction,
    RawBatchResponse,
    SONNET_MODEL_ID,
)
from bba.prompt_builder import EvidenceChunk, PromptBuildRequest
from test_application import synthetic_export
from test_audit_pipeline_postgres import fresh_db as fresh_db


class SyntheticBackend:
    def redact(self, text):
        return BackendRedactionResult(text=text)


@pytest.mark.parametrize("use_postgres", [False, True])
def test_cassette_cli_persists_model_rows_and_formats_then_evaluates_and_monitors(
    tmp_path, monkeypatch, request, use_postgres
):
    data = tmp_path / "data"
    monkeypatch.setenv("BBA_DATA_DIR", str(data))
    monkeypatch.setattr("bba.cli.main.code_version", lambda: "synthetic-v1")
    source = synthetic_export(tmp_path / "input")
    cassette = tmp_path / "cassette.json"
    config = tmp_path / "application.json"
    config.write_text(json.dumps({"transport": "cassette", "cassette": str(cassette)}))
    monkeypatch.setenv("BBA_APPLICATION_CONFIG", str(config))
    run_id = _resolve_run_id(AuditCommandInput(input_csv=source))
    audit_id = build_audit_id("synthetic-hn", "r2")
    version = RedactorVersion(
        version="synthetic-deid", model_sha="a" * 64, gazetteer_version="synthetic"
    )
    private = b"\x01" * 32
    public = tmp_path / "public.key"
    public.write_bytes(
        Ed25519PrivateKey.from_private_bytes(private).public_key().public_bytes_raw()
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "key_id": "synthetic",
                "public_key_file": str(public),
                "approved_redactor": version.model_dump(),
            }
        )
    )
    text = "Hb 11 g/dL. Synthetic clinical evidence."
    result = redact_bundle(
        RedactionRequest(
            notes=(NoteInput(note_id="E1", text=text),),
            quasi_identifiers=QuasiIdentifiers(
                ward="synthetic",
                icd_3char="D62",
                age_band="60-69",
                sex="M",
                admission_month="2025-06",
            ),
            admission_date=date(2025, 6, 1),
            patient_age_years=65,
            redactor_version=version,
        ),
        backend=SyntheticBackend(),
        k_gate=lambda _: 5,
    )
    chunks = (EvidenceChunk(evidence_id="E1", source="Lab", text=text),)
    prompt, proof = attest_redaction(
        result,
        PromptBuildRequest(
            task_mode="HB_GT_10_OVERRIDE", cohort_threshold=7.0, evidence_chunks=chunks
        ),
        audit_id=audit_id,
        run_id=run_id,
        key_id="synthetic",
        signing_key=private,
    )
    artifact_file = tmp_path / "artifacts.json"
    artifact_file.write_text(
        json.dumps(
            {
                audit_id: PrivacyArtifact(
                    evidence_chunks=chunks, attestation=proof
                ).model_dump(mode="json")
            }
        )
    )
    submission = BatchSubmissionRequest(
        audit_id=audit_id,
        run_id=run_id,
        task_mode=prompt.task_mode,
        prompt=prompt,
        privacy_attestation=proof,
    )
    response = BatchSubmissionResult(
        custom_id=audit_id,
        model_id=SONNET_MODEL_ID,
        raw_response_json={
            "content": [
                {
                    "type": "tool_use",
                    "name": "classify_audit",
                    "input": {
                        "classification": "INAPPROPRIATE",
                        "indications": [],
                        "negative_evidence": [],
                        "reasoning_summary_en": "Synthetic absence of documented indication.",
                        "reasoning_summary_th": "กรณีทดสอบ",
                    },
                }
            ],
            "stop_reason": "tool_use",
        },
        request_json=submission.model_dump(mode="json"),
        response_headers={"anthropic-version": "2023-06-01"},
        request_timestamp=datetime.now(UTC),
        latency_ms=1,
        anthropic_version="2023-06-01",
        prompt_cache_id=None,
        extended_thinking_blocks=None,
    )
    interaction = CassetteInteraction(
        model=SONNET_MODEL_ID,
        custom_ids=(audit_id,),
        response=RawBatchResponse(batch_id="synthetic-batch", results=(response,)),
    )
    cassette.write_text(
        json.dumps(
            {
                "interactions": [
                    interaction.model_dump(
                        mode="json", fallback=lambda value: dict(value), warnings=False
                    )
                ]
            }
        )
    )
    monkeypatch.setenv("BBA_PRIVACY_POLICY", str(policy))
    monkeypatch.setenv("BBA_PRIVACY_ARTIFACTS", str(artifact_file))
    if use_postgres:
        dsn = request.getfixturevalue("fresh_db")
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                "ALTER ROLE audit_pipeline_app WITH LOGIN PASSWORD 'synthetic-batch-password'"
            )
        parsed = urlsplit(dsn)
        app_dsn = urlunsplit(
            parsed._replace(
                netloc=f"audit_pipeline_app:synthetic-batch-password@{parsed.hostname}:{parsed.port}"
            )
        )
        monkeypatch.setenv("BBA_DB_URL", app_dsn)
        batches = PostgresBatchRunStore(app_dsn)
    else:
        monkeypatch.setenv("BBA_DB_URL", "postgresql://synthetic/test")
        batches = InMemoryBatchRunStore()
        monkeypatch.setattr("bba.application.audit._batch_store", lambda _: batches)
    runner = CliRunner()
    audit = runner.invoke(cli, ["audit", "--input", str(source)])
    assert audit.exit_code == 0, str(audit.exception)
    store = AuditStore(
        AuditStoreConfig(root_dir=data / "audit_store", code_version="synthetic-v1")
    )
    rows = store.read_audit_results(run_id=run_id)
    assert len(rows) == 2
    model_row = next(r for r in rows if r.audit_id == audit_id)
    assert model_row.model_id == SONNET_MODEL_ID
    assert model_row.prompt_hash == prompt.prompt_hash
    assert model_row.final_classification == "INAPPROPRIATE"
    if use_postgres:
        batches.open()
    try:
        assert len(batches.list_all()) == 1
        assert batches.list_all()[0].state.value == "complete"
    finally:
        if use_postgres:
            batches.close()
    monkeypatch.setattr(
        "bba.cli.main._get_ward_resolver", lambda: lambda row: "synthetic-ward"
    )
    monkeypatch.setattr(
        "bba.cli.main._get_physician_resolver", lambda: lambda row: "synthetic-doctor"
    )
    for fmt in ("html", "json", "pdf"):
        report = runner.invoke(cli, ["report", "--run-id", run_id, "--format", fmt])
        assert report.exit_code == 0, str(report.exception)
        path = Path(report.output.strip())
        assert path.suffix == f".{fmt}" and path.is_file()
        if fmt == "pdf":
            assert path.read_bytes().startswith(b"%PDF")
    labels = tmp_path / "reviewed.csv"
    labels.write_text(
        "audit_id,classification\n"
        + "\n".join(f"{r.audit_id},{r.final_classification}" for r in rows)
    )
    evaluated = runner.invoke(
        cli, ["evaluate", "--run-id", run_id, "--labels", str(labels)]
    )
    assert evaluated.exit_code == 0, str(evaluated.exception)
    evaluated_path = Path(evaluated.output.strip())
    evaluated_data = json.loads(evaluated_path.read_text())
    assert evaluated_data["run_id"] == run_id
    assert evaluated_data["code_version"] == "synthetic-v1"
    assert evaluated_data["labels_sha256"]
    assert evaluated_data["audit_rows_sha256"]
    assert (
        json.loads(Path(evaluated.output.strip()).read_text())["agreement"][
            "observed_agreement"
        ]
        == 1.0
    )
    current_run = "synthetic-current"
    for row in rows:
        calls = tuple(
            call.model_copy(
                update={"run_id": current_run, "call_id": call.call_id + "-current"}
            )
            for call in store.read_llm_calls(run_id=run_id)
            if call.audit_id == row.audit_id
        )
        store.write(row.model_copy(update={"run_id": current_run}), calls)
    manifest = tmp_path / "monitoring.json"
    manifest.write_text(
        json.dumps(
            {
                "baseline_run": run_id,
                "current_run": current_run,
                "baseline_code_version": "synthetic-v1",
                "current_code_version": "synthetic-v1",
                "sentinel": {
                    "size": 2,
                    "seed": 42,
                    "audit_ids": [r.audit_id for r in rows],
                    "built_at": datetime.now(UTC).isoformat(),
                },
            }
        )
    )
    for cadence in ("weekly", "quarterly"):
        monitored = runner.invoke(
            cli, ["sentinel", f"--{cadence}", "--manifest", str(manifest)]
        )
        assert monitored.exit_code == 0, str(monitored.exception)
        monitored_path = Path(monitored.output.strip())
        monitored_data = json.loads(monitored_path.read_text())
        assert monitored_data["comparison"] == json.loads(
            MonitoringManifest.model_validate_json(
                manifest.read_text()
            ).model_dump_json()
        )
        assert monitored_data["cadence"] == cadence
        changed = json.loads(manifest.read_text())
        changed["sentinel"]["seed"] += 1
        manifest.write_text(json.dumps(changed))
        repeated = runner.invoke(
            cli, ["sentinel", f"--{cadence}", "--manifest", str(manifest)]
        )
        assert repeated.exit_code == 0, str(repeated.exception)
        assert Path(repeated.output.strip()) != monitored_path
        assert json.loads(monitored_path.read_text()) == monitored_data
