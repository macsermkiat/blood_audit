"""Exporter edits must never change clinical facts beneath a recorded run ID."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from bba.application import audit as application
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.cli.main import _resolve_run_id, cli
from bba.cli.models import AuditCommandInput
from test_application import synthetic_export


def _store(data_dir: Path) -> AuditStore:
    return AuditStore(
        AuditStoreConfig(root_dir=data_dir / "audit_store", code_version="synthetic-v1")
    )


def _change_hb(source: Path) -> None:
    lab = source.parent / "Lab.csv"
    original = lab.read_text()
    assert "6.0" in original
    lab.write_text(original.replace("6.0", "12.0"))


def test_exporter_edit_after_validation_cannot_change_classified_facts(
    tmp_path, monkeypatch
):
    data = tmp_path / "data"
    source = synthetic_export(tmp_path / "input")
    monkeypatch.setenv("BBA_DATA_DIR", str(data))
    monkeypatch.setattr("bba.cli.main.code_version", lambda: "synthetic-v1")
    original_ingest = application.ingest
    validated = []

    def mutate_after_validation(config):
        validated.append(config.input_dir)
        result = original_ingest(config)
        _change_hb(source)
        return result

    monkeypatch.setattr(application, "ingest", mutate_after_validation)
    result = CliRunner().invoke(cli, ["audit", "--input", str(source)])
    assert result.exit_code == 0, str(result.exception)
    rows = _store(data).read_audit_results()
    assert next(r for r in rows if r.reqno == "r1").hb_value == 6.0
    assert validated[0] != source.parent
    assert validated[0].parent == data / "source_snapshots"
    assert all(path.stat().st_mode & 0o077 == 0 for path in validated[0].iterdir())
    assert all(path.stat().st_mode & 0o222 == 0 for path in validated[0].iterdir())


def test_edit_between_run_identity_and_snapshot_is_rejected_before_commits(
    tmp_path, monkeypatch
):
    data = tmp_path / "data"
    source = synthetic_export(tmp_path / "input")
    monkeypatch.setenv("BBA_DATA_DIR", str(data))
    monkeypatch.setattr("bba.cli.main.code_version", lambda: "synthetic-v1")
    original_resolve = _resolve_run_id

    def mutate_after_identity(inputs):
        identity = original_resolve(inputs)
        _change_hb(source)
        return identity

    monkeypatch.setattr("bba.cli.main._resolve_run_id", mutate_after_identity)
    monkeypatch.setattr(
        application,
        "run_pipeline",
        lambda *args, **kwargs: pytest.fail(
            "source drift must stop before the pipeline"
        ),
    )
    result = CliRunner().invoke(cli, ["audit", "--input", str(source)])
    assert result.exit_code != 0
    assert (
        "source" in str(result.exception).lower()
        and "identity" in str(result.exception).lower()
    )
    assert not _store(data).read_audit_results()
    assert not list((data / "audit_runs").glob("run_*.complete"))
    assert not list((data / "source_snapshots").glob(".*"))


def test_retry_reuses_verified_snapshot_and_rejects_snapshot_corruption(
    tmp_path, monkeypatch
):
    data = tmp_path / "data"
    source = synthetic_export(tmp_path / "input")
    monkeypatch.setenv("BBA_DATA_DIR", str(data))
    monkeypatch.setattr("bba.cli.main.code_version", lambda: "synthetic-v1")
    result = CliRunner().invoke(cli, ["audit", "--input", str(source)])
    assert result.exit_code == 0, str(result.exception)
    run_id = _resolve_run_id(AuditCommandInput(input_csv=source))
    snapshot = data / "source_snapshots" / run_id
    _change_hb(source)
    retry = CliRunner().invoke(cli, ["audit", "--run-id", run_id, "--force"])
    assert retry.exit_code == 0, str(retry.exception)
    os.chmod(snapshot / "Lab.csv", 0o600)
    _change_hb(snapshot / "BDVST.csv")
    os.chmod(snapshot / "Lab.csv", 0o400)
    corrupted = CliRunner().invoke(cli, ["audit", "--run-id", run_id, "--force"])
    assert corrupted.exit_code != 0
    assert any(
        word in str(corrupted.exception).lower() for word in ("source", "snapshot")
    )
    assert len(_store(data).read_audit_results()) == 2


def test_configuration_edit_after_identity_cannot_change_analysis(
    tmp_path, monkeypatch
):
    import json

    data = tmp_path / "data"
    source = synthetic_export(tmp_path / "input")
    config = tmp_path / "application.json"
    config.write_text(json.dumps({"policy_version": "approved-v1"}))
    monkeypatch.setenv("BBA_DATA_DIR", str(data))
    monkeypatch.setenv("BBA_APPLICATION_CONFIG", str(config))
    monkeypatch.setattr("bba.cli.main.code_version", lambda: "synthetic-v1")
    original_resolve = _resolve_run_id

    def mutate_after_identity(inputs):
        identity = original_resolve(inputs)
        config.write_text(json.dumps({"policy_version": "changed-v2"}))
        return identity

    monkeypatch.setattr("bba.cli.main._resolve_run_id", mutate_after_identity)
    result = CliRunner().invoke(cli, ["audit", "--input", str(source)])
    assert result.exit_code != 0
    assert "identity" in str(result.exception).lower()
    assert not _store(data).read_audit_results()
    assert not list((data / "source_snapshots").glob(".*"))


def test_private_snapshot_change_after_read_is_rejected_before_pipeline(
    tmp_path, monkeypatch
):
    data = tmp_path / "data"
    source = synthetic_export(tmp_path / "input")
    monkeypatch.setenv("BBA_DATA_DIR", str(data))
    monkeypatch.setattr("bba.cli.main.code_version", lambda: "synthetic-v1")
    original_read = application.read_tables

    def alter_snapshot_after_read(directory):
        tables = original_read(directory)
        os.chmod(directory / "Lab.csv", 0o600)
        _change_hb(directory / "BDVST.csv")
        os.chmod(directory / "Lab.csv", 0o400)
        return tables

    monkeypatch.setattr(application, "read_tables", alter_snapshot_after_read)
    monkeypatch.setattr(
        application,
        "run_pipeline",
        lambda *args, **kwargs: pytest.fail(
            "changed snapshot must not reach the pipeline"
        ),
    )
    result = CliRunner().invoke(cli, ["audit", "--input", str(source)])
    assert result.exit_code != 0
    assert "snapshot" in str(result.exception).lower()
    assert not _store(data).read_audit_results()
    assert not list((data / "audit_runs").glob("run_*.complete"))


def test_manifest_preserves_full_source_identity_and_fsyncs_before_publish(
    tmp_path, monkeypatch
):
    import json
    from bba.application import source_snapshot
    from bba.application.config import ApplicationConfig
    from bba.ingest import RunIdentity, schema_fingerprint

    data = tmp_path / "data"
    source = synthetic_export(tmp_path / "input")
    monkeypatch.setattr("bba.cli.main.code_version", lambda: "synthetic-v1")
    run_id = _resolve_run_id(AuditCommandInput(input_csv=source))
    synced = []
    original_fsync = source_snapshot.os.fsync

    def record_fsync(descriptor):
        synced.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    monkeypatch.setattr(source_snapshot.os, "fsync", record_fsync)
    directory = source_snapshot.snapshot_source(
        input_dir=source.parent,
        data_dir=data,
        version="synthetic-v1",
        run_id=run_id,
        config=ApplicationConfig(),
        explicit_config=False,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    identity = RunIdentity.from_inputs(
        manifest["file_hashes"], schema_fingerprint(), "synthetic-v1"
    ).run_id
    assert manifest["source_identity"] == identity
    assert identity[:16] == run_id
    binding_path = data / "audit_runs" / f"{run_id}.input.json"
    binding = json.loads(binding_path.read_text())
    assert Path(binding["input_csv"]).parent == directory
    assert binding_path.stat().st_mode & 0o177 == 0
    assert binding_path.parent.stat().st_mode & 0o077 == 0
    assert len(synced) >= 17
    assert directory.stat().st_mode & 0o277 == 0
    assert (data / "source_snapshots").stat().st_mode & 0o077 == 0


def test_interrupted_run_resumes_from_snapshot_after_exporter_removal(
    tmp_path, monkeypatch
):
    data = tmp_path / "data"
    source = synthetic_export(tmp_path / "input")
    monkeypatch.setenv("BBA_DATA_DIR", str(data))
    monkeypatch.setattr("bba.cli.main.code_version", lambda: "synthetic-v1")
    run_id = _resolve_run_id(AuditCommandInput(input_csv=source))
    original_pipeline = application.run_pipeline

    def interrupt_before_commit(*args, **kwargs):
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(application, "run_pipeline", interrupt_before_commit)
    interrupted = CliRunner().invoke(cli, ["audit", "--input", str(source)])
    assert interrupted.exit_code != 0
    assert not _store(data).read_audit_results()
    source.parent.rename(tmp_path / "exporter-removed")
    monkeypatch.setattr(application, "run_pipeline", original_pipeline)
    resumed = CliRunner().invoke(cli, ["audit", "--run-id", run_id])
    assert resumed.exit_code == 0, str(resumed.exception)
    rows = _store(data).read_audit_results()
    assert len(rows) == 2
    assert next(r for r in rows if r.reqno == "r1").hb_value == 6.0
