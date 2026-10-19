"""The supported command must persist classifications, never ingestion markers."""

from pathlib import Path
import csv
import pytest

from click.testing import CliRunner

from bba.cli.main import cli
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.ingest import all_tables
from bba.ingest.schemas import get_schema


def synthetic_export(path: Path) -> Path:
    path.mkdir()
    rows = {
        "BDVST": [
            dict(
                HN="synthetic-hn",
                AN="synthetic-an",
                REQNO="r1",
                BDVSTST="4",
                REQTYPE="P",
                REQDATE="2025-06-01",
                REQTIME="120000",
            ),
            dict(
                HN="synthetic-hn",
                AN="synthetic-an",
                REQNO="r2",
                BDVSTST="4",
                REQTYPE="P",
                REQDATE="2025-06-01",
                REQTIME="140000",
            ),
        ],
        "BDVSTDT": [
            dict(HN="synthetic-hn", REQNO=r, BDTYPE="LPRC", UNITAMT="1", USETYPE="1")
            for r in ("r1", "r2")
        ],
        "Lab": [
            dict(
                HN="synthetic-hn",
                AN="synthetic-an",
                LVSTDATE="2025-06-01",
                LVSTTIME=t,
                LABEXM="290095",
                RESULT=v,
            )
            for t, v in (("110000", "6.0"), ("130000", "11.0"))
        ],
        "IPDADMPROGRESS": [
            dict(
                HN="synthetic-hn",
                AN="synthetic-an",
                PROGDATE="2025-06-01 13:30:00",
                OBJECTIVE="Synthetic patient stable, BP 120/80 HR 80",
            )
        ],
    }
    for table in all_tables():
        with (path / f"{table}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(get_schema(table).columns))
            writer.writeheader()
            writer.writerows(rows.get(table, []))
    return path / "BDVST.csv"


def test_canonical_progress_timestamp_reaches_clinical_extractors(tmp_path):
    from datetime import UTC, datetime
    from bba.application.config import ApplicationConfig
    from bba.application.contexts import build_contexts
    from bba.application.hosxp import read_tables, notes, orders

    source = synthetic_export(tmp_path / "input")
    tables = read_tables(source.parent)
    progress, _ = notes(tables, orders(tables, "synthetic")[1])
    assert len(progress) == 1
    assert progress[0].timestamp == datetime(2025, 6, 1, 6, 30, tzinfo=UTC)
    contexts = build_contexts(
        tables,
        config=ApplicationConfig(),
        version="synthetic",
        identity_key=b"x" * 32,
        artifacts={},
    )
    assert contexts[1].vitals_result.vitals.sbp == 120
    assert contexts[1].vitals_result.vitals.hr == 80


def test_transfusion_history_stays_with_its_patient_and_admission(tmp_path):
    from bba.application.config import ApplicationConfig
    from bba.application.contexts import build_contexts
    from bba.application.hosxp import read_tables

    source = synthetic_export(tmp_path / "input")
    for table, changes in (
        ("BDVST", {"AN": "other-admission", "REQNO": "r3"}),
        ("BDVSTDT", {"REQNO": "r3", "UNITAMT": "2"}),
        ("Lab", {"AN": "other-admission"}),
    ):
        path = source.parent / f"{table}.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            row = next(reader)
        with path.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow({**row, **changes})
    contexts = build_contexts(
        read_tables(source.parent),
        config=ApplicationConfig(),
        version="synthetic",
        identity_key=b"x" * 32,
        artifacts={},
    )
    by_order = {context.order.reqno: context for context in contexts}
    assert by_order["r2"].prior_rbc_units_24h == 1
    assert by_order["r3"].prior_rbc_units_24h == 0


def test_local_audit_commits_actual_rows_and_no_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("BBA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("bba.cli.main.code_version", lambda: "synthetic-v1")
    source = synthetic_export(tmp_path / "input")
    result = CliRunner().invoke(cli, ["audit", "--input", str(source)])
    assert result.exit_code == 0, result.output or str(result.exception)
    store = AuditStore(
        AuditStoreConfig(
            root_dir=tmp_path / "data" / "audit_store", code_version="synthetic-v1"
        )
    )
    rows = store.read_audit_results(code_version="synthetic-v1")
    assert len(rows) == 2
    assert {row.final_classification for row in rows} == {"APPROPRIATE", "NEEDS_REVIEW"}
    assert {row.model_id for row in rows} == {"deterministic", "privacy-filter"}
    assert all("Synthetic patient" not in row.model_dump_json() for row in rows)
    again = CliRunner().invoke(cli, ["audit", "--input", str(source)])
    assert again.exit_code == 0
    assert len(store.read_audit_results()) == 2


@pytest.mark.parametrize("with_policy", [False, True])
def test_configured_model_without_privacy_stays_local(
    tmp_path, monkeypatch, with_policy
):
    import json

    monkeypatch.setenv("BBA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("bba.cli.main.code_version", lambda: "synthetic-v1")
    monkeypatch.delenv("BBA_PRIVACY_POLICY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("BBA_DB_URL", raising=False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"transport": "live", "allow_live_api": True}))
    monkeypatch.setenv("BBA_APPLICATION_CONFIG", str(config))
    if with_policy:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        public = tmp_path / "public.key"
        public.write_bytes(Ed25519PrivateKey.generate().public_key().public_bytes_raw())
        policy = tmp_path / "policy.json"
        policy.write_text(
            json.dumps(
                {
                    "key_id": "synthetic",
                    "public_key_file": str(public),
                    "approved_redactor": {
                        "version": "synthetic",
                        "model_sha": "a" * 64,
                        "gazetteer_version": "synthetic",
                    },
                }
            )
        )
        monkeypatch.setenv("BBA_PRIVACY_POLICY", str(policy))
    source = synthetic_export(tmp_path / "input")
    monkeypatch.setattr(
        "bba.application.audit.AnthropicBatchTransport",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("no transport before admission")
        ),
    )
    result = CliRunner().invoke(cli, ["audit", "--input", str(source)])
    assert result.exit_code == 0, str(result.exception)
    rows = AuditStore(
        AuditStoreConfig(
            root_dir=tmp_path / "data" / "audit_store", code_version="synthetic-v1"
        )
    ).read_audit_results()
    assert {row.model_id for row in rows} == {"deterministic", "privacy-filter"}


def test_application_configuration_rejects_unsafe_or_inconsistent_modes():
    from bba.application.config import ApplicationConfig
    import pytest

    for values in (
        {"transport": "live"},
        {"allow_live_api": True},
        {"transport": "cassette"},
        {"cassette": "/tmp/x"},
        {"execute": "arbitrary-python"},
    ):
        with pytest.raises(ValueError):
            ApplicationConfig.model_validate(values)


def test_dashboard_cli_preserves_signed_auth_boundary(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from fastapi.testclient import TestClient
    from bba.review_actions import ReviewActionsStore

    monkeypatch.setenv("BBA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BBA_DB_URL", "postgresql://reviewer/test")
    monkeypatch.setenv(
        "BBA_DASHBOARD_AUTH_SECRET", "synthetic-secret-32-bytes-minimum!"
    )
    monkeypatch.setenv("BBA_DASHBOARD_PUBLIC_ORIGIN", "https://audit.example.test")
    monkeypatch.setattr("bba.cli.main._get_ward_resolver", lambda: lambda row: "ward")
    monkeypatch.setattr(
        "bba.cli.main._get_physician_resolver", lambda: lambda row: "doctor"
    )
    reviews = MagicMock(spec=ReviewActionsStore)
    monkeypatch.setattr("bba.review_actions.ReviewActionsStore", lambda config: reviews)
    observed = []

    def serve(app, **kwargs):
        observed.append(kwargs)
        with TestClient(app, base_url="https://audit.example.test") as client:
            assert client.get("/queue").status_code == 401

    monkeypatch.setattr("uvicorn.run", serve)
    result = CliRunner().invoke(
        cli, ["serve-dashboard", "--port", "8001", "--run-id", "synthetic-run"]
    )
    assert result.exit_code == 0, str(result.exception)
    assert observed == [{"host": "127.0.0.1", "port": 8001}]
    reviews.close.assert_called_once()


def test_legacy_ingestion_markers_do_not_skip_classification(tmp_path, monkeypatch):
    from bba.cli.audit_run_store import FileBackedAuditRunStore
    from bba.cli.main import _resolve_run_id
    from bba.cli.models import AuditCommandInput

    monkeypatch.setenv("BBA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("bba.cli.main.code_version", lambda: "synthetic-v1")
    source = synthetic_export(tmp_path / "input")
    run_id = _resolve_run_id(AuditCommandInput(input_csv=source))
    runs = FileBackedAuditRunStore(tmp_path / "data")
    runs.record_row(run_id, "phase1_ingest_BDVST")
    runs.mark_run_complete(run_id)
    assert not runs.run_complete(run_id)
    result = CliRunner().invoke(cli, ["audit", "--input", str(source)])
    assert result.exit_code == 0, str(result.exception)
    assert runs.run_count(run_id) == 2
    assert runs.run_complete(run_id)
