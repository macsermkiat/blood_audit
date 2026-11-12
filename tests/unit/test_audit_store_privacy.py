from __future__ import annotations

import os
import runpy
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pytest

from bba.audit_store import AuditStore, AuditStoreConfig, SnapshotView


def test_audit_history_is_private_under_default_umask(tmp_path: Path):
    helpers = runpy.run_path("tests/unit/test_audit_store.py")
    store = AuditStore(AuditStoreConfig(root_dir=tmp_path / "audit", code_version="v1"))
    previous = os.umask(0o022)
    try:
        store.write(helpers["_row"](), [helpers["_call"]()])
        SnapshotView.open(store, date(2026, 10, 19))
    finally:
        os.umask(previous)
    paths = list(store.config.root_dir.rglob("*"))
    assert paths
    for path in paths:
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)


def test_unsafe_existing_audit_records_are_not_read(tmp_path: Path):
    helpers = runpy.run_path("tests/unit/test_audit_store.py")
    store = AuditStore(AuditStoreConfig(root_dir=tmp_path / "audit", code_version="v1"))
    store.write(helpers["_row"](), [helpers["_call"]()])
    record = next((store.config.root_dir / "audit_results").glob("*.parquet"))
    record.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        store.read_audit_results()


def test_concurrent_snapshot_materialization_keeps_one_complete_snapshot(
    tmp_path: Path,
):
    helpers = runpy.run_path("tests/unit/test_audit_store.py")
    store = AuditStore(AuditStoreConfig(root_dir=tmp_path / "audit", code_version="v1"))
    store.write(helpers["_row"](), [helpers["_call"]()])

    def materialize(_):
        return SnapshotView.open(store, date(2026, 10, 19)).read_audit_results()

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(materialize, range(5)))
    assert all(result == results[0] for result in results)
    assert len(results[0]) == 1
    assert len(list(store.snapshots_dir.glob("*.parquet"))) == 1
