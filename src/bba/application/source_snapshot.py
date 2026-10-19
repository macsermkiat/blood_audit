"""Seal exporter-owned bytes before validation or classification can read them."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from bba.application.config import ApplicationConfig
from bba.audit_store.models import SafeId
from bba.ingest import RunIdentity, all_tables, content_hash, schema_fingerprint
from bba.ingest.models import CSVTable
from bba.secure_files import private_directory, write_private_bytes


class SourceSnapshotError(ValueError):
    """Source bytes or configuration differ from the requested audit identity."""


class SourceSnapshotManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: SafeId
    source_identity: str = Field(pattern=r"^[a-f0-9]{64}$")
    schema_fingerprint: str
    code_version: str
    config_json: str | None
    file_hashes: dict[CSVTable, str]


def _run_id(identity: str, config_json: str | None) -> str:
    base = identity[:16]
    return (
        base
        if config_json is None
        else hashlib.sha256(f"{base}|{config_json}".encode()).hexdigest()[:16]
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _private(path: Path, *, directory: bool = False, sealed: bool = True) -> None:
    metadata = path.lstat()
    expected = (
        stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    )
    if (
        not expected
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
        or (sealed and metadata.st_mode & 0o222)
    ):
        raise SourceSnapshotError(
            "source snapshot must use private, owned, read-only files and directories"
        )


def verify_snapshot(
    directory: Path, *, run_id: str, version: str, config_json: str | None
) -> SourceSnapshotManifest:
    _private(directory, directory=True)
    manifest_path = directory / "manifest.json"
    _private(manifest_path)
    manifest = SourceSnapshotManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    expected_names = {f"{table}.csv" for table in all_tables()} | {"manifest.json"}
    if {path.name for path in directory.iterdir()} != expected_names or set(
        manifest.file_hashes
    ) != set(all_tables()):
        raise SourceSnapshotError(
            "source snapshot membership differs from the canonical bundle"
        )
    if (
        manifest.run_id != run_id
        or manifest.code_version != version
        or manifest.schema_fingerprint != schema_fingerprint()
        or manifest.config_json != config_json
    ):
        raise SourceSnapshotError("source snapshot identity or configuration changed")
    hashes = {}
    for table in all_tables():
        path = directory / f"{table}.csv"
        _private(path)
        hashes[table] = content_hash(path)
    identity = RunIdentity.from_inputs(
        hashes, manifest.schema_fingerprint, version
    ).run_id
    if (
        hashes != manifest.file_hashes
        or identity != manifest.source_identity
        or _run_id(identity, config_json) != run_id
    ):
        raise SourceSnapshotError(
            "source snapshot bytes no longer match the requested identity"
        )
    return manifest


def _persist_source_binding(
    directory: Path, *, data_dir: Path, run_id: str, config: ApplicationConfig
) -> None:
    bindings = data_dir / "audit_runs"
    private_directory(bindings)
    target = bindings / f"{run_id}.input.json"
    payload = {
        "input_csv": str((directory / "BDVST.csv").resolve()),
        "config": config.model_dump(mode="json"),
    }
    write_private_bytes(target, json.dumps(payload).encode("utf-8"), replace=True)


def snapshot_source(
    *,
    input_dir: Path,
    data_dir: Path,
    version: str,
    run_id: str,
    config: ApplicationConfig,
    explicit_config: bool,
) -> Path:
    """Copy and hash once, then publish only bytes matching the existing CLI ID."""
    config_json = config.model_dump_json() if explicit_config else None
    root = data_dir / "source_snapshots"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _private(root, directory=True, sealed=False)
    target = root / TypeAdapter(SafeId).validate_python(run_id)
    if target.exists():
        verify_snapshot(target, run_id=run_id, version=version, config_json=config_json)
        _persist_source_binding(target, data_dir=data_dir, run_id=run_id, config=config)
        return target
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=root))
    published = False
    try:
        hashes: dict[CSVTable, str] = {}
        for table in all_tables():
            source = input_dir / f"{table}.csv"
            destination = staging / source.name
            digest = hashlib.sha256()
            with source.open("rb") as original:
                descriptor = os.open(
                    destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                with os.fdopen(descriptor, "wb") as copied:
                    for chunk in iter(lambda: original.read(65536), b""):
                        digest.update(chunk)
                        copied.write(chunk)
                    copied.flush()
                    os.fsync(copied.fileno())
            hashes[table] = digest.hexdigest()
            destination.chmod(0o400)
        identity = RunIdentity.from_inputs(hashes, schema_fingerprint(), version).run_id
        if _run_id(identity, config_json) != run_id:
            raise SourceSnapshotError(
                "source bytes or configuration changed after CLI identity computation"
            )
        manifest = SourceSnapshotManifest(
            run_id=run_id,
            source_identity=identity,
            schema_fingerprint=schema_fingerprint(),
            code_version=version,
            config_json=config_json,
            file_hashes=hashes,
        )
        descriptor = os.open(
            staging / "manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(manifest.model_dump_json(indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        (staging / "manifest.json").chmod(0o400)
        _fsync_directory(staging)
        staging.chmod(0o500)
        staging.rename(target)
        published = True
        _fsync_directory(root)
        _fsync_directory(data_dir)
        verify_snapshot(target, run_id=run_id, version=version, config_json=config_json)
        _persist_source_binding(target, data_dir=data_dir, run_id=run_id, config=config)
        return target
    finally:
        if not published:
            staging.chmod(0o700)
            shutil.rmtree(staging)
