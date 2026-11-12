"""Exact provider request identity and durable transport manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bba.llm_client.models import BatchSubmissionRequest
from bba.secure_files import private_directory, read_private_bytes, write_private_bytes


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def request_manifest_hash(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(manifest)).hexdigest()


def canonical_request_manifest(
    *,
    requests: Sequence[BatchSubmissionRequest],
    model: str,
    prompt_cache_enabled: bool,
) -> dict[str, Any]:
    from bba.llm_client.transport import build_anthropic_request

    ids = [request.audit_id for request in requests]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("request manifest requires nonempty unique audit IDs")
    return {
        "format_version": 1,
        "model": model,
        "prompt_cache_enabled": prompt_cache_enabled,
        "rows": [
            {
                "audit_id": request.audit_id,
                "run_id": request.run_id,
                "task_mode": request.task_mode,
                "prompt_hash": request.prompt.prompt_hash,
                "payload": build_anthropic_request(
                    request, model=model, prompt_cache_enabled=prompt_cache_enabled
                ),
            }
            for request in requests
        ],
    }


class RequestManifestStore:
    """Persist the submission before remote creation and reject recovery drift."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def prepare(self, manifest: dict[str, Any]) -> str:
        submission_id = f"submission-{request_manifest_hash(manifest)}"
        private_directory(self.root)
        path = self.root / f"{submission_id}.json"
        try:
            self._write(path, manifest)
        except FileExistsError:
            if canonical_bytes(self._read(path)) != canonical_bytes(manifest):
                raise ValueError("submission identity conflicts with stored request")
            if self.remote_batch_id(submission_id) is None:
                raise ValueError(
                    "submission outcome is uncertain; reconcile before resubmitting"
                )
        return submission_id

    def bind(self, submission_id: str, batch_id: str) -> None:
        manifest = self._read(self.root / f"{submission_id}.json")
        for path, value in (
            (self._batch_path(batch_id), manifest),
            (self.root / f"{submission_id}.remote.json", {"batch_id": batch_id}),
        ):
            try:
                self._write(path, value)
            except FileExistsError:
                if self._read(path) != value:
                    raise ValueError(
                        "remote binding conflicts with original submission"
                    )

    def remote_batch_id(self, submission_id: str) -> str | None:
        path = self.root / f"{submission_id}.remote.json"
        if not path.exists():
            return None
        value = self._read(path).get("batch_id")
        if not isinstance(value, str) or not value:
            raise ValueError("invalid remote submission binding")
        return value

    def load_matching(self, batch_id: str, expected: dict[str, Any]) -> dict[str, Any]:
        stored = self._read(self._batch_path(batch_id))
        if canonical_bytes(stored) != canonical_bytes(expected):
            raise ValueError(
                "request manifest drift; refusing cached or remote results"
            )
        return stored

    def _batch_path(self, batch_id: str) -> Path:
        digest = hashlib.sha256(batch_id.encode("utf-8")).hexdigest()
        return self.root / f"batch-{digest}.json"

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(read_private_bytes(path))
        except FileNotFoundError as exc:
            raise ValueError(
                "request manifest missing; refusing unproven recovery"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError("invalid request manifest")
        return data

    @staticmethod
    def _write(path: Path, manifest: dict[str, Any]) -> None:
        write_private_bytes(path, canonical_bytes(manifest))
