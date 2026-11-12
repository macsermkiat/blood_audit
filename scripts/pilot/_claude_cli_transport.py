"""Subscription-backed transport for the pilot LLM leg: one ``claude -p`` per row.

NOT a production transport. ``run_llm_leg.py`` selects it with
``BBA_PILOT_TRANSPORT=claude-cli`` so an A/B on a prompt or evidence change
can run on the claude.ai subscription instead of Anthropic API credit. It
mirrors the :class:`bba.llm_client.AnthropicTransport` Protocol so the rest of
the leg (custom_id assertion, parser, quote grounder, audit store) is
untouched:

* ``submit_batch_only`` creates ``<batch_root>/<batch_id>/`` with a manifest
  and returns the id (no model call yet).
* ``fetch_batch_results`` runs each pending request once, ``workers`` at a
  time. Completed answers are checkpointed. An uncertain outcome retains an
  inflight marker and requires reconciliation before another execution.
  Resume uses ``BBA_PILOT_BATCH_ID=<batch_id>`` under the original
  ``BBA_PILOT_RUN_ID`` and exact request configuration. Once the CLI reports
  the session limit the rest of the batch fails fast.

Each call is ``claude -p --model M --tools "" --json-schema <tool input
schema> --system-prompt <system blocks>`` with the user blocks on stdin. The
CLI's ``structured_output`` is wrapped as a ``tool_use`` block named after the
audit tool, so the parser sees the API shape. A CLI failure becomes the same
empty-content envelope the batch path emits for an errored entry (routes to
NEEDS_REVIEW). Provenance is written on the envelope (``_transport``,
``_cli_session_id``, ``_cli_models_used``) so a reviewer can tell these rows
from API rows.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bba.deid_redactor.admission import PrivacyAdmissionError, PrivacyAdmissionPolicy
from bba.llm_client import BatchSubmissionRequest, BatchSubmissionResult
from bba.llm_client.request_identity import canonical_bytes, canonical_request_manifest
from bba.llm_client.models import RawBatchResponse
from bba.llm_client.transport import build_anthropic_request
from bba.secure_files import (
    private_directory,
    read_private_bytes,
    unlink_private,
    write_private_bytes,
)

TRANSPORT_NAME = "claude-cli"
_TOOL_NAME = "classify_transfusion_order"
_CLI_TIMEOUT_S = 900.0


@dataclass(frozen=True)
class CliOutcome:
    stdout: str
    stderr: str
    returncode: int


Runner = Callable[[list[str], str], CliOutcome]


def _subprocess_runner(argv: list[str], stdin: str) -> CliOutcome:
    proc = subprocess.run(
        argv,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=_CLI_TIMEOUT_S,
        check=False,
    )
    return CliOutcome(
        stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode
    )


def build_cli_argv(*, model: str, payload: dict[str, Any]) -> tuple[list[str], str]:
    """Translate an Anthropic Messages payload into ``claude -p`` argv + stdin."""
    system_text = "\n\n".join(b["text"] for b in payload["system"])
    user_text = "\n\n".join(b["text"] for b in payload["messages"][0]["content"])
    schema = payload["tools"][0]["input_schema"]
    argv = [
        "claude",
        "-p",
        "--model",
        model,
        "--output-format",
        "json",
        "--tools",
        "",
        "--no-session-persistence",
        "--json-schema",
        json.dumps(schema, ensure_ascii=False),
        "--system-prompt",
        system_text,
    ]
    return argv, user_text


def _error_envelope(model: str, detail: str, session_id: str | None) -> dict[str, Any]:
    # Same shape as transport._result_from_batch_entry's error path: no
    # content, so the parser raises EMPTY_RESPONSE and the row routes to
    # NEEDS_REVIEW with the detail preserved for the reviewer.
    return {
        "_transport": TRANSPORT_NAME,
        "_cli_error": detail,
        "_cli_session_id": session_id,
        "id": "",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": "cli_error",
    }


def _envelope_from_cli(model: str, outcome: CliOutcome) -> dict[str, Any]:
    try:
        data = json.loads(outcome.stdout)
    except ValueError:
        detail = f"claude exited {outcome.returncode}; stderr: {outcome.stderr.strip()[:2000]}"
        return _error_envelope(model, detail, None)
    if not isinstance(data, dict):
        return _error_envelope(model, "CLI result is not an object", None)
    session_id = data.get("session_id")
    structured = data.get("structured_output")
    if data.get("is_error") or not isinstance(structured, dict):
        detail = str(data.get("result") or "no structured_output in CLI result")
        return _error_envelope(model, detail, session_id)
    return {
        "_transport": TRANSPORT_NAME,
        "_cli_session_id": session_id,
        "_cli_models_used": sorted((data.get("modelUsage") or {}).keys()),
        "id": f"msg_cli_{session_id}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "tool_use",
                "id": f"toolu_cli_{session_id}",
                "name": _TOOL_NAME,
                "input": structured,
            }
        ],
        "stop_reason": data.get("stop_reason") or "tool_use",
        "usage": data.get("usage") or {},
    }


def _is_error(envelope: dict[str, Any]) -> bool:
    return envelope.get("stop_reason") == "cli_error"


# The claude.ai subscription meters a rolling session; once the CLI reports
# the limit every further call fails the same way until the reset time, so
# retrying or asking the next case only burns wall clock.
_QUOTA_PATTERN = re.compile(r"session limit|usage limit|rate limit", re.IGNORECASE)


def _is_quota_exhausted(envelope: dict[str, Any]) -> bool:
    return _is_error(envelope) and bool(
        _QUOTA_PATTERN.search(envelope.get("_cli_error") or "")
    )


class ClaudeCliTransport:
    def __init__(
        self,
        *,
        batch_root: Path,
        workers: int = 3,
        privacy_policy: PrivacyAdmissionPolicy | None = None,
        runner: Runner | None = None,
    ) -> None:
        self._privacy_policy = privacy_policy
        self._live = runner is None
        self._root = batch_root
        self._workers = max(1, workers)
        self._runner = runner or _subprocess_runner
        # Set by the first worker that sees the subscription limit; the rest
        # of the batch then fails fast with the same message.
        self._quota_error: str | None = None
        self._quota_lock = threading.Lock()

    def submit_batch_only(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> str:
        self._admit(requests)
        manifest = canonical_request_manifest(
            requests=requests, model=model, prompt_cache_enabled=False
        )
        digest = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
        batch_id = f"cli-{digest}"
        batch_dir = self._root / batch_id
        private_directory(self._root)
        private_directory(batch_dir)
        path = batch_dir / "manifest.json"
        try:
            write_private_bytes(path, canonical_bytes(manifest))
        except FileExistsError:
            if read_private_bytes(path) != canonical_bytes(manifest):
                raise ValueError("CLI batch identity conflicts with original request")
        return batch_id

    def fetch_batch_results(
        self,
        batch_id: str,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        if Path(batch_id).name != batch_id or batch_id in {".", ".."}:
            raise ValueError("invalid CLI batch identifier")
        batch_dir = self._root / batch_id
        if not batch_dir.is_dir():
            raise FileNotFoundError(
                f"no CLI batch directory for {batch_id!r}: {batch_dir}"
            )

        self._admit(requests)
        expected = canonical_request_manifest(
            requests=requests, model=model, prompt_cache_enabled=False
        )
        manifest = json.loads(read_private_bytes(batch_dir / "manifest.json"))
        if canonical_bytes(manifest) != canonical_bytes(expected):
            raise ValueError("request manifest drift; refusing cached CLI results")
        payloads = {row["audit_id"]: row["payload"] for row in manifest["rows"]}
        for request in requests:
            result_path = self._result_path(batch_dir, request)
            if result_path.exists():
                record = json.loads(read_private_bytes(result_path))
                if canonical_bytes(record["request"]) != canonical_bytes(
                    payloads[request.audit_id]
                ):
                    raise ValueError("cached request differs from original manifest")
        pending = [r for r in requests if not self._result_path(batch_dir, r).exists()]
        print(
            f"  claude-cli: {len(requests) - len(pending)} answers on disk, "
            f"{len(pending)} to ask ({self._workers} workers)"
        )
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            failed = dict(
                pool.map(
                    lambda r: (r.audit_id, self._run_one(batch_dir, model, r)), pending
                )
            )

        results = tuple(
            failed[r.audit_id]
            if failed.get(r.audit_id) is not None
            else self._load_result(batch_dir, model, r)
            for r in requests
        )
        if self._quota_error:
            print(
                f"  claude-cli: stopped early, subscription limit: {self._quota_error}"
            )
        return RawBatchResponse(batch_id=batch_id, results=results)

    def submit_batch(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        batch_id = self.submit_batch_only(
            model=model, requests=requests, prompt_cache_enabled=prompt_cache_enabled
        )
        return self.fetch_batch_results(
            batch_id,
            model=model,
            requests=requests,
            prompt_cache_enabled=prompt_cache_enabled,
        )

    def _admit(self, requests: Sequence[BatchSubmissionRequest]) -> None:
        if not self._live:
            return
        if self._privacy_policy is None:
            raise PrivacyAdmissionError(
                "live CLI requires a configured privacy trust root"
            )
        for request in requests:
            self._privacy_policy.verify(request)

    @staticmethod
    def _result_path(batch_dir: Path, request: BatchSubmissionRequest) -> Path:
        return batch_dir / f"{request.audit_id}.json"

    def _run_one(
        self, batch_dir: Path, model: str, request: BatchSubmissionRequest
    ) -> BatchSubmissionResult | None:
        """Ask the CLI for one case.

        A successful answer is checkpointed and the caller reads it back.
        An error is returned for local review while its inflight marker
        prevents repeating an execution whose remote outcome is unknown.
        """
        payload = build_anthropic_request(
            request, model=model, prompt_cache_enabled=False
        )
        result_path = self._result_path(batch_dir, request)
        if result_path.exists():
            return None
        intent_path = result_path.with_suffix(".inflight")
        try:
            write_private_bytes(intent_path, canonical_bytes(payload))
        except FileExistsError as exc:
            raise ValueError(
                "CLI attempt outcome is uncertain; reconcile before resubmitting"
            ) from exc
        if result_path.exists():
            unlink_private(intent_path)
            return None
        started = datetime.now(UTC)
        t0 = time.monotonic()
        if self._quota_error:
            envelope = _error_envelope(model, self._quota_error, None)
            executed = False
        else:
            argv, stdin = build_cli_argv(model=model, payload=payload)
            envelope = _envelope_from_cli(model, self._runner(argv, stdin))
            executed = True
            if _is_quota_exhausted(envelope):
                with self._quota_lock:
                    self._quota_error = envelope["_cli_error"]
        latency_ms = int((time.monotonic() - t0) * 1000)
        record = {
            "response": envelope,
            "request": payload,
            "request_timestamp": started.isoformat(),
            "latency_ms": latency_ms,
            "attempts": int(executed),
        }
        print(
            f"    {request.audit_id}: {envelope.get('stop_reason')} in {latency_ms / 1000:.0f}s"
        )
        if _is_error(envelope):
            if not executed:
                unlink_private(intent_path)
            return self._result_from_record(model, request, record)
        write_private_bytes(result_path, canonical_bytes(record))
        unlink_private(intent_path)
        return None

    def _load_result(
        self, batch_dir: Path, model: str, request: BatchSubmissionRequest
    ) -> BatchSubmissionResult:
        record = json.loads(read_private_bytes(self._result_path(batch_dir, request)))
        return self._result_from_record(model, request, record)

    @staticmethod
    def _result_from_record(
        model: str, request: BatchSubmissionRequest, record: dict[str, Any]
    ) -> BatchSubmissionResult:
        return BatchSubmissionResult(
            custom_id=request.audit_id,
            model_id=record["request"]["model"],
            raw_response_json=record["response"],
            request_json=record["request"],
            response_headers={"anthropic-version": TRANSPORT_NAME},
            request_timestamp=datetime.fromisoformat(record["request_timestamp"]),
            latency_ms=record["latency_ms"],
            anthropic_version=TRANSPORT_NAME,
            prompt_cache_id=None,
            extended_thinking_blocks=None,
        )


__all__ = ("ClaudeCliTransport", "CliOutcome", "build_cli_argv")
