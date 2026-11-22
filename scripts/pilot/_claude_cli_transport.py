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
  time. Each outcome, an answer or an error, is checkpointed and never asked
  again, because an errored call may still have been served. The exception
  is an error of a systemic class (see ``_SYSTEMIC_ERROR_CLASSES``): it is
  held in memory, and saved only once a later call shows the CLI serves
  (or the batch ends without the breaker tripping). If the breaker trips or
  Ctrl+C ends the run, held outcomes are dropped and the resume asks them. An inflight
  marker without an outcome means a run died mid-call; the batch then
  refuses to start and prints the command that clears the marker. Resume
  uses ``BBA_PILOT_BATCH_ID=<batch_id>`` under the original
  ``BBA_PILOT_RUN_ID`` and exact request configuration.
* When the CLI reports the subscription limit every worker waits and the
  case is asked again, backing off up to a wait budget. Past the budget, or
  after ``MAX_CONSECUTIVE_CLI_ERRORS`` errors in a row that point at the CLI
  rather than one case, no further case is asked and
  :class:`CliBatchStopped` is raised. A refused case, a held systemic error
  at the stop, or a call ended by Ctrl+C saves nothing, so the resume asks exactly the cases without an
  outcome.
* Each failed attempt leaves ``errors/<audit_id>.<time>.<attempt>.json``
  with the error class, exit code and duration, never prompt, stderr or
  response text.

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
import math
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
THROTTLE_BACKOFF_FIRST_S = 60.0
THROTTLE_BACKOFF_MAX_S = 1800.0
# The claude.ai limit window is five hours, so a six-hour budget lets an
# unattended run outlast one window per throttle episode.
THROTTLE_WAIT_S = 6 * 3600.0
MAX_CONSECUTIVE_CLI_ERRORS = 5


class CliBatchStopped(RuntimeError):
    """The batch stopped before asking every case; answered cases are on disk."""


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


# Observed CLI wording: "You've hit your session limit · resets 11:30am"
# (September 2026) and "You've hit your 5-hour limit · resets 3:30am
# (Asia/Bangkok)" (January 2025 v2 run).
_THROTTLE_PATTERN = re.compile(
    r"hit your [\w -]*limit|session limit|usage limit|rate limit|rate_limit_error",
    re.IGNORECASE,
)
# ASCII only: the reset time is copied into summaries, so it must never pick
# up note text.
_RESETS_PATTERN = re.compile(
    r"resets (\d{1,2}(?::\d{2})?\s?[ap]m(?: \([A-Za-z]+(?:/[A-Za-z_+-]+)?\))?)",
    re.IGNORECASE | re.ASCII,
)
# Failures that point at the CLI or the account rather than at one case. A
# run of them stops the batch; per-case failures (no structured output,
# structured-output retries exhausted, timeout) do not.
_SYSTEMIC_ERROR_CLASSES = frozenset(
    {"non_json_output", "non_object_output", "cli_is_error"}
)
# Per-case failures that still prove the model was reached: they end a run of
# systemic errors. A timeout proves nothing either way.
_SERVED_ERROR_CLASSES = frozenset({"structured_output_retries", "no_structured_output"})
_INTERRUPTED = "interrupted"


def _failure_class(text: str, fallback: str) -> str:
    return "throttled" if _THROTTLE_PATTERN.search(text) else fallback


def _resets(envelope: dict[str, Any]) -> str | None:
    match = _RESETS_PATTERN.search(envelope.get("_cli_error") or "")
    return match.group(1) if match else None


def _error_envelope(
    model: str, detail: str, session_id: str | None, error_class: str
) -> dict[str, Any]:
    # Same shape as transport._result_from_batch_entry's error path: no
    # content, so the parser raises EMPTY_RESPONSE and the row routes to
    # NEEDS_REVIEW with the detail preserved for the reviewer.
    return {
        "_transport": TRANSPORT_NAME,
        "_cli_error": detail,
        "_cli_error_class": error_class,
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
        return _error_envelope(
            model, detail, None, _failure_class(outcome.stderr, "non_json_output")
        )
    if not isinstance(data, dict):
        return _error_envelope(
            model, "CLI result is not an object", None, "non_object_output"
        )
    session_id = data.get("session_id")
    structured = data.get("structured_output")
    detail = str(data.get("result") or "no structured_output in CLI result")
    if data.get("is_error"):
        # Only an error result is matched: an answer without structured
        # output is model prose, which may quote the evidence.
        fallback = (
            "structured_output_retries"
            if data.get("subtype") == "error_max_structured_output_retries"
            else "cli_is_error"
        )
        return _error_envelope(
            model, detail, session_id, _failure_class(detail, fallback)
        )
    if not isinstance(structured, dict):
        return _error_envelope(model, detail, session_id, "no_structured_output")
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


def _uncertain_message(markers: Sequence[Path]) -> str:
    paths = " ".join(f"'{marker}'" for marker in markers)
    return (
        f"CLI outcome uncertain for {len(markers)} case(s): a run stopped "
        "mid-call without saving the outcome, and the call may have been "
        "served. To ask them again, make sure no other run uses this batch, "
        f"then run: rm {paths}"
    )


class _BatchGate:
    """Shared by the workers: the throttle backoff and the batch stop reason.

    The subscription limit refuses every call until it resets, so one
    backoff per throttle episode holds all workers. Calls already in flight
    when the limit is first seen join that backoff instead of doubling it.
    """

    def __init__(
        self,
        *,
        wait_budget_s: float,
        clock: Callable[[], float],
        sleep: Callable[[float], object] | None,
    ) -> None:
        self._lock = threading.Lock()
        self._budget = wait_budget_s
        self._clock = clock
        # The default sleep is the stop event, so stop() ends a backoff early.
        self._stopped = threading.Event()
        self._sleep = sleep or self._stopped.wait
        self._backoff = 0.0
        self._waited = 0.0
        self._resume_at = -math.inf
        self._last_throttle = -math.inf
        self._errors_in_a_row = 0
        self.stop_reason: str | None = None

    def wait_turn(self) -> bool:
        while True:
            with self._lock:
                if self.stop_reason:
                    return False
                delay = self._resume_at - self._clock()
            if delay <= 0:
                return True
            self._sleep(delay)

    def stop(self, reason: str) -> None:
        with self._lock:
            self.stop_reason = self.stop_reason or reason
        self._stopped.set()

    @property
    def interrupted(self) -> bool:
        return self.stop_reason == _INTERRUPTED

    def answered(self) -> None:
        with self._lock:
            self._backoff = 0.0
            self._waited = 0.0
            self._errors_in_a_row = 0

    def failed(self) -> None:
        with self._lock:
            self._errors_in_a_row += 1
            count = self._errors_in_a_row
        if count >= MAX_CONSECUTIVE_CLI_ERRORS:
            self.stop(f"{count} CLI errors in a row")

    def throttled(self, call_started: float, resets: str | None) -> None:
        with self._lock:
            if self.stop_reason or call_started < self._last_throttle:
                return
            self._backoff = (
                min(THROTTLE_BACKOFF_MAX_S, self._backoff * 2)
                if self._backoff
                else THROTTLE_BACKOFF_FIRST_S
            )
            reset_note = f", resets {resets}" if resets else ""
            over_budget = self._waited + self._backoff > self._budget
            if not over_budget:
                self._waited += self._backoff
                self._last_throttle = self._clock()
                self._resume_at = self._last_throttle + self._backoff
            backoff = self._backoff
        if over_budget:
            self.stop(
                f"subscription limit outlasted the {self._budget:.0f}s wait "
                f"budget{reset_note}"
            )
            return
        print(
            f"  claude-cli: subscription limit{reset_note}; retrying in {backoff:.0f}s"
        )


class ClaudeCliTransport:
    def __init__(
        self,
        *,
        batch_root: Path,
        workers: int = 3,
        privacy_policy: PrivacyAdmissionPolicy | None = None,
        runner: Runner | None = None,
        throttle_wait_s: float = THROTTLE_WAIT_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] | None = None,
    ) -> None:
        self._privacy_policy = privacy_policy
        self._live = runner is None
        self._root = batch_root
        self._workers = max(1, workers)
        self._runner = runner or _subprocess_runner
        self._clock = clock
        self._gate = _BatchGate(wait_budget_s=throttle_wait_s, clock=clock, sleep=sleep)
        self._held_lock = threading.Lock()
        self._held: list[tuple[Path, Path, dict[str, Any]]] = []

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
        markers = [
            marker
            for r in pending
            if (marker := self._marker_path(batch_dir, r)).exists()
        ]
        if markers:
            raise ValueError(_uncertain_message(markers))
        print(
            f"  claude-cli: {len(requests) - len(pending)} outcomes on disk, "
            f"{len(pending)} to ask ({self._workers} workers)"
        )
        try:
            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                try:
                    for _ in pool.map(
                        lambda r: self._run_one(batch_dir, model, r), pending
                    ):
                        pass
                except BaseException:
                    # Wake workers waiting out a backoff so the pool can shut
                    # down, and mark calls still running as ended by the
                    # interrupt.
                    self._gate.stop(_INTERRUPTED)
                    raise
        finally:
            self._settle_held(keep=not self._gate.stop_reason)

        if self._gate.stop_reason:
            saved = sum(self._result_path(batch_dir, r).exists() for r in requests)
            raise CliBatchStopped(
                f"claude-cli batch {batch_id} stopped: {self._gate.stop_reason}. "
                f"{saved}/{len(requests)} cases have a saved outcome; the resume "
                f"asks the other {len(requests) - saved}. Resume with "
                f"BBA_PILOT_BATCH_ID={batch_id} under the same BBA_PILOT_RUN_ID."
            )
        results = tuple(self._load_result(batch_dir, model, r) for r in requests)
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

    @staticmethod
    def _marker_path(batch_dir: Path, request: BatchSubmissionRequest) -> Path:
        return batch_dir / f"{request.audit_id}.inflight"

    def _run_one(
        self, batch_dir: Path, model: str, request: BatchSubmissionRequest
    ) -> None:
        """Ask the CLI for one case and save its outcome.

        An answer or an error is saved, and a saved case is never asked
        again: an errored call may still have been served. A case refused by
        the subscription limit is asked again after the backoff. If the batch
        stops first, or Ctrl+C ends the call, nothing is saved and the resume
        asks the case. The inflight marker outlives this call only when the
        process dies mid-call.
        """
        payload = build_anthropic_request(
            request, model=model, prompt_cache_enabled=False
        )
        result_path = self._result_path(batch_dir, request)
        if result_path.exists():
            return
        intent_path = self._marker_path(batch_dir, request)
        try:
            write_private_bytes(intent_path, canonical_bytes(payload))
        except FileExistsError as exc:
            raise ValueError(_uncertain_message([intent_path])) from exc
        if result_path.exists():
            unlink_private(intent_path)
            return
        started = datetime.now(UTC)
        t0 = self._clock()
        attempts = 0
        while self._gate.wait_turn():
            attempts += 1
            call_started = self._clock()
            envelope, exit_code = self._ask(model, payload)
            duration_ms = int((self._clock() - call_started) * 1000)
            error_class = envelope.get("_cli_error_class")
            label = error_class or envelope.get("stop_reason")
            print(f"    {request.audit_id}: {label} in {duration_ms / 1000:.0f}s")
            if error_class is None or error_class in _SERVED_ERROR_CLASSES:
                self._gate.answered()
                if not self._gate.stop_reason:
                    self._settle_held(keep=True)
            if error_class is not None:
                resets = _resets(envelope)
                self._write_error_summary(
                    batch_dir,
                    request.audit_id,
                    attempts,
                    error_class,
                    exit_code,
                    duration_ms,
                    resets,
                )
                if self._gate.interrupted:
                    # Ctrl+C reaches the claude child too, so this failure is
                    # the interrupt, not the case's outcome.
                    break
                if error_class == "throttled":
                    self._gate.throttled(call_started, resets)
                    continue
                if error_class in _SYSTEMIC_ERROR_CLASSES:
                    self._gate.failed()
            record = {
                "response": envelope,
                "request": payload,
                "request_timestamp": started.isoformat(),
                "latency_ms": int((self._clock() - t0) * 1000),
                "attempts": attempts,
            }
            if error_class in _SYSTEMIC_ERROR_CLASSES:
                # The CLI may not have served this case at all. Keep the
                # marker and hold the outcome until the batch shows whether
                # the CLI works; the held list is settled by the caller.
                with self._held_lock:
                    self._held.append((result_path, intent_path, record))
                return
            write_private_bytes(result_path, canonical_bytes(record))
            break
        unlink_private(intent_path)

    def _settle_held(self, *, keep: bool) -> None:
        """Save held systemic outcomes, or drop them so the resume asks again."""
        with self._held_lock:
            held, self._held = self._held, []
        for result_path, intent_path, record in held:
            if keep:
                write_private_bytes(result_path, canonical_bytes(record))
            unlink_private(intent_path)

    def _ask(
        self, model: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], int | None]:
        argv, stdin = build_cli_argv(model=model, payload=payload)
        try:
            outcome = self._runner(argv, stdin)
        except subprocess.TimeoutExpired as exc:
            detail = f"claude timed out after {exc.timeout:.0f}s"
            return _error_envelope(model, detail, None, "timeout"), None
        return _envelope_from_cli(model, outcome), outcome.returncode

    @staticmethod
    def _write_error_summary(
        batch_dir: Path,
        audit_id: str,
        attempt: int,
        error_class: str,
        exit_code: int | None,
        duration_ms: int,
        resets: str | None,
    ) -> None:
        errors_dir = batch_dir / "errors"
        private_directory(errors_dir)
        at = datetime.now(UTC)
        summary = {
            "audit_id": audit_id,
            "attempt": attempt,
            "at": at.isoformat(),
            "error_class": error_class,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "resets": resets,
        }
        name = f"{audit_id}.{at:%Y%m%dT%H%M%S%f}.{attempt}.json"
        write_private_bytes(errors_dir / name, canonical_bytes(summary))

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


__all__ = (
    "MAX_CONSECUTIVE_CLI_ERRORS",
    "THROTTLE_BACKOFF_FIRST_S",
    "THROTTLE_WAIT_S",
    "ClaudeCliTransport",
    "CliBatchStopped",
    "CliOutcome",
    "build_cli_argv",
)
