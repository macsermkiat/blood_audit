"""Subscription-backed pilot transport: one ``claude -p`` call per request.

The pilot LLM leg normally submits an Anthropic Message Batch (API key). For
an A/B on a prompt or evidence change that should not spend API credit, the
same requests can go through the local ``claude`` CLI on the claude.ai
subscription. The transport must hand the parser a response envelope shaped
like the API's (a ``tool_use`` block named after the audit tool), persist
every answer to disk as it lands so a crash resumes instead of re-asking, and
turn a CLI failure into the same empty-content envelope the batch path uses
so the row routes to NEEDS_REVIEW rather than vanishing.

The CLI is faked here: these tests never touch the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
if str(PILOT_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_DIR))

from _claude_cli_transport import (  # noqa: E402
    CliOutcome,
    ClaudeCliTransport,
    build_cli_argv,
)
from bba.llm_client import BatchSubmissionRequest  # noqa: E402
from bba.llm_client.parser import parse_structured_response  # noqa: E402
from bba.llm_client.transport import build_anthropic_request  # noqa: E402
from bba.prompt_builder import (  # noqa: E402
    InjectionVerdict,
    PromptBlock,
    PromptBuildResult,
    build_envelope,
    compute_prompt_hash,
)

MODEL = "claude-opus-4-8"


def _request(audit_id: str) -> BatchSubmissionRequest:
    blocks = (
        PromptBlock(role="system", text="System A", cache_marker=False),
        PromptBlock(role="system", text="System B", cache_marker=True),
        PromptBlock(
            role="user",
            text=f'<evidence id="E1" untrusted="true">{audit_id}</evidence>',
            cache_marker=False,
        ),
    )
    envelope = build_envelope(
        blocks=[
            {"role": b.role, "text": b.text, "cache_marker": b.cache_marker}
            for b in blocks
        ],
        task_mode="HB_7_10_REVIEW",
        cohort_threshold=7.0,
        injection_matches=[],
        route_to_needs_review=False,
        needs_review_reasons=[],
    )
    prompt = PromptBuildResult(
        blocks=blocks,
        task_mode="HB_7_10_REVIEW",
        cohort_threshold=7.0,
        injection_verdict=InjectionVerdict(flagged=False, matches=()),
        route_to_needs_review=False,
        needs_review_reasons=(),
        prompt_hash=compute_prompt_hash(envelope),
    )
    return BatchSubmissionRequest(
        audit_id=audit_id, run_id="run-1", task_mode="HB_7_10_REVIEW", prompt=prompt
    )


def _cli_success(structured: dict[str, object]) -> CliOutcome:
    return CliOutcome(
        stdout=json.dumps(
            {
                "is_error": False,
                "structured_output": structured,
                "stop_reason": "tool_use",
                "session_id": "sess-1",
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "modelUsage": {MODEL: {}},
            }
        ),
        stderr="",
        returncode=0,
    )


class TestArgv:
    def test_argv_carries_model_schema_system_and_no_tools(self) -> None:
        payload = build_anthropic_request(
            _request("a1"), model=MODEL, prompt_cache_enabled=False
        )
        argv, stdin = build_cli_argv(model=MODEL, payload=payload)
        assert argv[:2] == ["claude", "-p"]
        assert argv[argv.index("--model") + 1] == MODEL
        assert argv[argv.index("--tools") + 1] == ""
        assert "--no-session-persistence" in argv
        schema = json.loads(argv[argv.index("--json-schema") + 1])
        assert schema == payload["tools"][0]["input_schema"]
        assert argv[argv.index("--system-prompt") + 1] == "System A\n\nSystem B"
        # The user prompt travels on stdin: evidence bundles can exceed a
        # comfortable argv, and stdin keeps them out of `ps` output.
        assert stdin == '<evidence id="E1" untrusted="true">a1</evidence>'


class TestFetchResults:
    def test_success_becomes_a_tool_use_envelope_the_parser_accepts(
        self, tmp_path: Path
    ) -> None:
        answer = {"reasoning": "Hb 6.5, symptomatic", "classification": "APPROPRIATE"}
        calls: list[tuple[list[str], str]] = []

        def runner(argv: list[str], stdin: str) -> CliOutcome:
            calls.append((argv, stdin))
            return _cli_success(answer)

        transport = ClaudeCliTransport(batch_root=tmp_path, workers=1, runner=runner)
        req = _request("a1")
        batch_id = transport.submit_batch_only(
            model=MODEL, requests=[req], prompt_cache_enabled=True
        )
        response = transport.fetch_batch_results(
            batch_id, model=MODEL, requests=[req], prompt_cache_enabled=True
        )

        assert response.batch_id == batch_id
        (result,) = response.results
        assert result.custom_id == "a1"
        assert result.model_id == MODEL
        (block,) = result.raw_response_json["content"]
        assert block["type"] == "tool_use"
        assert block["name"] == "classify_transfusion_order"
        assert dict(block["input"]) == answer
        assert result.raw_response_json["_transport"] == "claude-cli"
        assert result.request_json["model"] == MODEL
        assert len(calls) == 1
        # The parser must not be able to tell this from an API envelope.
        outcome = parse_structured_response(result)
        assert outcome.parse_failure_reason != "empty_response"

    def test_cli_error_yields_empty_content_not_an_exception(
        self, tmp_path: Path
    ) -> None:
        def runner(argv: list[str], stdin: str) -> CliOutcome:
            return CliOutcome(
                stdout=json.dumps(
                    {"is_error": True, "result": "API Error: 500", "session_id": "s"}
                ),
                stderr="",
                returncode=0,
            )

        transport = ClaudeCliTransport(
            batch_root=tmp_path, workers=1, runner=runner, retries=0
        )
        req = _request("a1")
        batch_id = transport.submit_batch_only(
            model=MODEL, requests=[req], prompt_cache_enabled=True
        )
        (result,) = transport.fetch_batch_results(
            batch_id, model=MODEL, requests=[req], prompt_cache_enabled=True
        ).results
        assert list(result.raw_response_json["content"]) == []
        assert result.raw_response_json["stop_reason"] == "cli_error"
        assert "500" in result.raw_response_json["_cli_error"]
        # Same NEEDS_REVIEW route as a failed batch entry.
        assert (
            parse_structured_response(result).parse_failure_reason == "empty_response"
        )

    def test_non_json_stdout_is_an_error_envelope(self, tmp_path: Path) -> None:
        def runner(argv: list[str], stdin: str) -> CliOutcome:
            return CliOutcome(stdout="not json", stderr="boom", returncode=1)

        transport = ClaudeCliTransport(
            batch_root=tmp_path, workers=1, runner=runner, retries=0
        )
        req = _request("a1")
        batch_id = transport.submit_batch_only(
            model=MODEL, requests=[req], prompt_cache_enabled=True
        )
        (result,) = transport.fetch_batch_results(
            batch_id, model=MODEL, requests=[req], prompt_cache_enabled=True
        ).results
        assert result.raw_response_json["stop_reason"] == "cli_error"
        assert "boom" in result.raw_response_json["_cli_error"]

    def test_transient_error_is_retried(self, tmp_path: Path) -> None:
        attempts = {"n": 0}

        def runner(argv: list[str], stdin: str) -> CliOutcome:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return CliOutcome(
                    stdout=json.dumps({"is_error": True, "result": "API Error: 500"}),
                    stderr="",
                    returncode=0,
                )
            return _cli_success({"classification": "APPROPRIATE"})

        transport = ClaudeCliTransport(
            batch_root=tmp_path, workers=1, runner=runner, retries=1, retry_delay_s=0
        )
        req = _request("a1")
        batch_id = transport.submit_batch_only(
            model=MODEL, requests=[req], prompt_cache_enabled=True
        )
        (result,) = transport.fetch_batch_results(
            batch_id, model=MODEL, requests=[req], prompt_cache_enabled=True
        ).results
        assert attempts["n"] == 2
        assert result.raw_response_json["stop_reason"] == "tool_use"

    def test_results_already_on_disk_are_not_re_asked(self, tmp_path: Path) -> None:
        asked: list[str] = []

        def runner(argv: list[str], stdin: str) -> CliOutcome:
            asked.append(stdin)
            return _cli_success({"classification": "APPROPRIATE"})

        reqs = [_request("a1"), _request("a2")]
        first = ClaudeCliTransport(batch_root=tmp_path, workers=1, runner=runner)
        batch_id = first.submit_batch_only(
            model=MODEL, requests=reqs, prompt_cache_enabled=True
        )
        first.fetch_batch_results(
            batch_id, model=MODEL, requests=reqs[:1], prompt_cache_enabled=True
        )
        assert len(asked) == 1

        # A resumed process (BBA_PILOT_BATCH_ID) asks only for the missing one.
        second = ClaudeCliTransport(batch_root=tmp_path, workers=1, runner=runner)
        response = second.fetch_batch_results(
            batch_id, model=MODEL, requests=reqs, prompt_cache_enabled=True
        )
        assert len(asked) == 2
        assert {r.custom_id for r in response.results} == {"a1", "a2"}

    def test_unknown_batch_id_raises(self, tmp_path: Path) -> None:
        transport = ClaudeCliTransport(
            batch_root=tmp_path, workers=1, runner=lambda a, s: _cli_success({})
        )
        with pytest.raises(FileNotFoundError):
            transport.fetch_batch_results(
                "cli-nope",
                model=MODEL,
                requests=[_request("a1")],
                prompt_cache_enabled=True,
            )

    def test_model_drift_is_recorded_on_the_envelope(self, tmp_path: Path) -> None:
        # The CLI may fall back to another model; the audit row must say so.
        def runner(argv: list[str], stdin: str) -> CliOutcome:
            out = json.loads(_cli_success({"classification": "APPROPRIATE"}).stdout)
            out["modelUsage"] = {"claude-haiku-4-5-20251001": {}}
            return CliOutcome(stdout=json.dumps(out), stderr="", returncode=0)

        transport = ClaudeCliTransport(batch_root=tmp_path, workers=1, runner=runner)
        req = _request("a1")
        batch_id = transport.submit_batch_only(
            model=MODEL, requests=[req], prompt_cache_enabled=True
        )
        (result,) = transport.fetch_batch_results(
            batch_id, model=MODEL, requests=[req], prompt_cache_enabled=True
        ).results
        assert list(result.raw_response_json["_cli_models_used"]) == [
            "claude-haiku-4-5-20251001"
        ]


class TestFailureIsNotCheckpointed:
    def test_error_envelope_is_not_written_so_a_resume_re_asks(
        self, tmp_path: Path
    ) -> None:
        # First process: the CLI fails. Second process (BBA_PILOT_BATCH_ID
        # resume): the same case must be asked again, not read back as done.
        outcomes = iter(
            [
                CliOutcome(
                    stdout=json.dumps({"is_error": True, "result": "API Error: 500"}),
                    stderr="",
                    returncode=0,
                ),
                _cli_success({"classification": "APPROPRIATE"}),
            ]
        )
        runner = lambda argv, stdin: next(outcomes)  # noqa: E731
        req = _request("a1")
        first = ClaudeCliTransport(
            batch_root=tmp_path, workers=1, runner=runner, retries=0
        )
        batch_id = first.submit_batch_only(
            model=MODEL, requests=[req], prompt_cache_enabled=True
        )
        (r1,) = first.fetch_batch_results(
            batch_id, model=MODEL, requests=[req], prompt_cache_enabled=True
        ).results
        assert r1.raw_response_json["stop_reason"] == "cli_error"
        assert not (tmp_path / batch_id / "a1.json").exists()

        second = ClaudeCliTransport(
            batch_root=tmp_path, workers=1, runner=runner, retries=0
        )
        (r2,) = second.fetch_batch_results(
            batch_id, model=MODEL, requests=[req], prompt_cache_enabled=True
        ).results
        assert r2.raw_response_json["stop_reason"] == "tool_use"

    def test_session_limit_fails_the_rest_of_the_batch_without_calling(
        self, tmp_path: Path
    ) -> None:
        calls: list[str] = []

        def runner(argv: list[str], stdin: str) -> CliOutcome:
            calls.append(stdin)
            return CliOutcome(
                stdout=json.dumps(
                    {
                        "is_error": True,
                        "result": "You've hit your session limit · resets 11:30am",
                    }
                ),
                stderr="",
                returncode=0,
            )

        reqs = [_request(f"a{i}") for i in range(5)]
        transport = ClaudeCliTransport(
            batch_root=tmp_path, workers=1, runner=runner, retries=2, retry_delay_s=0
        )
        batch_id = transport.submit_batch_only(
            model=MODEL, requests=reqs, prompt_cache_enabled=True
        )
        response = transport.fetch_batch_results(
            batch_id, model=MODEL, requests=reqs, prompt_cache_enabled=True
        )
        # One call revealed the limit; no retries and no further cases asked.
        assert len(calls) == 1
        assert all(
            r.raw_response_json["stop_reason"] == "cli_error" for r in response.results
        )
        assert all(
            "session limit" in r.raw_response_json["_cli_error"]
            for r in response.results
        )
        assert not any((tmp_path / batch_id).glob("a*.json"))
