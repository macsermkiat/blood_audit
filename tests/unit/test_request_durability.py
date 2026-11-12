from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bba.llm_client.request_identity import RequestManifestStore


def test_manifest_storage_is_private_even_under_permissive_umask(tmp_path: Path):
    original = os.umask(0o022)
    root = tmp_path / "manifests"
    try:
        store = RequestManifestStore(root)
        submission = store.prepare({"rows": [{"text": "synthetic clinical note"}]})
        store.bind(submission, "remote-1")
    finally:
        os.umask(original)
    assert root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in root.iterdir())


def test_unresolved_submission_cannot_be_automatically_repeated(tmp_path: Path):
    manifest = {"model": "synthetic", "rows": [{"audit_id": "a1"}]}
    first = RequestManifestStore(tmp_path)
    first.prepare(manifest)
    with pytest.raises(ValueError, match="uncertain"):
        RequestManifestStore(tmp_path).prepare(manifest)


def test_known_remote_submission_is_reused_after_restart(tmp_path: Path):
    manifest = {"model": "synthetic", "rows": [{"audit_id": "a1"}]}
    store = RequestManifestStore(tmp_path)
    submission = store.prepare(manifest)
    store.bind(submission, "remote-1")
    restarted = RequestManifestStore(tmp_path)
    same = restarted.prepare(manifest)
    assert same == submission
    assert restarted.remote_batch_id(same) == "remote-1"
    assert restarted.load_matching("remote-1", manifest) == manifest


def test_unsafe_existing_storage_and_symlinks_are_rejected(tmp_path: Path):
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="private"):
        RequestManifestStore(unsafe).prepare({"rows": [1]})
    safe = tmp_path / "safe"
    safe.mkdir(mode=0o700)
    linked = tmp_path / "link"
    linked.symlink_to(safe, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        RequestManifestStore(linked).prepare({"rows": [1]})


def test_cli_crash_after_remote_execution_is_not_repeated(tmp_path: Path):
    import runpy

    helpers = runpy.run_path("tests/unit/test_pilot_claude_cli_transport.py")
    called = []

    def runner(argv, stdin):
        called.append(stdin)
        raise RuntimeError("synthetic crash after remote execution")

    transport = helpers["ClaudeCliTransport"](
        batch_root=tmp_path,
        workers=1,
        runner=runner,
    )
    request = helpers["_request"]("a1")
    batch = transport.submit_batch_only(
        model=helpers["MODEL"],
        requests=[request],
        prompt_cache_enabled=False,
    )
    with pytest.raises(RuntimeError, match="synthetic crash"):
        transport.fetch_batch_results(
            batch,
            model=helpers["MODEL"],
            requests=[request],
            prompt_cache_enabled=False,
        )
    with pytest.raises(ValueError, match="uncertain"):
        transport.fetch_batch_results(
            batch,
            model=helpers["MODEL"],
            requests=[request],
            prompt_cache_enabled=False,
        )
    assert len(called) == 1
    manifest = tmp_path / batch / "manifest.json"
    assert json.loads(manifest.read_text())["model"] == helpers["MODEL"]
    assert manifest.stat().st_mode & 0o777 == 0o600


def test_provider_client_cannot_retry_an_uncertain_batch_creation(monkeypatch):
    from types import SimpleNamespace
    from bba.llm_client import transport

    class Client:
        def __init__(self, *, max_retries=2, **kwargs):
            self.max_retries = max_retries

    monkeypatch.setattr(
        transport, "import_module", lambda _: SimpleNamespace(Anthropic=Client)
    )
    client = transport.AnthropicBatchTransport(api_key="synthetic")._client()
    assert client.max_retries == 0


def test_real_sdk_response_loss_cannot_create_multiple_batches(tmp_path, monkeypatch):
    from bba.llm_client import transport
    from test_privacy_admission import admitted_request, policy

    import anthropic
    import httpx

    accepted = []

    def lose_response(request):
        accepted.append(request)
        raise httpx.ReadError(
            "synthetic response loss after acceptance", request=request
        )

    original = anthropic.Anthropic
    monkeypatch.setattr(
        anthropic,
        "Anthropic",
        lambda **kwargs: original(
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(lose_response)),
        ),
    )
    provider = transport.AnthropicBatchTransport(
        api_key="synthetic", privacy_policy=policy(), manifest_root=tmp_path
    )
    options = dict(
        model="synthetic", requests=[admitted_request()], prompt_cache_enabled=False
    )
    with pytest.raises(transport.AnthropicAPIError):
        provider.submit_batch_only(**options)
    with pytest.raises(ValueError, match="uncertain"):
        provider.submit_batch_only(**options)
    assert len(accepted) == 1
    assert accepted[0].url.path == "/v1/messages/batches"


def test_malformed_cli_output_preserves_uncertainty_without_reexecution(tmp_path: Path):
    import runpy

    helpers = runpy.run_path("tests/unit/test_pilot_claude_cli_transport.py")
    calls = []

    def runner(argv, stdin):
        calls.append(stdin)
        return helpers["CliOutcome"](
            stdout="incomplete output", stderr="", returncode=0
        )

    transport = helpers["ClaudeCliTransport"](
        batch_root=tmp_path, workers=1, runner=runner
    )
    request = helpers["_request"]("a1")
    batch = transport.submit_batch_only(
        model=helpers["MODEL"], requests=[request], prompt_cache_enabled=False
    )
    transport.fetch_batch_results(
        batch, model=helpers["MODEL"], requests=[request], prompt_cache_enabled=False
    )
    assert len(calls) == 1
    with pytest.raises(ValueError, match="uncertain"):
        transport.fetch_batch_results(
            batch,
            model=helpers["MODEL"],
            requests=[request],
            prompt_cache_enabled=False,
        )
    assert len(calls) == 1
