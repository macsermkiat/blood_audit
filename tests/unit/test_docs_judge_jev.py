"""Tests for scripts/docs_judge/jev.py (TypeSafe transport seam + cache).

WHY these tests exist:
- The judge must be re-runnable without re-paying or drifting: an identical
  (model, state, questions) request must hit the on-disk cache, and a changed
  question must miss it. If the cache key ignored the questions, a re-judge
  after editing a prompt would silently return stale answers and the
  "same question set, once" re-judge discipline would be meaningless.
- 429/5xx must retry with backoff and honor retry-after; a 4xx other than
  429 must fail loud, never be retried into a cache entry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from docs_judge.jev import JevClient, JevError, TransportResponse  # noqa: E402

QUESTIONS = {"q": {"type": "noul", "instructions": "Is `text` puffery?"}}


def _ok(answer: float) -> TransportResponse:
    body = {"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": answer}}}
    return TransportResponse(status=200, body=json.dumps(body).encode(), headers={})


def test_identical_request_is_served_from_cache_without_transport(
    tmp_path: Path,
) -> None:
    calls: list[bytes] = []

    def transport(payload: bytes) -> TransportResponse:
        calls.append(payload)
        return _ok(0.9)

    client = JevClient(
        api_key="k", cache_path=tmp_path / "cache.jsonl", transport=transport
    )
    first = client.ask({"text": "x"}, QUESTIONS)
    second = client.ask({"text": "x"}, QUESTIONS)
    assert first == second
    assert len(calls) == 1
    # A fresh client over the same cache file still hits the cache.
    reopened = JevClient(
        api_key="k", cache_path=tmp_path / "cache.jsonl", transport=transport
    )
    assert reopened.ask({"text": "x"}, QUESTIONS)["answers"]["q"]["noul"] == 0.9
    assert len(calls) == 1


def test_changed_question_text_misses_cache(tmp_path: Path) -> None:
    calls: list[bytes] = []

    def transport(payload: bytes) -> TransportResponse:
        calls.append(payload)
        return _ok(0.1)

    client = JevClient(
        api_key="k", cache_path=tmp_path / "c.jsonl", transport=transport
    )
    client.ask({"text": "x"}, QUESTIONS)
    changed = {"q": {"type": "noul", "instructions": "Is `text` promotional?"}}
    client.ask({"text": "x"}, changed)
    assert len(calls) == 2


def test_payload_carries_model_state_questions_and_auth_is_not_in_body(
    tmp_path: Path,
) -> None:
    seen: list[dict[str, object]] = []

    def transport(payload: bytes) -> TransportResponse:
        seen.append(json.loads(payload))
        return _ok(0.5)

    client = JevClient(
        api_key="secret", cache_path=tmp_path / "c.jsonl", transport=transport
    )
    client.ask({"text": "x"}, QUESTIONS)
    assert seen[0] == {
        "model": "jev-latest",
        "state": {"text": "x"},
        "questions": QUESTIONS,
    }
    assert "secret" not in json.dumps(seen[0])


def test_rate_limit_retries_with_retry_after_then_succeeds(tmp_path: Path) -> None:
    statuses = iter([429, 503, 200])
    sleeps: list[float] = []

    def transport(payload: bytes) -> TransportResponse:
        status = next(statuses)
        if status == 200:
            return _ok(0.3)
        return TransportResponse(
            status=status, body=b"busy", headers={"retry-after": "2"}
        )

    client = JevClient(
        api_key="k",
        cache_path=tmp_path / "c.jsonl",
        transport=transport,
        sleep=sleeps.append,
    )
    assert client.ask({"text": "x"}, QUESTIONS)["answers"]["q"]["noul"] == 0.3
    assert sleeps == [2.0, 2.0]


def test_non_retryable_error_fails_loud_and_is_not_cached(tmp_path: Path) -> None:
    def transport(payload: bytes) -> TransportResponse:
        return TransportResponse(status=400, body=b"bad question", headers={})

    client = JevClient(
        api_key="k", cache_path=tmp_path / "c.jsonl", transport=transport
    )
    with pytest.raises(JevError, match="400"):
        client.ask({"text": "x"}, QUESTIONS)
    assert (
        not (tmp_path / "c.jsonl").exists() or (tmp_path / "c.jsonl").read_text() == ""
    )


def test_retries_are_bounded(tmp_path: Path) -> None:
    def transport(payload: bytes) -> TransportResponse:
        return TransportResponse(status=500, body=b"down", headers={})

    client = JevClient(
        api_key="k",
        cache_path=tmp_path / "c.jsonl",
        transport=transport,
        sleep=lambda s: None,
        max_attempts=3,
    )
    with pytest.raises(JevError, match="3 attempts"):
        client.ask({"text": "x"}, QUESTIONS)
