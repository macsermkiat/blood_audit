"""Thin TypeSafe System One client with an exact-request cache and retries.

The transport is injectable so tests never touch the network, and so a
dry run can substitute canned answers. The cache key covers model, state,
and questions: any change to a question text is a cache miss by design.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
_RETRYABLE = frozenset({0, 429, 500, 502, 503, 504})


class JevError(RuntimeError):
    """Raised when the API returns a non-retryable error or retries run out."""


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]


Transport = Callable[[bytes], TransportResponse]


def _http_transport(api_key: str, endpoint: str, timeout: float) -> Transport:
    def send(payload: bytes) -> TransportResponse:
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return TransportResponse(
                    status=response.status,
                    body=response.read(),
                    headers={k.lower(): v for k, v in response.headers.items()},
                )
        except urllib.error.HTTPError as err:
            return TransportResponse(
                status=err.code,
                body=err.read(),
                headers={k.lower(): v for k, v in err.headers.items()},
            )
        except urllib.error.URLError as err:
            return TransportResponse(status=0, body=str(err).encode(), headers={})

    return send


def _cache_key(model: str, state: Any, questions: Any) -> str:
    canonical = json.dumps(
        {"model": model, "state": state, "questions": questions},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class JevClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        cache_path: Path | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 5,
        endpoint: str = ENDPOINT,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._cache_path = cache_path
        self._transport = transport or _http_transport(api_key, endpoint, timeout)
        self._sleep = sleep
        self._max_attempts = max_attempts
        self._lock = threading.Lock()
        self._cache: dict[str, Mapping[str, Any]] = self._load_cache()

    def _load_cache(self) -> dict[str, Mapping[str, Any]]:
        if self._cache_path is None or not self._cache_path.exists():
            return {}
        loaded: dict[str, Mapping[str, Any]] = {}
        for line in self._cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            loaded[entry["key"]] = entry["response"]
        return loaded

    def _store(self, key: str, response: Mapping[str, Any]) -> None:
        with self._lock:
            self._cache[key] = response
            if self._cache_path is not None:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self._cache_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {"key": key, "response": response}, ensure_ascii=False
                        )
                        + "\n"
                    )

    def ask(
        self, state: Mapping[str, Any], questions: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Evaluate ``questions`` against ``state``; cached by exact request."""
        key = _cache_key(self._model, state, questions)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        payload = json.dumps(
            {"model": self._model, "state": state, "questions": questions},
            ensure_ascii=False,
        ).encode()
        response = self._send_with_retries(payload)
        self._store(key, response)
        return response

    def _send_with_retries(self, payload: bytes) -> Mapping[str, Any]:
        last: TransportResponse | None = None
        for attempt in range(1, self._max_attempts + 1):
            result = self._transport(payload)
            if 200 <= result.status < 300:
                parsed: Mapping[str, Any] = json.loads(result.body.decode("utf-8"))
                return parsed
            if result.status not in _RETRYABLE:
                raise JevError(
                    f"TypeSafe API error {result.status}: {result.body[:300]!r}"
                )
            last = result
            if attempt < self._max_attempts:
                self._sleep(_backoff(result, attempt))
        status = last.status if last is not None else "?"
        raise JevError(
            f"TypeSafe API still failing ({status}) after {self._max_attempts} attempts"
        )


def _backoff(result: TransportResponse, attempt: int) -> float:
    header = result.headers.get("retry-after")
    if header is not None:
        try:
            return float(header)
        except ValueError:
            pass
    return float(2 ** (attempt - 1))
