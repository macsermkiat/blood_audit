"""Open provider and checkpoint resources only after request admission succeeds."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from bba.audit_pipeline import BatchRun, BatchRunState, PostgresBatchRunStore
from bba.audit_pipeline.store import BatchRunStore
from bba.llm_client.models import (
    AnthropicTransport,
    BatchSubmissionRequest,
    RawBatchResponse,
)


class DeferredTransport:
    def __init__(self, factory: Callable[[], AnthropicTransport]) -> None:
        self._factory = factory
        self._value: AnthropicTransport | None = None

    def _get(self) -> AnthropicTransport:
        if self._value is None:
            self._value = self._factory()
        return self._value

    def submit_batch_only(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> str:
        return self._get().submit_batch_only(
            model=model, requests=requests, prompt_cache_enabled=prompt_cache_enabled
        )

    def fetch_batch_results(
        self,
        batch_id: str,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        return self._get().fetch_batch_results(
            batch_id,
            model=model,
            requests=requests,
            prompt_cache_enabled=prompt_cache_enabled,
        )

    def submit_batch(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        return self._get().submit_batch(
            model=model, requests=requests, prompt_cache_enabled=prompt_cache_enabled
        )


class DeferredBatchStore:
    def __init__(self, factory: Callable[[], BatchRunStore]) -> None:
        self._factory = factory
        self._value: BatchRunStore | None = None

    def _get(self) -> BatchRunStore:
        if self._value is None:
            value = self._factory()
            if isinstance(value, PostgresBatchRunStore):
                value.open()
            self._value = value
        return self._value

    def create(self, run: BatchRun) -> None:
        self._get().create(run)

    def get(self, batch_id: str) -> BatchRun:
        return self._get().get(batch_id)

    def update(self, run: BatchRun) -> None:
        self._get().update(run)

    def list_by_state(self, state: BatchRunState) -> tuple[BatchRun, ...]:
        return self._get().list_by_state(state)

    def list_all(self) -> tuple[BatchRun, ...]:
        return self._get().list_all()

    def close(self) -> None:
        if isinstance(self._value, PostgresBatchRunStore):
            self._value.close()
