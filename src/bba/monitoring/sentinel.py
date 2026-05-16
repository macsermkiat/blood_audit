"""Weekly intra-model κ on a 200-case sentinel set (PRD §18).

The sentinel set is constructed ONCE with a deterministic seed (default 42)
and re-run weekly through :mod:`bba.audit_pipeline` to verify the model's
output is stable week-over-week. Cohen's κ + Gwet's AC1 are computed
between consecutive runs over the paired ``Classification`` labels; an
alarm fires when ``κ < SENTINEL_KAPPA_ALARM_THRESHOLD`` (0.90).

Metric implementations are imported from :mod:`bba.eval_harness.agreement`
— this module does NOT re-implement κ or AC1.

Operational, not clinical: the sentinel comparison only cares about the
classification label, never about the underlying clinical reasoning.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from bba.audit_store import Classification
from bba.eval_harness.agreement import cohen_kappa, gwet_ac1
from bba.monitoring.exceptions import InsufficientHistoryError
from bba.monitoring.models import (
    SENTINEL_KAPPA_ALARM_THRESHOLD,
    SENTINEL_SET_SEED,
    SENTINEL_SET_SIZE,
    SentinelComparison,
    SentinelManifest,
)


def _extract_audit_ids(audit_rows: Sequence[Any]) -> list[str]:
    """Pull ``audit_id`` off each row; accepts AuditRow or bare strings
    (test-fixture convenience)."""
    ids: list[str] = []
    for row in audit_rows:
        audit_id = getattr(row, "audit_id", None)
        ids.append(audit_id if audit_id is not None else str(row))
    return ids


def _stable_seed(*, size: int, seed: int) -> int:
    """Stable RNG seed for sentinel construction; SHA-256-based so
    cross-process determinism survives PYTHONHASHSEED randomization."""
    payload = f"sentinel|{size}|{seed}".encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        byteorder="big",
        signed=False,
    )


def build_sentinel_manifest(
    audit_rows: Sequence[Any],
    *,
    size: int = SENTINEL_SET_SIZE,
    seed: int = SENTINEL_SET_SEED,
) -> SentinelManifest:
    """Build the 200-case sentinel manifest.

    Deterministic: the same ``audit_rows`` + ``seed`` + ``size`` always
    produces the same ``audit_ids``. Designed to be called ONCE per
    deployment; the manifest is then persisted (e.g., via
    :class:`bba.monitoring.MonitoringStore`) and reused.

    Raises :class:`ValueError` if ``size`` exceeds the row population.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size!r}")
    ids = _extract_audit_ids(audit_rows)
    if size > len(ids):
        raise ValueError(
            f"size ({size}) exceeds population ({len(ids)})"
        )
    rng = random.Random(_stable_seed(size=size, seed=seed))
    chosen = rng.sample(ids, size)
    return SentinelManifest(
        size=size,
        seed=seed,
        audit_ids=tuple(chosen),
        built_at=datetime.now(UTC),
    )


def evaluate_sentinel_run(
    *,
    manifest: SentinelManifest,
    previous: Mapping[str, Classification],
    current: Mapping[str, Classification],
    kappa_threshold: float = SENTINEL_KAPPA_ALARM_THRESHOLD,
) -> SentinelComparison:
    """Compare last week's run to this week's run on the sentinel set.

    Only audit_ids present in BOTH ``previous`` AND ``current`` AND
    ``manifest.audit_ids`` are paired. κ and AC1 are computed via
    :func:`bba.eval_harness.agreement.cohen_kappa` and
    :func:`bba.eval_harness.agreement.gwet_ac1` — NOT re-implemented.

    Raises :class:`InsufficientHistoryError` when ``previous`` is empty
    or no audit_id pairs survive the manifest intersection — both mean
    there is nothing to compute κ against.
    """
    if not previous:
        raise InsufficientHistoryError(
            "previous run is empty; cannot evaluate sentinel κ "
            "without a prior week's classifications"
        )
    manifest_set = set(manifest.audit_ids)
    paired_ids = [
        aid for aid in manifest.audit_ids
        if aid in manifest_set and aid in previous and aid in current
    ]
    if not paired_ids:
        raise InsufficientHistoryError(
            "no audit_ids appear in BOTH previous and current runs "
            "AND the sentinel manifest; nothing to compute κ against"
        )
    prev_labels = [previous[aid] for aid in paired_ids]
    curr_labels = [current[aid] for aid in paired_ids]
    kappa = cohen_kappa(prev_labels, curr_labels)
    ac1 = gwet_ac1(prev_labels, curr_labels)
    return SentinelComparison(
        n_paired=len(paired_ids),
        cohen_kappa=kappa,
        gwet_ac1=ac1,
        kappa_threshold=kappa_threshold,
        alarm_fired=kappa < kappa_threshold,
    )


__all__ = (
    "build_sentinel_manifest",
    "evaluate_sentinel_run",
)
