"""Weekly clinical-reviewer sample (PRD §18 / issue #27).

Draws 50–75 random audit rows per ISO week for human reviewer inspection.
Determinism is the load-bearing property: the same ``(week_iso, sample_size,
seed)`` tuple ALWAYS produces the same ``audit_ids`` so a historical sample
can be re-derived from the manifest alone, without RNG-state replay.

Operational, not clinical: this module reads audit rows from
:mod:`bba.audit_store` but never inspects classification logic. The sample
is a uniform-random draw from the row population, NOT a stratified or
prevalence-weighted sample — those are eval-harness concerns (#20).
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from bba.monitoring.models import (
    WEEKLY_REVIEWER_SAMPLE_MAX,
    WEEKLY_REVIEWER_SAMPLE_MIN,
    WeeklyReviewerSample,
)


def _extract_audit_ids(audit_rows: Sequence[Any]) -> list[str]:
    """Pull ``audit_id`` off each row.

    Accepts both :class:`bba.audit_store.AuditRow` (production) and bare
    strings (test fixtures): rows with an ``audit_id`` attribute use it;
    everything else is stringified. The duck-typed accessor keeps the
    test surface narrow without forcing fixtures to construct full
    AuditRows for a sampling-only test.
    """
    ids: list[str] = []
    for row in audit_rows:
        audit_id = getattr(row, "audit_id", None)
        ids.append(audit_id if audit_id is not None else str(row))
    return ids


def _stable_seed(*, week_iso: str, sample_size: int, seed: int) -> int:
    """Derive a deterministic integer RNG seed from (week_iso, sample_size, seed).

    Plain ``hash()`` is process-dependent on PYTHONHASHSEED — using it
    would silently break the cross-process determinism the sampler
    promises. SHA-256 of the canonical payload is stable across
    interpreters and machines; the first 8 bytes are folded into a 64-bit
    int that ``random.Random`` accepts.
    """
    payload = f"{week_iso}|{sample_size}|{seed}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def draw_weekly_reviewer_sample(
    audit_rows: Sequence[Any],
    *,
    week_iso: str,
    sample_size: int,
    seed: int,
) -> WeeklyReviewerSample:
    """Draw a deterministic weekly reviewer sample.

    ``sample_size`` MUST be in ``[WEEKLY_REVIEWER_SAMPLE_MIN,
    WEEKLY_REVIEWER_SAMPLE_MAX]`` (50–75). The RNG seed is derived from
    ``(week_iso, sample_size, seed)`` via SHA-256 so the same inputs
    produce the same draw across processes, machines, and Python
    interpreter invocations.

    Raises :class:`ValueError` if ``sample_size`` is out of range or
    exceeds the available population.
    """
    if not (WEEKLY_REVIEWER_SAMPLE_MIN <= sample_size <= WEEKLY_REVIEWER_SAMPLE_MAX):
        raise ValueError(
            f"sample_size must be in [{WEEKLY_REVIEWER_SAMPLE_MIN}, "
            f"{WEEKLY_REVIEWER_SAMPLE_MAX}] (got {sample_size!r})"
        )
    ids = _extract_audit_ids(audit_rows)
    if sample_size > len(ids):
        raise ValueError(f"sample_size ({sample_size}) exceeds population ({len(ids)})")
    rng = random.Random(
        _stable_seed(week_iso=week_iso, sample_size=sample_size, seed=seed)
    )
    # ``random.sample`` order is deterministic for a given RNG state +
    # population order. We preserve that order in the manifest so the
    # tuple itself encodes both *which* rows and *what order* they were
    # surfaced to the reviewer.
    chosen = rng.sample(ids, sample_size)
    return WeeklyReviewerSample(
        week_iso=week_iso,
        sample_size=sample_size,
        seed=seed,
        audit_ids=tuple(chosen),
        drawn_at=datetime.now(UTC),
    )


__all__ = ("draw_weekly_reviewer_sample",)
