"""Agreement-based confidence: Sonnet x 3 reshuffled-few-shot vote.

PRD §14 / user-story #40: "agreement-based confidence: Sonnet x 3 with
reshuffled few-shot, confidence = fraction agreeing". This module is the
pure-Python piece that consumes the three classification labels and
produces an :class:`AgreementResult`. The actual prompt reshuffle and
LLM dispatch live in :mod:`bba.prompt_builder` and
:mod:`bba.llm_client`; this module only handles seed generation +
vote tabulation so the test surface stays deterministic.

Tie-breaking is first-seen: on a 1-1-1 split, the classification that
appears first in the input wins. This gives the audit pipeline a stable
deterministic verdict on adversarial 3-way ties (PRD §14
"deterministic seed control").
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from bba.confidence_calibrator.exceptions import InvalidCalibrationDataError
from bba.confidence_calibrator.models import (
    DEFAULT_AGREEMENT_RUNS,
    AgreementResult,
)


def shuffle_seeds(
    base_seed: int,
    n_runs: int = DEFAULT_AGREEMENT_RUNS,
) -> tuple[int, ...]:
    """Deterministic seed sequence for ``n_runs`` few-shot shufflings.

    Same ``base_seed`` + ``n_runs`` -> same returned tuple, so a re-run
    of the monthly audit reproduces the exact 3 classifications. Seeds
    are derived from ``sha256(f"{base_seed}:{i}")`` (not ``base_seed +
    i``) so a small change in ``base_seed`` does not produce three
    near-correlated shufflings.

    ``n_runs < 1`` or negative ``base_seed`` raises
    :class:`InvalidCalibrationDataError`.
    """
    if base_seed < 0:
        raise InvalidCalibrationDataError(
            f"base_seed must be non-negative; got {base_seed}",
        )
    if n_runs < 1:
        raise InvalidCalibrationDataError(
            f"n_runs must be >= 1; got {n_runs}",
        )
    seeds: list[int] = []
    for i in range(n_runs):
        digest = hashlib.sha256(f"{base_seed}:{i}".encode("ascii")).digest()
        seeds.append(int.from_bytes(digest[:4], "big"))
    return tuple(seeds)


def agreement_confidence(
    classifications: Sequence[str],
) -> AgreementResult:
    """Tabulate the agreement-based confidence verdict.

    ``confidence`` is the count of the majority classification divided
    by ``len(classifications)``. Empty input raises
    :class:`InvalidCalibrationDataError`. Three-way ties resolve to the
    first-seen classification.
    """
    if not classifications:
        raise InvalidCalibrationDataError(
            "classifications must be non-empty; agreement-based confidence "
            "requires at least one run",
        )
    cls_tuple = tuple(classifications)
    counts: dict[str, int] = {}
    for c in cls_tuple:
        counts[c] = counts.get(c, 0) + 1
    max_count = max(counts.values())
    # First-seen tie-break: iterate in original order, pick the first
    # classification that hits the max count. dict insertion order is
    # preserved in Python 3.7+, but we walk the input tuple directly
    # so the contract is independent of dict implementation.
    majority = next(c for c in cls_tuple if counts[c] == max_count)
    return AgreementResult(
        classifications=cls_tuple,
        majority=majority,
        agreement_count=max_count,
        confidence=max_count / len(cls_tuple),
    )
