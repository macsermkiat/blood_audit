"""Pure confusion-matrix + before/after comparison for the peri-op fix.

Given the human labels (``reqno -> classification``) and a run's per-case
:class:`CaseVerdict`s, build the bucketed confusion matrix — split by
mechanism — and compare two runs to surface the two honest costs.

Only cases present in BOTH the labels and the run are scored; the
intersection is reported so a caller can see how many of the 300 a run
actually covered (a run over a data subset must not silently look complete).
"""

from __future__ import annotations

from collections.abc import Mapping

from bba.verification.models import (
    BUCKETS,
    CaseVerdict,
    ConfusionCell,
    ConfusionMatrix,
    MatrixScope,
    RunComparison,
    bucket_of,
)


def _scored_reqnos(
    labels: Mapping[str, str], verdicts: Mapping[str, CaseVerdict]
) -> list[str]:
    """REQNOs present in both the labels and the run, in sorted order
    (deterministic output regardless of dict insertion order)."""
    return sorted(set(labels) & set(verdicts))


def build_matrix(
    labels: Mapping[str, str],
    verdicts: Mapping[str, CaseVerdict],
    *,
    scope: MatrixScope = "all",
) -> ConfusionMatrix:
    """Build the full 3×3 confusion matrix for ``scope``.

    ``scope="all"`` pools both legs; ``"deterministic"`` / ``"llm"`` restrict
    to cases that leg decided. Every one of the 9 cells is emitted (zeros
    included) so the grid is always complete.
    """
    counts: dict[tuple[str, str], int] = {}
    for reqno in _scored_reqnos(labels, verdicts):
        verdict = verdicts[reqno]
        if scope != "all" and verdict.mechanism != scope:
            continue
        key = (bucket_of(labels[reqno]), bucket_of(verdict.classification))
        counts[key] = counts.get(key, 0) + 1
    cells = tuple(
        ConfusionCell(truth=truth, predicted=predicted, count=counts.get((truth, predicted), 0))
        for truth in BUCKETS
        for predicted in BUCKETS
    )
    return ConfusionMatrix(scope=scope, cells=cells)


def confusion_by_mechanism(
    labels: Mapping[str, str], verdicts: Mapping[str, CaseVerdict]
) -> tuple[ConfusionMatrix, ...]:
    """Return the (deterministic, llm, all) matrices for one run."""
    return (
        build_matrix(labels, verdicts, scope="deterministic"),
        build_matrix(labels, verdicts, scope="llm"),
        build_matrix(labels, verdicts, scope="all"),
    )


def find_regressions(
    labels: Mapping[str, str],
    before: Mapping[str, CaseVerdict],
    after: Mapping[str, CaseVerdict],
) -> tuple[str, ...]:
    """REQNOs a truly-appropriate order was correctly cleared before the fix
    but is no longer after it.

    Scoped to cases the human labelled ``appropriate`` and that BOTH runs
    covered: a regression is ``before`` bucket == appropriate (correct) and
    ``after`` bucket != appropriate. Sorted for deterministic output."""
    regressions: list[str] = []
    for reqno in sorted(set(labels) & set(before) & set(after)):
        if bucket_of(labels[reqno]) != "appropriate":
            continue
        was_correct = bucket_of(before[reqno].classification) == "appropriate"
        now_wrong = bucket_of(after[reqno].classification) != "appropriate"
        if was_correct and now_wrong:
            regressions.append(reqno)
    return tuple(regressions)


def _llm_volume(
    labels: Mapping[str, str], verdicts: Mapping[str, CaseVerdict]
) -> int:
    """Count of scored cases a run routed to the LLM leg."""
    return sum(
        1
        for reqno in _scored_reqnos(labels, verdicts)
        if verdicts[reqno].mechanism == "llm"
    )


def compare_runs(
    labels: Mapping[str, str],
    before: Mapping[str, CaseVerdict],
    after: Mapping[str, CaseVerdict],
) -> RunComparison:
    """Full before/after comparison: per-mechanism matrices for each run plus
    the two costs (regressions, LLM-volume delta)."""
    return RunComparison(
        before=confusion_by_mechanism(labels, before),
        after=confusion_by_mechanism(labels, after),
        regressions=find_regressions(labels, before, after),
        llm_volume_before=_llm_volume(labels, before),
        llm_volume_after=_llm_volume(labels, after),
    )


__all__ = (
    "build_matrix",
    "compare_runs",
    "confusion_by_mechanism",
    "find_regressions",
)
