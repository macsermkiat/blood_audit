"""Frozen models for the peri-op-fix verification harness.

The harness scores a pipeline run against the 300-case human review
(``bba.attribution.human_label_verdict_source``) and, crucially, compares
a *before* run to an *after* run so the peri-op fix can be judged on honest
metrics rather than a single headline number:

* the confusion matrix is computed **per mechanism** (deterministic vs LLM)
  so a headline can't hide which leg moved which cases;
* two costs are surfaced explicitly — regressions (a currently-correct
  ``APPROPRIATE`` flipped away) and the LLM-volume delta (the pre-op
  deferral raises the LLM run-rate, a real throughput cost).

All comparison happens in the 3-bucket space the human labels live in
(``appropriate`` / ``inappropriate`` / ``unresolved``); "no longer
force-cleared" (an APPROPRIATE moving to ``unresolved``) is therefore
visible as a distinct cell rather than being conflated with "correct".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bba.attribution.models import Bucket

Mechanism = Literal["deterministic", "llm"]
"""Which pipeline leg produced a case's final verdict. The deterministic
classifier answers some cases outright; the rest route to the LLM."""

MatrixScope = Literal["deterministic", "llm", "all"]
"""Scope of a :class:`ConfusionMatrix`: one mechanism, or the pooled total."""


VerificationBucket = Literal["appropriate", "inappropriate", "unresolved", "excluded"]


_BUCKET_OF: Mapping[str, VerificationBucket] = {
    "APPROPRIATE": "appropriate",
    "INAPPROPRIATE": "inappropriate",
    "NEEDS_REVIEW": "unresolved",
    "INSUFFICIENT_EVIDENCE": "unresolved",
    "POTENTIALLY_INAPPROPRIATE": "unresolved",
    "PREOP_RESERVATION_UNCONFIRMED": "unresolved",
    "RETURNED_NOT_TRANSFUSED": "excluded",
}
"""Registration of every pipeline / human classification.

``excluded`` is deliberately outside :data:`BUCKETS`, so returned orders are
registered but omitted from the scored 3×3 committee matrix.
"""

BUCKETS: tuple[Bucket, ...] = ("appropriate", "inappropriate", "unresolved")
"""Canonical bucket order for rendering a full 3×3 matrix (no missing cells)."""


def bucket_of(classification: str) -> VerificationBucket:
    """Map a classification label to its committee bucket.

    Fail-loud: an unrecognized label raises rather than silently landing in
    ``unresolved`` — a typo in a verdict source must not skew the matrix.
    """
    bucket = _BUCKET_OF.get(classification)
    if bucket is None:
        raise ValueError(
            f"unknown classification {classification!r}; expected one of "
            f"{sorted(_BUCKET_OF)}"
        )
    return bucket


class CaseVerdict(BaseModel):
    """One order's pipeline outcome: its final classification + which leg
    decided it. ``classification`` is any canonical classification label;
    the harness collapses it via :func:`bucket_of`."""

    model_config = ConfigDict(frozen=True)

    reqno: str = Field(min_length=1)
    classification: str = Field(min_length=1)
    mechanism: Mechanism


class ConfusionCell(BaseModel):
    """One (truth, predicted) cell of a bucketed confusion matrix."""

    model_config = ConfigDict(frozen=True)

    truth: Bucket
    predicted: Bucket
    count: int = Field(ge=0)


class ConfusionMatrix(BaseModel):
    """A bucketed confusion matrix over a scoped set of cases.

    ``cells`` always covers the full 3×3 grid (zero cells included) so a
    reader never has to distinguish "absent" from "zero". ``scope`` records
    whether this is the deterministic-only, LLM-only, or pooled matrix.
    """

    model_config = ConfigDict(frozen=True)

    scope: MatrixScope
    cells: tuple[ConfusionCell, ...]

    def count(self, truth: Bucket, predicted: Bucket) -> int:
        """Count of cases whose human bucket is ``truth`` and pipeline
        bucket is ``predicted``."""
        for cell in self.cells:
            if cell.truth == truth and cell.predicted == predicted:
                return cell.count
        return 0

    @property
    def total(self) -> int:
        return sum(cell.count for cell in self.cells)

    @property
    def correct(self) -> int:
        """Cases on the diagonal (pipeline bucket == human bucket)."""
        return sum(cell.count for cell in self.cells if cell.truth == cell.predicted)

    @property
    def accuracy(self) -> float:
        """Diagonal fraction, or 0.0 for an empty matrix."""
        total = self.total
        return self.correct / total if total else 0.0


class RunComparison(BaseModel):
    """Before/after comparison of two pipeline runs against the same labels.

    ``regressions`` are the REQNOs whose human bucket is ``appropriate`` and
    whose pipeline verdict was correct (``appropriate``) *before* but is no
    longer after — the honest cost of tightening. ``llm_volume_*`` count how
    many cases each run routed to the LLM; the delta is the throughput cost
    of the pre-op deferral.
    """

    model_config = ConfigDict(frozen=True)

    before: tuple[ConfusionMatrix, ...]
    after: tuple[ConfusionMatrix, ...]
    regressions: tuple[str, ...]
    llm_volume_before: int = Field(ge=0)
    llm_volume_after: int = Field(ge=0)

    @property
    def llm_volume_delta(self) -> int:
        return self.llm_volume_after - self.llm_volume_before

    def matrix(
        self, run: Literal["before", "after"], scope: MatrixScope
    ) -> ConfusionMatrix:
        """Fetch one scoped matrix from either run."""
        matrices = self.before if run == "before" else self.after
        for m in matrices:
            if m.scope == scope:
                return m
        raise KeyError(f"no {scope!r} matrix in the {run!r} run")


__all__: Sequence[str] = (
    "BUCKETS",
    "Bucket",
    "CaseVerdict",
    "ConfusionCell",
    "ConfusionMatrix",
    "Mechanism",
    "MatrixScope",
    "RunComparison",
    "bucket_of",
)
