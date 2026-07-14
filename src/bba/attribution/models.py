"""Frozen models and sentinels for ordering-doctor attribution (Feature 2).

The attribution package supplies the missing M0/M1 piece: mapping an
order (``REQNO``) to its ordering doctor (``BDVST.DCTREQ``) and the
doctor to a department (``DCT.csv`` ``Dct`` → ``Deptlct`` / ``Deptname``).
Scorecard aggregation reuses :mod:`bba.dashboard.models`
(:class:`~bba.dashboard.models.PhysicianScorecard` /
:class:`~bba.dashboard.models.WardScorecard`); the models here cover only
what those do not: the doctor-registry record and the ranking output.

Doctor names in ``DCT.csv`` are masked at source (``ส*****``); doctor
codes and department names are non-PHI, so a :class:`RankedRow` is safe
to render in committee-facing artifacts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Bucket = Literal["appropriate", "inappropriate", "unresolved"]
"""The committee-facing 3-bucket collapse of the 4-value
:data:`~bba.report_generator.models.Classification`:
``unresolved`` = ``NEEDS_REVIEW`` + ``INSUFFICIENT_EVIDENCE`` (PRD
§"Documentation absence ≠ INAPPROPRIATE" — neither is a confident
terminal verdict, so neither may count for or against a doctor)."""


Dimension = Literal["doctor", "department"]


UNATTRIBUTED_DOCTOR_ID = "unattributed-doctor"
"""Sentinel physician_id for orders whose BDVST row has no ``DCTREQ``
(~0.2% of the cohort). Mirrors the dashboard's ``unattributed-ward``
convention: surface the gap as its own bucket, never drop rows."""


UNATTRIBUTED_DEPARTMENT_ID = "unattributed-department"
"""Sentinel ward_id for orders whose doctor is unknown or whose DCT
record carries no ``Deptlct``."""


class DoctorRecord(BaseModel):
    """One doctor row from the ``DCT.csv`` registry.

    ``display_name`` is composed from the masked Prefix/Fname/Lname
    columns — masking happened upstream, so the value is renderable.
    ``deptlct`` / ``deptname`` may be empty (about a third of the raw
    registry rows carry no department); consumers must fall back to
    :data:`UNATTRIBUTED_DEPARTMENT_ID`, not guess.
    """

    model_config = ConfigDict(frozen=True)

    dct: str = Field(min_length=1)
    display_name: str
    deptlct: str
    deptname: str


class RankedRow(BaseModel):
    """One row of a top-N ranking table (doctor or department).

    Carries all three bucket counts plus the ranked bucket's count and
    rate so the table is self-reconciling: ``appropriate_count +
    inappropriate_count + unresolved_count == total_orders``.

    ``meets_min_orders`` is the thin-sample guard from the feature plan:
    rows below the minimum-order threshold are ranked by count *after*
    every qualified row and must be visually flagged — a 1/1 = 100% rate
    is not presentable as a "top" rate.
    """

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    group_id: str = Field(min_length=1)
    group_name: str
    total_orders: int = Field(ge=0)
    appropriate_count: int = Field(ge=0)
    inappropriate_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    returned_not_transfused_count: int = Field(default=0, ge=0)
    periop_transfusion_exempt_count: int = Field(default=0, ge=0)
    bucket: Bucket
    bucket_count: int = Field(ge=0)
    bucket_rate: float = Field(ge=0.0, le=1.0)
    meets_min_orders: bool
    mean_hb: float | None = None
    """Mean pre-transfusion Hb (g/dL) over the group's scorable red-cell
    orders; ``None`` when ``hb_order_n == 0``. Populated from the per-order
    lab join at assembly; defaulted so existing callers are unaffected."""
    hb_order_n: int = Field(default=0, ge=0)
    """Number of scorable red-cell orders the ``mean_hb`` is based on;
    never exceeds ``total_orders``."""


class RankingTable(BaseModel):
    """A complete ranking for one dimension, with its parameters.

    The parameters (``bucket``, ``n``, ``min_orders``) travel with the
    rows so an output writer can state them in the artifact — the plan
    requires the threshold to be visible in the deliverable, not implied.
    """

    model_config = ConfigDict(frozen=True)

    dimension: Dimension
    bucket: Bucket
    n: int = Field(ge=1)
    min_orders: int = Field(ge=1)
    rows: tuple[RankedRow, ...]


class BucketTotals(BaseModel):
    """Whole-cohort bucket totals, for reconciliation against the raw
    verdict source (the 300 human labels reconcile to 162/32/106)."""

    model_config = ConfigDict(frozen=True)

    appropriate: int = Field(ge=0)
    inappropriate: int = Field(ge=0)
    unresolved: int = Field(ge=0)
    returned_not_transfused: int = Field(default=0, ge=0)
    periop_transfusion_exempt: int = Field(default=0, ge=0)
    total: int = Field(ge=0)


class RankingResult(BaseModel):
    """The full Feature-2 deliverable: both ranking tables + totals."""

    model_config = ConfigDict(frozen=True)

    doctors: RankingTable
    departments: RankingTable
    totals: BucketTotals
