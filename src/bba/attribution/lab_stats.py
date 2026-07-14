"""Per-order lab values joined onto the ranking cohort.

Feature 2 addendum (spec #131): the committee ranking report shows, per
ordering doctor and per department, the mean *pre-transfusion trigger* —
the mean Hb the group's red-cell orders were transfused at. The values
are not recomputed here: the pipeline already emits a per-order Hb
(``hb_value_g_dl``), its freshness (``hb_freshness``), and the blood
``component`` in ``report.csv``. This module joins those onto the same
scorable REQNO cohort the scorecards already count, so a returned /
never-transfused or out-of-cohort order can never be presented as a
trigger, and a group's sample size ``n`` can never exceed its Orders (N).

Strict reuse of the lookup layer's analytic range (``[2, 25]`` g/dL) means
a corrupt reading cannot distort a mean, and the loader fails loud on
schema drift or a conflicting duplicate REQNO — a silently dropped column
or a concatenated export must never quietly zero out the join.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bba.attribution.csv_support import require_columns
from bba.attribution.models import (
    UNATTRIBUTED_DEPARTMENT_ID,
    UNATTRIBUTED_DOCTOR_ID,
    DoctorRecord,
)

# The lookup layer owns the plausibility bounds; reuse them rather than
# re-inventing so this join and the ingest parsers reject the same values.
from bba.hb_lookup.parse import _MAX_G_DL as _HB_MAX_G_DL
from bba.hb_lookup.parse import _MIN_G_DL as _HB_MIN_G_DL

# Terminals excluded from the scorable denominator (spec #119). Aggregating
# only over the complement guarantees n <= Orders (N) for every group,
# mirroring the total_orders arithmetic in
# :func:`bba.attribution.scorecards.build_doctor_scorecards`.
_EXCLUDED_FROM_SCORING = frozenset(
    {"RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"}
)

_RED_CELL = "red_cell"
_MISSING_FRESHNESS = "missing"

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "reqno",
    "component",
    "hb_value_g_dl",
    "hb_freshness",
)


class OrderLabValue(BaseModel):
    """One order's usable pre-transfusion lab value(s), keyed by REQNO.

    ``hb_value_g_dl`` is the red-cell Hb only when it is genuinely usable
    (red-cell component, non-missing freshness, finite and in range);
    otherwise it is ``None``. Usability is decided once, in the loader, so
    the aggregators are a pure sum-and-count.
    """

    model_config = ConfigDict(frozen=True)

    reqno: str = Field(min_length=1)
    component: str
    hb_value_g_dl: float | None = None


class GroupLabStats(BaseModel):
    """Mean pre-transfusion trigger for one group (doctor or department).

    Enforces the reporting invariant ``hb_order_n == 0`` iff ``mean_hb is
    None``: a group with no usable value must render as absent (``—``),
    never as a misleading ``0.0`` trigger.
    """

    model_config = ConfigDict(frozen=True)

    mean_hb: float | None = None
    hb_order_n: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _mean_iff_sample(self) -> GroupLabStats:
        if (self.hb_order_n == 0) != (self.mean_hb is None):
            raise ValueError(
                "GroupLabStats invariant violated: hb_order_n == 0 must hold "
                "if and only if mean_hb is None (got "
                f"hb_order_n={self.hb_order_n}, mean_hb={self.mean_hb!r})"
            )
        return self


def _usable_hb(value_raw: str, freshness: str, component: str) -> float | None:
    """The pre-transfusion Hb a report row contributes, or ``None`` if it
    must not: a non-red-cell (or blank/unknown) component, a missing
    freshness sentinel, an unparseable cell, a non-finite number, or a
    value outside the lookup layer's ``[2, 25]`` g/dL range."""
    if component != _RED_CELL:
        return None
    if freshness == _MISSING_FRESHNESS:
        return None
    raw = value_raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    if not (_HB_MIN_G_DL <= value <= _HB_MAX_G_DL):
        return None
    return value


def load_order_labs(path: Path) -> Mapping[str, OrderLabValue]:
    """Read the pipeline ``report.csv`` into ``REQNO`` → :class:`OrderLabValue`.

    Fails loud — naming the file and column — on schema drift (a required
    column absent), mirroring the other attribution loaders: a renamed
    column would silently zero out every trigger. A REQNO that re-appears
    with a *conflicting* lab record fails loud (a corrupted or concatenated
    export); an identical duplicate is tolerated.
    """
    labs: dict[str, OrderLabValue] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        require_columns(reader.fieldnames, _REQUIRED_COLUMNS, path)
        for row in reader:
            reqno = (row["reqno"] or "").strip()
            if not reqno:
                continue
            component = (row["component"] or "").strip()
            record = OrderLabValue(
                reqno=reqno,
                component=component,
                hb_value_g_dl=_usable_hb(
                    row["hb_value_g_dl"] or "",
                    (row["hb_freshness"] or "").strip(),
                    component,
                ),
            )
            existing = labs.get(reqno)
            if existing is not None and existing != record:
                raise ValueError(
                    f"{path} maps REQNO {reqno!r} to two different lab records "
                    f"({existing!r} and {record!r}); REQNO must be unique — "
                    "refusing to aggregate on a corrupted or concatenated export"
                )
            labs[reqno] = record
    return labs


def _stats_from_hb(hb_values: list[float]) -> GroupLabStats:
    """Collapse a group's usable Hb values into its :class:`GroupLabStats`,
    preserving the ``n == 0`` iff ``mean is None`` invariant."""
    n = len(hb_values)
    if n == 0:
        return GroupLabStats()
    return GroupLabStats(mean_hb=sum(hb_values) / n, hb_order_n=n)


def aggregate_doctor_lab_stats(
    verdicts: Mapping[str, str],
    reqno_to_doctor: Mapping[str, str],
    order_labs: Mapping[str, OrderLabValue],
) -> Mapping[str, GroupLabStats]:
    """One :class:`GroupLabStats` per ordering doctor, over the scorable
    REQNO cohort only.

    Doctor identity resolves exactly as
    :func:`bba.attribution.scorecards.build_doctor_scorecards` (a miss
    lands on :data:`UNATTRIBUTED_DOCTOR_ID`), so the group ids line up with
    the ranking rows. Returns terminals are skipped and lab rows outside
    the verdict cohort are never consulted, so ``n <= Orders (N)``.
    """
    hb_by_doctor: dict[str, list[float]] = {}
    for reqno, classification in verdicts.items():
        if classification in _EXCLUDED_FROM_SCORING:
            continue
        doctor = reqno_to_doctor.get(reqno, UNATTRIBUTED_DOCTOR_ID)
        hb_by_doctor.setdefault(doctor, [])
        lab = order_labs.get(reqno)
        if lab is not None and lab.hb_value_g_dl is not None:
            hb_by_doctor[doctor].append(lab.hb_value_g_dl)
    return {doctor: _stats_from_hb(values) for doctor, values in hb_by_doctor.items()}


def aggregate_department_lab_stats(
    verdicts: Mapping[str, str],
    reqno_to_doctor: Mapping[str, str],
    dct_registry: Mapping[str, DoctorRecord],
    order_labs: Mapping[str, OrderLabValue],
) -> Mapping[str, GroupLabStats]:
    """One :class:`GroupLabStats` per department (``Deptlct``), over the
    scorable REQNO cohort only.

    Department identity resolves via ``reqno`` → doctor → registry
    ``Deptlct`` exactly as
    :func:`bba.attribution.scorecards.build_department_scorecards` (any
    broken link lands on :data:`UNATTRIBUTED_DEPARTMENT_ID`), so the group
    ids line up with the ranking rows.
    """
    hb_by_dept: dict[str, list[float]] = {}
    for reqno, classification in verdicts.items():
        if classification in _EXCLUDED_FROM_SCORING:
            continue
        doctor = reqno_to_doctor.get(reqno)
        record = dct_registry.get(doctor) if doctor is not None else None
        dept = (
            record.deptlct
            if record is not None and record.deptlct
            else (UNATTRIBUTED_DEPARTMENT_ID)
        )
        hb_by_dept.setdefault(dept, [])
        lab = order_labs.get(reqno)
        if lab is not None and lab.hb_value_g_dl is not None:
            hb_by_dept[dept].append(lab.hb_value_g_dl)
    return {dept: _stats_from_hb(values) for dept, values in hb_by_dept.items()}
