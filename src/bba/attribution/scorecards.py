"""Scorecard aggregation over a verdict source, per doctor / department.

Reuses the dashboard's frozen scorecard DTOs
(:class:`~bba.dashboard.models.PhysicianScorecard` /
:class:`~bba.dashboard.models.WardScorecard`) — they already carry the
four classification counts the 3-bucket ranking collapses from. The
department dimension flows through the *ward* model on purpose: the
established grouping key in the report / dashboard layers is "ward",
and ``Deptlct`` is supplied through that same shape rather than a
forked aggregator (feature plan, "Department vs ward").

``average_confidence`` is a required field on both reused models; a
human-label verdict source carries no model confidence, so it is set
to ``0.0`` and this feature's output writers never render it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from bba.attribution.models import (
    UNATTRIBUTED_DEPARTMENT_ID,
    DoctorRecord,
    EventVerdict,
)
from bba.attribution.order_doctor_map import (
    attributed_department,
    attributed_doctor,
)
from bba.dashboard.models import PhysicianScorecard, WardScorecard


def _count_classifications(
    classifications: list[str],
) -> dict[str, int]:
    """Zero-initialised label counts, mirroring
    :func:`bba.report_generator.aggregate._count_classifications` so the
    two aggregation surfaces cannot drift on label handling."""
    counts = {
        "APPROPRIATE": 0,
        "INAPPROPRIATE": 0,
        "NEEDS_REVIEW": 0,
        "INSUFFICIENT_EVIDENCE": 0,
        "PREOP_OVER_RESERVATION": 0,
        "RETURNED_NOT_TRANSFUSED": 0,
        "PERIOP_TRANSFUSION_EXEMPT": 0,
    }
    for classification in classifications:
        if classification not in counts:
            raise ValueError(
                f"unsupported attribution classification {classification!r}"
            )
        counts[classification] += 1
    return counts


def build_doctor_scorecards(
    verdicts: Sequence[EventVerdict],
    reqno_to_doctor: Mapping[str, str],
    dct_registry: Mapping[str, DoctorRecord],
) -> tuple[PhysicianScorecard, ...]:
    """One :class:`PhysicianScorecard` per distinct ordering doctor.

    Each event counts once, for the doctor of its linked order; events
    without attribution land on :data:`UNATTRIBUTED_DOCTOR_ID`.
    ``total_orders`` is the scorable event denominator; returned/not-transfused
    and peri-op-exempt events remain visible in their own counters but are
    excluded from it. Output is sorted by ``physician_id`` for byte-stable
    artifacts.
    """
    groups: dict[str, list[str]] = {}
    for event in verdicts:
        doctor = attributed_doctor(event, reqno_to_doctor)
        groups.setdefault(doctor, []).append(event.verdict)

    cards: list[PhysicianScorecard] = []
    for doctor in sorted(groups):
        counts = _count_classifications(groups[doctor])
        record = dct_registry.get(doctor)
        name = record.display_name if record and record.display_name else doctor
        ward_id = (
            record.deptlct if record and record.deptlct else UNATTRIBUTED_DEPARTMENT_ID
        )
        cards.append(
            PhysicianScorecard(
                physician_id=doctor,
                physician_name=name,
                ward_id=ward_id,
                total_orders=len(groups[doctor])
                - counts["RETURNED_NOT_TRANSFUSED"]
                - counts["PERIOP_TRANSFUSION_EXEMPT"],
                appropriate_count=counts["APPROPRIATE"],
                inappropriate_count=(
                    counts["INAPPROPRIATE"] + counts["PREOP_OVER_RESERVATION"]
                ),
                needs_review_count=counts["NEEDS_REVIEW"],
                insufficient_evidence_count=counts["INSUFFICIENT_EVIDENCE"],
                returned_not_transfused_count=counts["RETURNED_NOT_TRANSFUSED"],
                periop_transfusion_exempt_count=counts["PERIOP_TRANSFUSION_EXEMPT"],
                over_reservation_count=counts["PREOP_OVER_RESERVATION"],
                average_confidence=0.0,
            )
        )
    return tuple(cards)


def build_department_scorecards(
    verdicts: Sequence[EventVerdict],
    reqno_to_doctor: Mapping[str, str],
    dct_registry: Mapping[str, DoctorRecord],
) -> tuple[WardScorecard, ...]:
    """One :class:`WardScorecard` per distinct department (``Deptlct``).

    Resolution chain per event: ``linked_reqno`` → doctor → registry
    ``Deptlct``; any broken link lands on
    :data:`UNATTRIBUTED_DEPARTMENT_ID`. Same conservation and sorting
    guarantees as :func:`build_doctor_scorecards`.
    """
    groups: dict[str, list[str]] = {}
    names: dict[str, str] = {}
    for event in verdicts:
        record = attributed_department(event, reqno_to_doctor, dct_registry)
        if record is not None:
            dept = record.deptlct
            if record.deptname and dept not in names:
                names[dept] = record.deptname
        else:
            dept = UNATTRIBUTED_DEPARTMENT_ID
        groups.setdefault(dept, []).append(event.verdict)

    cards: list[WardScorecard] = []
    for dept in sorted(groups):
        counts = _count_classifications(groups[dept])
        cards.append(
            WardScorecard(
                ward_id=dept,
                ward_name=names.get(dept, dept),
                total_orders=len(groups[dept])
                - counts["RETURNED_NOT_TRANSFUSED"]
                - counts["PERIOP_TRANSFUSION_EXEMPT"],
                appropriate_count=counts["APPROPRIATE"],
                inappropriate_count=(
                    counts["INAPPROPRIATE"] + counts["PREOP_OVER_RESERVATION"]
                ),
                needs_review_count=counts["NEEDS_REVIEW"],
                insufficient_evidence_count=counts["INSUFFICIENT_EVIDENCE"],
                returned_not_transfused_count=counts["RETURNED_NOT_TRANSFUSED"],
                periop_transfusion_exempt_count=counts["PERIOP_TRANSFUSION_EXEMPT"],
                over_reservation_count=counts["PREOP_OVER_RESERVATION"],
                average_confidence=0.0,
            )
        )
    return tuple(cards)
