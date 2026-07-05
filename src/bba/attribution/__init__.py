"""Ordering-doctor / department attribution and appropriateness ranking.

Feature 2: rank the top-10 ordering doctors and departments by
blood-order appropriateness in three buckets (Appropriate /
Inappropriate / Unresolved). Attribution is ``BDVST.DCTREQ`` →
``DCT.csv``; the verdict source is swappable (:data:`VerdictSource`) —
the current build ranks on the 300-case human review, the next build
swaps in full-cohort pipeline verdicts without touching anything else.
"""

from __future__ import annotations

from bba.attribution.dct_registry import load_dct_registry
from bba.attribution.models import (
    UNATTRIBUTED_DEPARTMENT_ID,
    UNATTRIBUTED_DOCTOR_ID,
    Bucket,
    BucketTotals,
    Dimension,
    DoctorRecord,
    RankedRow,
    RankingResult,
    RankingTable,
)
from bba.attribution.order_doctor_map import load_reqno_to_doctor
from bba.attribution.outputs import (
    RANKING_CSV_COLUMNS,
    write_ranking_csv,
    write_rankings_html,
)
from bba.attribution.pipeline import build_rankings
from bba.attribution.ranking import (
    DEFAULT_MIN_ORDERS,
    DEFAULT_TOP_N,
    rank_department_scorecards,
    rank_doctor_scorecards,
    rank_top_n,
)
from bba.attribution.resolvers import (
    SupportsReqno,
    make_physician_resolver,
    make_ward_resolver,
)
from bba.attribution.scorecards import (
    build_department_scorecards,
    build_doctor_scorecards,
)
from bba.attribution.verdicts import (
    HUMAN_LABEL_TO_CLASSIFICATION,
    VerdictSource,
    human_label_verdict_source,
)


__all__ = [
    "Bucket",
    "BucketTotals",
    "DEFAULT_MIN_ORDERS",
    "DEFAULT_TOP_N",
    "Dimension",
    "DoctorRecord",
    "HUMAN_LABEL_TO_CLASSIFICATION",
    "RANKING_CSV_COLUMNS",
    "RankedRow",
    "RankingResult",
    "RankingTable",
    "SupportsReqno",
    "UNATTRIBUTED_DEPARTMENT_ID",
    "UNATTRIBUTED_DOCTOR_ID",
    "VerdictSource",
    "build_department_scorecards",
    "build_doctor_scorecards",
    "build_rankings",
    "human_label_verdict_source",
    "load_dct_registry",
    "load_reqno_to_doctor",
    "make_physician_resolver",
    "make_ward_resolver",
    "rank_department_scorecards",
    "rank_doctor_scorecards",
    "rank_top_n",
    "write_ranking_csv",
    "write_rankings_html",
]
