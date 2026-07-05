"""bba.verification — honest before/after scoring for the peri-op fix.

Scores a pipeline run against the 300-case human review
(:func:`bba.attribution.human_label_verdict_source`) in the 3-bucket space
the human labels live in, split by mechanism (deterministic vs LLM), and
compares a before run to an after run so the fix is judged on its real
costs — regressions and the LLM-volume delta — not a single headline.

Pure and data-source-agnostic: callers assemble ``reqno -> CaseVerdict``
maps from whatever run artifacts they have (report.csv / llm_report.json)
and pass them in. The end-to-end scoring run over the pilot bundle lives in
the pilot scripts, gated on the bundle data being present.
"""

from bba.verification.confusion import (
    build_matrix,
    compare_runs,
    confusion_by_mechanism,
    find_regressions,
)
from bba.verification.models import (
    BUCKETS,
    Bucket,
    CaseVerdict,
    ConfusionCell,
    ConfusionMatrix,
    MatrixScope,
    Mechanism,
    RunComparison,
    bucket_of,
)

__all__ = [
    "BUCKETS",
    "Bucket",
    "CaseVerdict",
    "ConfusionCell",
    "ConfusionMatrix",
    "MatrixScope",
    "Mechanism",
    "RunComparison",
    "bucket_of",
    "build_matrix",
    "compare_runs",
    "confusion_by_mechanism",
    "find_regressions",
]
