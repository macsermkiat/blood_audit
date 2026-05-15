"""bba.eval_harness — stratified gold-standard + statistical metrics.

See issue #20 for acceptance criteria. PRD §11 (Implementation Decisions)
defines the harness contract: stratified-with-enrichment sampling, dual-
review of the calibration set, Horvitz-Thompson reweighting back to the
population, Wilson CIs and cluster-robust SEs on the headline rates,
κ + Gwet's AC1 + PABAK per stratum, LOMO-CV or blocked-temporal CV
auto-selected by dataset span, hierarchical Bonferroni / BH-FDR correction
on pre-registered primary / exploratory subgroup tests, the verifier-as-
classifier confusion-matrix consumer from #11/#18, and an outcome-anchored
falsification check on the hand-labeled subset.

The harness is the *graded layer* of the audit pipeline: it consumes audit
rows produced upstream (#3-#19) and produces population-level metrics with
publication-grade uncertainty. It is the producer for #28 (report_generator).
"""

from bba.eval_harness.agreement import (
    agreement_with_metrics,
    cohen_kappa,
    gwet_ac1,
    pabak,
)
from bba.eval_harness.classifier import evaluate_confusion_matrix
from bba.eval_harness.cluster import cluster_robust_proportion_ci
from bba.eval_harness.correction import (
    benjamini_hochberg_correction,
    bonferroni_correction,
    hierarchical_correction,
)
from bba.eval_harness.exceptions import (
    EmptyInputError,
    EvalHarnessError,
    InsufficientStratumError,
    ShapeMismatchError,
)
from bba.eval_harness.falsification import outcome_anchored_falsification
from bba.eval_harness.intervals import normal_quantile, wilson_ci
from bba.eval_harness.models import (
    DEFAULT_INAPPROPRIATE_ENRICHMENT_PER_STRATUM,
    DEFAULT_STRATUM_TARGETS,
    LOMO_DATASET_THRESHOLD_MONTHS,
    VERIFIER_LABELED_SET_SIZE,
    AgreementResult,
    AuditCase,
    Classification,
    ClassifierMetrics,
    ClusterRobustEstimate,
    CorrectedTest,
    FalsificationOutcome,
    FalsificationResult,
    HierarchicalCorrectionResult,
    HorvitzThompsonEstimate,
    SamplingTargets,
    SplitStrategy,
    StratifiedSample,
    Stratum,
    StratumDraw,
    TemporalSplit,
    WilsonInterval,
)
from bba.eval_harness.reweight import horvitz_thompson_prevalence
from bba.eval_harness.sampling import stratified_with_enrichment
from bba.eval_harness.splits import (
    blocked_temporal_split,
    dataset_month_span,
    lomo_cv_splits,
    select_split_strategy,
    temporal_cv_splits,
)

__all__ = [
    "DEFAULT_INAPPROPRIATE_ENRICHMENT_PER_STRATUM",
    "DEFAULT_STRATUM_TARGETS",
    "LOMO_DATASET_THRESHOLD_MONTHS",
    "VERIFIER_LABELED_SET_SIZE",
    "AgreementResult",
    "AuditCase",
    "Classification",
    "ClassifierMetrics",
    "ClusterRobustEstimate",
    "CorrectedTest",
    "EmptyInputError",
    "EvalHarnessError",
    "FalsificationOutcome",
    "FalsificationResult",
    "HierarchicalCorrectionResult",
    "HorvitzThompsonEstimate",
    "InsufficientStratumError",
    "SamplingTargets",
    "ShapeMismatchError",
    "SplitStrategy",
    "StratifiedSample",
    "Stratum",
    "StratumDraw",
    "TemporalSplit",
    "WilsonInterval",
    "agreement_with_metrics",
    "benjamini_hochberg_correction",
    "blocked_temporal_split",
    "bonferroni_correction",
    "cluster_robust_proportion_ci",
    "cohen_kappa",
    "dataset_month_span",
    "evaluate_confusion_matrix",
    "gwet_ac1",
    "hierarchical_correction",
    "horvitz_thompson_prevalence",
    "lomo_cv_splits",
    "normal_quantile",
    "outcome_anchored_falsification",
    "pabak",
    "select_split_strategy",
    "stratified_with_enrichment",
    "temporal_cv_splits",
    "wilson_ci",
]
