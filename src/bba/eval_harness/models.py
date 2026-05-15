"""Pydantic v2 models, enums, and constants for the eval-harness contract.

This module is intentionally typed-only — every function across the eval
harness operates on these frozen records, so the contract a downstream
report cites cannot drift by accident. Field names follow PRD §11 and the
publication-grade naming Anthropic/Thai FDA reviewers will encounter.

Strata + their target sizes come from PRD User Story #25:
``Hb<7: 140; Hb 7-10: 350; Hb>10: 260; INSUFFICIENT: 100; adversarial: 80;
cohort-exception: 140`` — total 1,070 with the per-stratum INAPPROPRIATE
positive-enrichment target of ~138 (PRD Implementation Decisions §11).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Stratum(StrEnum):
    """Sampling stratum (PRD User Story #25).

    Mutually exclusive partition of the audited-orders population. Each
    audit case belongs to exactly one stratum; the stratification key is a
    function of the deterministic-classifier inputs (Hb tier, evidence
    sufficiency, adversarial flag, cohort tag), NOT of the LLM prediction —
    otherwise the per-stratum sensitivity targets would be self-fulfilling.
    """

    HB_LT_7 = "hb_lt_7"
    HB_7_10 = "hb_7_10"
    HB_GT_10 = "hb_gt_10"
    INSUFFICIENT = "insufficient_evidence"
    ADVERSARIAL = "adversarial"
    COHORT_EXCEPTION = "cohort_exception"


# Per-stratum target sample sizes from PRD User Story #25.
DEFAULT_STRATUM_TARGETS: Mapping[Stratum, int] = MappingProxyType(
    {
        Stratum.HB_LT_7: 140,
        Stratum.HB_7_10: 350,
        Stratum.HB_GT_10: 260,
        Stratum.INSUFFICIENT: 100,
        Stratum.ADVERSARIAL: 80,
        Stratum.COHORT_EXCEPTION: 140,
    }
)

# Per-stratum enrichment target for INAPPROPRIATE predictions (PRD §11).
# Enrichment oversamples positives so per-stratum sensitivity is testable at
# ±0.05 Wilson CI even when the natural prevalence is low.
DEFAULT_INAPPROPRIATE_ENRICHMENT_PER_STRATUM = 138

# Split-strategy threshold (PRD §11): below this many months of data, LOMO-CV
# is the auto-pick; at or above, blocked temporal split. The threshold of 12
# matches the natural seasonal-confounder horizon (one full year).
LOMO_DATASET_THRESHOLD_MONTHS = 12

# Verifier-as-classifier hand-labeled set size (PRD §11).
VERIFIER_LABELED_SET_SIZE = 200


Classification = Literal[
    "APPROPRIATE",
    "INAPPROPRIATE",
    "NEEDS_REVIEW",
    "INSUFFICIENT_EVIDENCE",
]
"""The four pipeline output classes (PRD §"Solution")."""


SplitStrategy = Literal["lomo", "blocked"]
"""Cross-validation strategy chosen by :func:`select_split_strategy`."""


class AuditCase(BaseModel):
    """One audited RBC order, in the shape the eval harness operates on.

    A projection of :class:`bba.audit_store.AuditRow` — only the fields the
    harness reads. Keeping a separate, narrower model lets eval consume a
    pre-projected stream without dragging the full ~30-field audit row and
    its persistence-layer validators; it also documents the eval boundary
    independently of upstream schema drift.

    ``physician_id`` and ``ward`` carry the two cluster keys for the
    cluster-robust SE (PRD §11: "cluster-robust SE clustered on physician +
    ward"). ``gold_classification`` is ``None`` until a reviewer has labeled
    the case — most of the population is unlabeled until the eval draw.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: str
    stratum: Stratum
    pred_classification: Classification
    gold_classification: Classification | None
    order_datetime: datetime
    physician_id: str
    ward: str


class SamplingTargets(BaseModel):
    """Per-stratum target sample sizes + per-stratum enrichment target."""

    model_config = ConfigDict(frozen=True)

    per_stratum: Mapping[Stratum, int]
    inappropriate_enrichment_per_stratum: int


class StratumDraw(BaseModel):
    """One stratum's sample after stratified-with-enrichment draw.

    ``inclusion_probability`` is the per-case probability used for
    Horvitz-Thompson reweighting back to the population
    (:mod:`bba.eval_harness.reweight`). The enrichment step makes inclusion
    probability NOT uniform within a stratum — positives are oversampled — so
    the HT weight is the per-case ``1 / P(included | population)``.
    """

    model_config = ConfigDict(frozen=True)

    stratum: Stratum
    cases: tuple[AuditCase, ...]
    population_size: int
    population_positives: int
    drawn_positives: int
    base_inclusion_probability: float
    positive_inclusion_probability: float


class StratifiedSample(BaseModel):
    """Container for the full multi-stratum draw."""

    model_config = ConfigDict(frozen=True)

    draws: tuple[StratumDraw, ...]


class WilsonInterval(BaseModel):
    """Wilson score confidence interval for a binomial proportion.

    ``point`` is the naive proportion (``successes / trials``);
    ``lower``/``upper`` are the Wilson-corrected bounds at ``confidence``.
    For ``successes == 0`` the lower bound stays at 0; for
    ``successes == trials`` the upper bound stays at 1. PRD §11 requires the
    Wilson form (not the normal approximation), which prevents boundary
    estimates falling outside [0, 1] — critical for the per-stratum
    sensitivity targets where the natural prevalence is very high or low.
    """

    model_config = ConfigDict(frozen=True)

    point: float
    lower: float
    upper: float
    confidence: float


class HorvitzThompsonEstimate(BaseModel):
    """Population-prevalence estimate reweighted by inverse inclusion prob.

    ``prevalence`` is the HT estimator: each sampled case contributes its
    inverse inclusion probability, summed across the sample, divided by the
    known population size. ``variance`` is the standard HT variance estimator
    over independent strata (with-replacement approximation); ``se`` is its
    square root.
    """

    model_config = ConfigDict(frozen=True)

    prevalence: float
    variance: float
    se: float


class AgreementResult(BaseModel):
    """Per-stratum inter-rater agreement bundle.

    Cohen's κ, Gwet's AC1, and PABAK are reported *together* on purpose:
    high-prevalence strata (PRD §11 Hb>10) produce a κ deflated by the
    kappa-prevalence paradox while AC1 and PABAK remain stable. Reporting
    only κ would hide a usable inter-rater quality signal.
    """

    model_config = ConfigDict(frozen=True)

    cohen_kappa: float
    gwet_ac1: float
    pabak: float
    observed_agreement: float
    n_items: int


class ClusterRobustEstimate(BaseModel):
    """Cluster-robust point + SE + Wald CI on a proportion.

    Audits are non-i.i.d.: orders cluster by ordering physician and by ward
    (PRD User Story #31). The naive binomial SE understates uncertainty by
    1.5–2x. The cluster-robust SE uses the CR0 sandwich estimator
    aggregated over physician+ward clusters and produces a normal-approximation
    Wald CI at the requested confidence.
    """

    model_config = ConfigDict(frozen=True)

    point: float
    cluster_robust_se: float
    naive_se: float
    n_clusters: int
    n_obs: int
    lower: float
    upper: float
    confidence: float


class TemporalSplit(BaseModel):
    """One train/holdout split for cross-validation.

    ``holdout_label`` names the held-out chunk: for LOMO, the ``YYYY-MM``
    month tag; for blocked temporal, a 1-indexed block id (``"block-1"``).
    Splits are exact, by ``audit_id`` — a downstream caller can re-key into
    its own data structure without worrying about ordering ambiguity.
    """

    model_config = ConfigDict(frozen=True)

    train_audit_ids: tuple[str, ...]
    holdout_audit_ids: tuple[str, ...]
    holdout_label: str


class CorrectedTest(BaseModel):
    """One hypothesis test after multiplicity correction."""

    model_config = ConfigDict(frozen=True)

    name: str
    raw_p: float
    adjusted_p: float
    rejected: bool


class HierarchicalCorrectionResult(BaseModel):
    """Bonferroni on primary hypotheses, BH-FDR on exploratory hypotheses.

    The PRD's pre-registration policy (§11) treats the primary tests as
    confirmatory (strict family-wise error control via Bonferroni) and the
    exploratory tests as hypothesis-generating (false-discovery-rate control
    via Benjamini-Hochberg). The two families are corrected *independently*;
    they do NOT pool into one multiplicity correction.
    """

    model_config = ConfigDict(frozen=True)

    primary: tuple[CorrectedTest, ...]
    exploratory: tuple[CorrectedTest, ...]
    alpha: float


class ClassifierMetrics(BaseModel):
    """Classifier metrics derived from a 2x2 confusion matrix.

    Each rate carries its Wilson CI at the requested confidence. F1 is a
    point estimate only — the CI for F1 has no published closed form and
    bootstrapping it would couple the harness to a heavy dependency without
    materially changing the reviewer-grade conclusion.
    """

    model_config = ConfigDict(frozen=True)

    accuracy: WilsonInterval
    sensitivity: WilsonInterval
    specificity: WilsonInterval
    ppv: WilsonInterval
    npv: WilsonInterval
    f1: float


class FalsificationOutcome(StrEnum):
    """Patient outcome category for the outcome-anchored falsification check.

    Drawn from the audit row's downstream chart-review labels on a small
    hand-labeled subset (PRD §11): an INAPPROPRIATE prediction is *supported*
    when the patient did not require further transfusion and was not
    re-admitted for anemia within the follow-up window; it is *contradicted*
    when the patient did require further transfusion within 24 hours or died
    from anemia within 30 days.

    Scope (v1 — issue #20): three outcome categories per the labeled-subset
    spec. Additional clinical endpoints (24h Hb response, transfusion-
    reaction occurrence, physician documentation amendment) are
    *out-of-scope* for v1 and tracked under follow-up tickets. Extending
    this enum is straightforward — the contradiction set in
    :mod:`bba.eval_harness.falsification` is the single switch a future
    ticket would update.
    """

    NO_FURTHER_TRANSFUSION = "no_further_transfusion"
    FURTHER_TRANSFUSION_24H = "further_transfusion_24h"
    DEATH_FROM_ANEMIA_30D = "death_from_anemia_30d"


class FalsificationResult(BaseModel):
    """Outcome-anchored falsification metrics on the labeled subset."""

    model_config = ConfigDict(frozen=True)

    n_labeled: int
    n_inappropriate_pred: int
    n_supported: int
    n_contradicted: int
    support_rate: WilsonInterval
    contradiction_rate: WilsonInterval
