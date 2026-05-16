"""Weekly intra-model κ on a 200-case sentinel set (PRD §18).

The sentinel set is constructed ONCE with a deterministic seed (default 42)
and re-run weekly through :mod:`bba.audit_pipeline` to verify the model's
output is stable week-over-week. Cohen's κ + Gwet's AC1 are computed
between consecutive runs over the paired ``Classification`` labels; an
alarm fires when ``κ < SENTINEL_KAPPA_ALARM_THRESHOLD`` (0.90).

Metric implementations are imported from :mod:`bba.eval_harness.agreement`
— this module does NOT re-implement κ or AC1.

Operational, not clinical: the sentinel comparison only cares about the
classification label, never about the underlying clinical reasoning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from bba.audit_store import AuditRow, Classification
from bba.monitoring.models import (
    SENTINEL_KAPPA_ALARM_THRESHOLD,
    SENTINEL_SET_SEED,
    SENTINEL_SET_SIZE,
    SentinelComparison,
    SentinelManifest,
)


def build_sentinel_manifest(
    audit_rows: Sequence[AuditRow],
    *,
    size: int = SENTINEL_SET_SIZE,
    seed: int = SENTINEL_SET_SEED,
) -> SentinelManifest:
    """Build the 200-case sentinel manifest.

    Deterministic: the same ``audit_rows`` + ``seed`` + ``size`` always
    produces the same ``audit_ids``. Designed to be called ONCE per
    deployment; the manifest is then persisted (e.g., via
    :class:`bba.monitoring.MonitoringStore`) and reused.

    Raises :class:`ValueError` if ``size`` exceeds the row population.
    """
    raise NotImplementedError


def evaluate_sentinel_run(
    *,
    manifest: SentinelManifest,
    previous: Mapping[str, Classification],
    current: Mapping[str, Classification],
    kappa_threshold: float = SENTINEL_KAPPA_ALARM_THRESHOLD,
) -> SentinelComparison:
    """Compare last week's run to this week's run on the sentinel set.

    ``previous`` and ``current`` map ``audit_id`` to the pipeline's
    ``final_classification``. Only audit_ids present in BOTH mappings
    AND in ``manifest.audit_ids`` are paired; any missing pairs are
    silently dropped (an audit row that no longer exists in this week's
    re-run is itself a signal — but the κ test reports on the paired
    subset, not on missingness, and the operator gets the count via
    ``SentinelComparison.n_paired``).

    κ and AC1 are computed via
    :func:`bba.eval_harness.agreement.cohen_kappa` and
    :func:`bba.eval_harness.agreement.gwet_ac1` — NOT re-implemented.

    Raises :class:`bba.monitoring.InsufficientHistoryError` when
    ``previous`` is empty (no prior week to compare against).
    """
    raise NotImplementedError


__all__ = (
    "build_sentinel_manifest",
    "evaluate_sentinel_run",
)
