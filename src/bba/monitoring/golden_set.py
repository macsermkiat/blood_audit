"""Quarterly 100-row golden-set drift probe (PRD §18).

A fixed 100-row golden set is re-run quarterly through
:mod:`bba.audit_pipeline` against the same Anthropic snapshot ID. The
results are compared to last quarter's run; alarms fire if:

* >5% of rows changed classification (``classification_alarm_fired``), OR
* >10% of rows changed cited indications (``indications_alarm_fired``).

Both alarms can fire independently. The per-row delta list is preserved so
the operator can drill into specific changes via the dashboard.

Test contract: NO live Anthropic calls. The quarterly probe in tests
uses recorded VCR cassettes from issue #22 to simulate "same model, same
prompt, same answer" — the golden-set drift probe is therefore validated
by injecting synthetic deltas into the cassette-replay output, not by
hitting the real API.

Operational, not clinical: this module reads the pipeline's
``final_classification`` and ``indications`` fields off audit rows and
never re-derives the underlying clinical interpretation.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.monitoring.exceptions import GoldenSetMismatchError
from bba.monitoring.models import (
    GOLDEN_SET_CLASSIFICATION_DRIFT_THRESHOLD,
    GOLDEN_SET_INDICATION_DRIFT_THRESHOLD,
    GoldenSetDriftReport,
    GoldenSetEntry,
    GoldenSetRowDelta,
)


def evaluate_golden_set_drift(
    *,
    baseline: Sequence[GoldenSetEntry],
    current: Sequence[GoldenSetEntry],
    classification_change_threshold: float = GOLDEN_SET_CLASSIFICATION_DRIFT_THRESHOLD,
    indication_change_threshold: float = GOLDEN_SET_INDICATION_DRIFT_THRESHOLD,
) -> GoldenSetDriftReport:
    """Compute the quarterly golden-set drift report.

    Rows are paired by ``audit_id``. ``classification_changed`` is True
    when ``baseline.classification != current.classification``;
    ``indications_changed`` is True when the indication SETS differ
    (order-independent — indications are a set, not a sequence).

    Raises :class:`GoldenSetMismatchError` when ``baseline`` and
    ``current`` do not cover the same ``audit_id`` set. A row missing
    from either side means the golden set itself was edited, which
    invalidates the comparison contract — the operator must rebuild
    both quarters from a common manifest before re-running the probe.
    """
    baseline_map = {entry.audit_id: entry for entry in baseline}
    current_map = {entry.audit_id: entry for entry in current}
    if baseline_map.keys() != current_map.keys():
        missing_in_current = sorted(baseline_map.keys() - current_map.keys())
        missing_in_baseline = sorted(current_map.keys() - baseline_map.keys())
        raise GoldenSetMismatchError(
            f"golden-set audit_id sets differ: "
            f"missing_in_current={missing_in_current!r}, "
            f"missing_in_baseline={missing_in_baseline!r}"
        )

    deltas: list[GoldenSetRowDelta] = []
    n_classification_changed = 0
    n_indications_changed = 0
    for audit_id in sorted(baseline_map.keys()):
        baseline_entry = baseline_map[audit_id]
        current_entry = current_map[audit_id]
        classification_changed = (
            baseline_entry.classification != current_entry.classification
        )
        indications_changed = set(baseline_entry.indications) != set(
            current_entry.indications
        )
        if classification_changed:
            n_classification_changed += 1
        if indications_changed:
            n_indications_changed += 1
        deltas.append(
            GoldenSetRowDelta(
                audit_id=audit_id,
                classification_changed=classification_changed,
                indications_changed=indications_changed,
                baseline_classification=baseline_entry.classification,
                current_classification=current_entry.classification,
                baseline_indications=baseline_entry.indications,
                current_indications=current_entry.indications,
            )
        )

    n_rows = len(deltas)
    classification_pct = n_classification_changed / n_rows if n_rows else 0.0
    indications_pct = n_indications_changed / n_rows if n_rows else 0.0

    return GoldenSetDriftReport(
        n_rows=n_rows,
        classification_changed_pct=classification_pct,
        indications_changed_pct=indications_pct,
        classification_alarm_fired=(
            classification_pct > classification_change_threshold
        ),
        indications_alarm_fired=(indications_pct > indication_change_threshold),
        deltas=tuple(deltas),
    )


__all__ = ("evaluate_golden_set_drift",)
