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

from bba.monitoring.models import (
    GOLDEN_SET_CLASSIFICATION_DRIFT_THRESHOLD,
    GOLDEN_SET_INDICATION_DRIFT_THRESHOLD,
    GoldenSetDriftReport,
    GoldenSetEntry,
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
    ``indications_changed`` is True when the indication sets differ
    (order-independent comparison — indications are a set, not a list).

    Raises :class:`bba.monitoring.GoldenSetMismatchError` when ``baseline``
    and ``current`` do not cover the same ``audit_id`` set (a row missing
    from either side means the golden set itself was edited, which
    invalidates the comparison contract).
    """
    raise NotImplementedError


__all__ = ("evaluate_golden_set_drift",)
