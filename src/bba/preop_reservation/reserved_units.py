"""Reserved-unit quantity keying for the MSBOS evaluator (#162).

Keying BDVSTDT lines on ``(HN, REQNO)`` prevents a REQNO reused across
admissions or patients from importing foreign reservation lines. This mirrors
the ``usetype_values_by_hn_reqno`` rationale in ``scripts/pilot/run_pipeline.py``.
Restricting each total to its evaluated :class:`ComponentFamily` also keeps an
RBC reservation from counting platelet units. Introduced inert in T0 (#162);
the pilot leg now consumes it for both the RBC (RED_CELL, T1-T3) and platelet
(PLATELET, T4) reservation verdicts, behind the default-OFF
``MSBOS_RESERVATION_ENABLED`` flag.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from bba.component_map import ComponentFamily, component_of_code


def reserved_units_by_component(
    bdvstdt_rows: Sequence[Mapping[str, str]],
) -> Mapping[tuple[str, str, ComponentFamily], int]:
    """Sum valid positive ``UNITAMT`` lines by HN, REQNO, and component.

    Invalid, blank, zero, and negative quantities are skipped per line so one
    malformed detail does not discard valid reservations on the same key.
    Keys are returned in sorted order for deterministic iteration (#162).
    """
    totals: dict[tuple[str, str, ComponentFamily], int] = {}
    for row in bdvstdt_rows:
        key = (
            (row.get("HN") or "").strip(),
            (row.get("REQNO") or "").strip(),
            component_of_code((row.get("BDTYPE") or "").strip()),
        )
        totals.setdefault(key, 0)
        try:
            units = int((row.get("UNITAMT") or "").strip(), 10)
        except ValueError:
            continue
        if units > 0:
            totals[key] += units
    return dict(sorted(totals.items()))


__all__: Sequence[str] = ("reserved_units_by_component",)
