"""Numeric validation for HOSxP platelet-count ``RESULT`` strings (LABEXM 290078).

The platelet-count column is messier than the Hb column, so this parser is
NOT a copy of :func:`bba.hb_lookup.parse.parse_hb_value`. Verified against the
raw Lab feed (2026-07-08), platelet ``RESULT`` cells include:

* plain integers/decimals: ``362``, ``10``
* comma-grouped thousands: ``1,117``, ``1,030`` (values above ~1000)
* left-censored values: ``<2`` (below the analyser's detection floor — a
  REAL, critically-low measurement, not missing data)
* a missing sentinel: ``--``

Design decisions (docs plan §5.4; project memory project_platelet_component_map):

* ``<N`` / ``>N`` are censored, not missing. They carry real clinical
  information (``<2`` is severe thrombocytopenia) and must not collapse to the
  missing-count contract, which would spuriously defer/insufficient a
  documented critical count. We return the censoring BOUND (``<2`` → ``2.0``,
  ``>1000`` → ``1000.0``). For the §5.1 gate this preserves the correct
  routing: a ``<N`` below 100 still routes to NEEDS_REVIEW, a ``>N`` at/above
  100 still routes to POTENTIALLY_INAPPROPRIATE.
* ``--`` / empty / non-numeric → ``None`` (missing), handled by the classifier's
  missing-count contract.
* out-of-range → ``None``: a transcription/unit error, failed loud, not coerced.

Unit is ×10³/µL. The analytic-validity window is
[:data:`MIN_PLATELET`, :data:`MAX_PLATELET`]; note this rejects nothing the Hb
window ([2, 25] g/dL) would accept and vice-versa — the two components cannot
share a validity range, which is why hb_lookup's parser cannot be reused here.
"""

from __future__ import annotations

import math

# Analytic-validity window (×10³/µL). Normal range is 150-450; clinically
# observed values span roughly 2-1100 in the raw feed. The window is
# deliberately generous at the top (post-reactive thrombocytosis can exceed
# 1000) and floored at 1.0 so a censored ``<2`` (→ 2.0) and a genuine
# critical ``1`` both pass, while 0 / negatives / absurd values are rejected
# as errors.
MIN_PLATELET: float = 1.0
MAX_PLATELET: float = 3000.0


def parse_platelet_count(raw: str | None) -> float | None:
    """Parse a Lab ``RESULT`` string to a platelet count in ×10³/µL.

    Returns a ``float`` for a value in the closed interval
    [:data:`MIN_PLATELET`, :data:`MAX_PLATELET`], handling comma-grouped
    thousands and ``<N`` / ``>N`` censoring (mapped to the bound ``N``).
    Returns ``None`` for the ``--`` sentinel, empty/``None`` input,
    non-numeric text, out-of-range values, or ``nan`` / ``inf``.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped or stripped == "--":
        return None

    # Left/right-censored: "<2", ">1000". Keep the bound as a real value.
    if stripped[0] in {"<", ">"}:
        stripped = stripped[1:].strip()

    # HOSxP writes thousands as "1,117"; the comma is a grouping separator,
    # never a decimal (Thai locale uses "." for decimals).
    stripped = stripped.replace(",", "")

    try:
        value = float(stripped)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    if not (MIN_PLATELET <= value <= MAX_PLATELET):
        return None
    return value


__all__ = ("MAX_PLATELET", "MIN_PLATELET", "parse_platelet_count")
