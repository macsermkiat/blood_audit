"""Numeric validation for HOSxP Lab ``RESULT`` strings.

Per PRD §3: a Hb value is a numeric string in [2, 25] g/dL. Anything else —
empty, non-numeric, out-of-range — is rejected with ``None``. The parser
never silently coerces; the strict-loud philosophy from the time parser
applies here too.
"""

from __future__ import annotations

import math

_MIN_G_DL = 2.0
_MAX_G_DL = 25.0


def parse_hb_value(raw: str | None) -> float | None:
    """Parse a Lab ``RESULT`` string to a Hb value in g/dL.

    Returns the parsed ``float`` when ``raw`` is a numeric string that lies
    in the closed interval [2.0, 25.0]. Returns ``None`` for non-numeric
    strings, out-of-range values, ``None`` input, empty input, or
    ``nan`` / ``inf``.

    Whitespace around the number is tolerated (HOSxP exports occasionally
    pad numeric columns). Unit suffixes like ``13.5 g/dL`` are NOT — the
    column is numeric-only by schema, and a unit suffix is a sign of
    upstream contamination that should fail loud.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        value = float(stripped)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    if not (_MIN_G_DL <= value <= _MAX_G_DL):
        return None
    return value
