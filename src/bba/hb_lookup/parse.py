"""Numeric validation for HOSxP Lab ``RESULT`` strings.

Per PRD §3: a Hb value is a numeric string in [2, 25] g/dL. Anything else —
empty, non-numeric, out-of-range — is rejected with ``None``. The parser
never silently coerces; the strict-loud philosophy from the time parser
applies here too.
"""

from __future__ import annotations


def parse_hb_value(raw: str | None) -> float | None:
    """Parse a Lab ``RESULT`` string to a Hb value in g/dL.

    Returns the parsed ``float`` when ``raw`` is a numeric string that lies
    in the closed interval [2.0, 25.0] (inclusive). Returns ``None`` for
    non-numeric strings, out-of-range values, ``None`` input, or empty input.

    Whitespace is tolerated (the parser strips before parsing). Unit
    suffixes like "g/dL" are NOT accepted — the column is numeric-only by
    schema, and a unit suffix is a sign of upstream contamination that
    should fail loud.
    """

    raise NotImplementedError("parse_hb_value: implement in GREEN phase")
