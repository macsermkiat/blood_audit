"""Age-cap transformation for the deid_redactor.

PRD §8: "age cap at 89" — collapses any age > 89 to exactly 89. The cap
is HIPAA-derived (re-identification risk in the elderly tail) and lives
here so the wrapper can stamp ``redacted_age`` + ``age_capped`` on every
:class:`bba.deid_redactor.models.RedactionResult`.

This is the only place ``AGE_CAP`` is applied in the wrapper. Callers
must NOT pre-cap before constructing :class:`RedactionRequest` — the
``age_capped`` flag on the result depends on the raw age being visible at
the boundary.
"""

from __future__ import annotations

from bba.deid_redactor.models import AGE_CAP


def apply_age_cap(age_years: int, *, cap: int = AGE_CAP) -> tuple[int, bool]:
    """Cap ``age_years`` at ``cap`` and report whether the cap fired.

    Returns ``(capped_age, was_capped)``. ``was_capped`` is ``True`` iff
    ``age_years > cap``; an age exactly equal to ``cap`` is NOT capped
    (it was already inside the safe range).
    """
    if age_years > cap:
        return cap, True
    return age_years, False
