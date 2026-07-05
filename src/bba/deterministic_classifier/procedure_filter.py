"""Minor-procedure denylist for the peri-op / pre-op procedure signal.

A blood order is auto-cleared by the deterministic classifier when a
procedure sits close to it in time — ``bypass_peri_procedural`` (a
procedure ≤ 6 h before the order) and ``bypass_pre_op_crossmatch`` (a
procedure ≤ 72 h after). That logic assumes the nearby procedure is an
operation that plausibly needs blood. Several common procedures are NOT:
they are minor bedside / diagnostic taps and line insertions that never
justify a transfusion on their own.

This module is the single source of truth for which ICD-9-CM Vol 3
procedure codes are *excluded* from the procedure-proximity computation.
Both pipeline legs drop denied procedures from ``prior_ops`` /
``upcoming_ops`` before deriving ``proximity_h`` / ``upcoming_h``, so a
perm-cath or a lumbar puncture never registers a peri-op signal (and a
patient with a real operation still does — the operation carries its own
code).

The list is clinician-approved and frozen before scoring; it is sourced
from the standard ICD-9-CM Vol 3 procedure vocabulary, never chosen by
which pilot cases it happens to flip. Codes are stored dot-stripped to
match :attr:`bba.cohort_detector.OperativeEvent.icd9`.

Matching is whole-code exact, never prefix: the "38" family (arterial /
central lines) and the "54" family (paracentesis) each share a two-digit
prefix with major blood-requiring operations (aortic resection 38.44,
exploratory laparotomy 54.11), so a prefix match would wrongly suppress a
genuine surgical signal.
"""

from __future__ import annotations

from collections.abc import Sequence

NON_BLOOD_PROCEDURE_ICD9: frozenset[str] = frozenset(
    {
        # Tracheostomy (temporary / mediastinal / other permanent).
        "311",
        "3121",
        "3129",
        # Lumbar puncture (spinal tap).
        "0331",
        # Perm / tunneled dialysis catheter (venous catheterization for
        # renal dialysis). Generic non-dialysis CVCs are covered separately
        # by the A-line / C-line codes below.
        "3895",
        # Thoracocentesis (pleural tap).
        "3491",
        # Abdominal paracentesis (percutaneous abdominal drainage).
        "5491",
        # Arthrocentesis (joint aspiration).
        "8191",
        # Arterial line (arterial catheterization).
        "3891",
        # Central venous catheter (NEC / with imaging guidance).
        "3893",
        "3897",
    }
)
"""ICD-9-CM Vol 3 procedure codes (dot-stripped) that do NOT count as a
blood-requiring procedure. Clinician-approved minor bedside / diagnostic
procedures; see module docstring for the sourcing and freezing policy."""


def _strip_icd9(code: str) -> str:
    """Return ``code`` without decimal points or surrounding whitespace.

    :attr:`OperativeEvent.icd9` is already dot-stripped upstream; this is a
    defensive normalization so a caller passing a raw dictionary code
    (``"38.93"``) or a padded value still matches the stored form.
    """
    return code.strip().replace(".", "")


def is_blood_requiring_procedure(icd9: str) -> bool:
    """True unless ``icd9`` is one of the denied minor procedures.

    Default is ``True`` (unknown / unmapped codes are treated as
    blood-requiring, the safe direction — an unrecognized operation keeps
    its peri-op signal). Returns ``False`` only for a code in
    :data:`NON_BLOOD_PROCEDURE_ICD9`.
    """
    return _strip_icd9(icd9) not in NON_BLOOD_PROCEDURE_ICD9


__all__: Sequence[str] = (
    "NON_BLOOD_PROCEDURE_ICD9",
    "is_blood_requiring_procedure",
)
