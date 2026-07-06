"""Contract tests for bba.deterministic_classifier.procedure_filter.

The denylist marks minor bedside / diagnostic procedures that never
require a blood transfusion. A perm-cath, tracheostomy, lumbar puncture,
thoracocentesis, paracentesis, arthrocentesis, or arterial/central line
must NOT register a peri-procedural (≤6 h prior) or pre-op crossmatch
(≤72 h upcoming) procedure signal in either pipeline leg — otherwise a
patient who merely had a line placed gets their transfusion auto-cleared.

Tests assert the clinical contract (WHY each code is excluded and WHY a
real operation is not), not the frozenset literal. In particular the
"54" and "38" families are exercised on both sides of the line so a
regression to prefix-matching (which would wrongly deny an exploratory
laparotomy or an aortic resection) fails loudly.
"""

from __future__ import annotations

import pytest

from bba.deterministic_classifier.procedure_filter import (
    NON_BLOOD_PROCEDURE_ICD9,
    is_blood_requiring_procedure,
)

# (dot-stripped ICD-9-CM Vol 3 code, clinical name) — the minor procedures
# the clinicians ruled do NOT count as a blood-requiring procedure.
DENIED_PROCEDURES: tuple[tuple[str, str], ...] = (
    ("311", "temporary tracheostomy"),
    ("3121", "mediastinal tracheostomy"),
    ("3129", "other permanent tracheostomy"),
    ("0331", "lumbar puncture / spinal tap"),
    ("3895", "perm/tunneled dialysis catheter"),
    ("3491", "thoracocentesis"),
    ("5491", "abdominal paracentesis"),
    ("8191", "arthrocentesis"),
    ("3891", "arterial line (arterial catheterization)"),
    ("3893", "central venous catheter, NEC"),
    ("3897", "central venous catheter with guidance"),
)

# Real operations (and unknown codes) that MUST keep registering as
# blood-requiring. The "54"/"38" entries deliberately share a two-digit
# prefix with a denied code to prove the match is code-exact, not prefix.
BLOOD_REQUIRING_PROCEDURES: tuple[tuple[str, str], ...] = (
    ("5411", "exploratory laparotomy (shares '54' with paracentesis 5491)"),
    ("3844", "resection of aorta w/ replacement (shares '38' with the lines)"),
    ("3601", "single-vessel PTCA"),
    ("8154", "total knee replacement"),
    ("9999", "unmapped / unknown code — blood-requiring by default"),
    ("", "empty code — unknown, blood-requiring by default"),
)


class TestDeniedProceduresAreNotBloodRequiring:
    """Each denied minor procedure returns False (not blood-requiring)."""

    @pytest.mark.parametrize(
        "code,name", DENIED_PROCEDURES, ids=[c for c, _ in DENIED_PROCEDURES]
    )
    def test_denied_procedure_is_not_blood_requiring(self, code: str, name: str) -> None:
        assert is_blood_requiring_procedure(code) is False, (
            f"{name} ({code}) must not count as a blood-requiring procedure"
        )

    @pytest.mark.parametrize(
        "code,name", DENIED_PROCEDURES, ids=[c for c, _ in DENIED_PROCEDURES]
    )
    def test_denied_procedure_is_in_denylist(self, code: str, name: str) -> None:
        assert code in NON_BLOOD_PROCEDURE_ICD9, f"{name} ({code}) missing from denylist"


class TestBloodRequiringProcedures:
    """Real operations and unknown codes stay blood-requiring (default True)."""

    @pytest.mark.parametrize(
        "code,name",
        BLOOD_REQUIRING_PROCEDURES,
        ids=[c or "empty" for c, _ in BLOOD_REQUIRING_PROCEDURES],
    )
    def test_blood_requiring(self, code: str, name: str) -> None:
        assert is_blood_requiring_procedure(code) is True, (
            f"{name} ({code!r}) must remain a blood-requiring procedure"
        )


class TestCodeExactNotPrefix:
    """The denylist matches whole dot-stripped codes, never prefixes.

    A prefix bug would deny an exploratory laparotomy (54xx) or an aortic
    resection (38xx) — both major, blood-requiring operations that happen
    to share a two-digit prefix with a denied minor procedure.
    """

    def test_paracentesis_denied_but_laparotomy_allowed(self) -> None:
        assert is_blood_requiring_procedure("5491") is False
        assert is_blood_requiring_procedure("5411") is True

    def test_lines_denied_but_vessel_resection_allowed(self) -> None:
        assert is_blood_requiring_procedure("3891") is False
        assert is_blood_requiring_procedure("3844") is True


class TestNormalization:
    """Dotted and whitespace-padded forms normalize to the stored code.

    OperativeEvent.icd9 is already dot-stripped upstream, but the filter
    normalizes defensively so a caller passing a raw dictionary code
    ("38.93") or a padded value still matches.
    """

    @pytest.mark.parametrize("raw", ["38.93", " 3893 ", "38.93 "])
    def test_dotted_and_padded_central_line_still_denied(self, raw: str) -> None:
        assert is_blood_requiring_procedure(raw) is False

    def test_dotted_laparotomy_still_allowed(self) -> None:
        assert is_blood_requiring_procedure("54.11") is True


class TestDenylistShape:
    """Structural invariants on the exported denylist."""

    def test_is_frozenset_of_stripped_codes(self) -> None:
        assert isinstance(NON_BLOOD_PROCEDURE_ICD9, frozenset)
        assert all("." not in code for code in NON_BLOOD_PROCEDURE_ICD9), (
            "denylist codes must be stored dot-stripped to match OperativeEvent.icd9"
        )

    def test_covers_exactly_the_agreed_clinical_set(self) -> None:
        # The clinician-approved set: tracheostomy, LP, perm dialysis cath,
        # thoracocentesis, paracentesis, arthrocentesis, A-line, C-line.
        expected = {c for c, _ in DENIED_PROCEDURES}
        assert NON_BLOOD_PROCEDURE_ICD9 == expected
