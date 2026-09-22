"""Aplastic anaemia on active therapy is an indication, not an exclusion.

Case 19 of the platelet review (68011945): very severe aplastic anaemia on ATG
and cyclosporine, count 2,000 /uL, projected 0, judged INAPPROPRIATE because
exclusion D ("aplastic anemia without active bleeding") barred every
prophylaxis rule. That line copies NICE NG24's generic "chronic bone marrow
failure" carve-out. The disease-specific BSH aplastic anaemia guideline
(Killick 2016; 2024 update) says the opposite for this subgroup: prophylaxis
below 10 x 10^9/L for patients on active therapy aimed at reversing the
thrombocytopenia, and below 20 x 10^9/L during the ATG course or with sepsis.
Clinician ruling 2026-09-23: follow the guideline. Ten orders in the hematology
cohort (counts 1,000 to 9,000, all on ATG or cyclosporine) turn on this.
"""

from __future__ import annotations

import re

from bba.llm_client.transport import _PLATELET_TOOL_INPUT_SCHEMA
from bba.platelet_guardrail.models import PlateletHardSignals
from bba.prompt_builder.system_prompt import platelet_system_prompt


def _prompt() -> str:
    return platelet_system_prompt()


class TestIndicationEight:
    def test_active_therapy_is_a_numbered_indication_at_ten_thousand(self) -> None:
        prompt = _prompt()
        assert re.search(
            r"8\. Aplastic an(a)?emia on active therapy[^.]*<10,000",
            prompt,
            re.IGNORECASE,
        )

    def test_names_the_therapies_that_count_as_active(self) -> None:
        text = _prompt().split("8. Aplastic", 1)[1][:900].lower()
        for therapy in ("atg", "cyclosporine", "eltrombopag", "transplant"):
            assert therapy in text

    def test_twenty_thousand_during_atg_or_sepsis(self) -> None:
        text = _prompt().split("8. Aplastic", 1)[1][:900]
        assert re.search(r"ATG course[^.]*<20,000|<20,000[^.]*ATG course", text)
        assert "sepsis" in text.lower()


class TestExclusionDIsNarrowed:
    def test_exclusion_d_covers_only_untreated_or_stable_disease(self) -> None:
        line = re.search(r"D\. [^\n]*", _prompt())
        assert line is not None
        text = line.group(0).lower()
        assert "aplastic" in text
        assert re.search(r"chronic|stable|not (under|on|receiving) active", text)
        assert "indication 8" in text

    def test_override_sentence_routes_active_therapy_to_indication_eight(self) -> None:
        prompt = _prompt()
        override = prompt.split("EXCLUSION OVERRIDE", 1)[1][:1200]
        assert "NOT" in override and "chemotherapy" in override
        assert "indication 8" in override


class TestHardSignalCoversIt:
    def test_prompt_hard_signal_definition_names_indication_eight(self) -> None:
        # Without this the over-clear guardrail would floor every such clear:
        # a clear needs at least one true hard signal.
        text = _prompt().split("prophylactic_marrow_failure", 1)[1][:400]
        assert "indication 8" in text or "aplastic" in text.lower()

    def test_schema_description_matches(self) -> None:
        desc = _PLATELET_TOOL_INPUT_SCHEMA["properties"]["prophylactic_marrow_failure"][
            "description"
        ]
        assert "aplastic" in desc.lower()

    def test_guardrail_docstring_matches(self) -> None:
        assert (
            "aplastic" in (PlateletHardSignals.__doc__ or "").lower()
            or "aplastic"
            in (
                __import__("bba.platelet_guardrail.models", fromlist=["x"]).__doc__
                or ""
            ).lower()
        )


def test_terminal_line_counts_indication_eight() -> None:
    assert "no indication 4–8 below its threshold" in _prompt()
