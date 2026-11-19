"""PERIOPERATIVE is operative blood loss, not "the patient is post-op".

REQNO 68004139: 75 y, post-op day 1 after a knee replacement, Hb 9.3, stable,
no bleeding, transfused on the ward. The LLM cited PERIOPERATIVE and returned
APPROPRIATE; the over-clear guardrail asserted INAPPROPRIATE, so the reviewer
saw reasoning that contradicts the verdict. In the January 2025 run 21 of the
32 guardrail overrides were this pattern.

The guardrail trusts a peri-op clear only for an intra-operative transfusion
or an EBL of at least PERIOP_MIN_EBL_ML (replay._has_windowed_periop_exemption
with DECLARED_USE_PREOP_EXEMPT_ENABLED). The prompt the model reads must state
the same bar, or the two keep disagreeing. These tests pin the prompt wording
because the model follows the text.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from bba.deterministic_classifier import PERIOP_MIN_EBL_ML
from bba.prompt_builder.system_prompt import system_prompt_for

_INDICATION_MODES = ("HB_7_10_REVIEW", "HB_GT_10_OVERRIDE")


def _periop_bullet(prompt: str) -> str:
    start = prompt.index("• PERIOPERATIVE")
    end = prompt.index("•", start + 1)
    return prompt[start:end]


def _guidance() -> str:
    pilot = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
    if str(pilot) not in sys.path:
        sys.path.insert(0, str(pilot))
    spec = importlib.util.spec_from_file_location(
        "pilot_run_llm_leg_periop_scope", pilot / "run_llm_leg.py"
    )
    assert spec is not None and spec.loader is not None
    mod: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return "\n".join(mod._PERIOP_EVIDENCE_GUIDANCE)


@pytest.mark.parametrize("mode", _INDICATION_MODES)
class TestSystemPromptPerioperativeCode:
    def test_requires_intraop_transfusion_or_major_ebl(self, mode: str) -> None:
        bullet = _periop_bullet(system_prompt_for(task_mode=mode, cohort_threshold=7.0))
        assert "intra-operative transfusion" in bullet
        assert f"at least {PERIOP_MIN_EBL_ML} mL" in bullet

    def test_being_post_op_is_not_an_indication(self, mode: str) -> None:
        bullet = _periop_bullet(system_prompt_for(task_mode=mode, cohort_threshold=7.0))
        assert "Being post-operative is NOT by itself an indication" in bullet
        assert "post-op day 1" in bullet

    def test_vague_peri_op_window_wording_is_gone(self, mode: str) -> None:
        prompt = system_prompt_for(task_mode=mode, cohort_threshold=7.0)
        assert "a true peri-operative context" not in prompt
        assert "peri-op context" not in prompt


class TestPilotPeriopGuidance:
    def test_surgery_alone_no_longer_satisfies_perioperative(self) -> None:
        guidance = _guidance()
        assert "SATISFIES the peri-operative indication" not in guidance
        assert "A surgery alone" in guidance

    def test_states_the_intraop_transfusion_or_major_ebl_bar(self) -> None:
        guidance = " ".join(_guidance().split())
        assert "intra-op transfusion or an EBL of at least" in guidance
        assert f"{PERIOP_MIN_EBL_ML} mL satisfies PERIOPERATIVE" in guidance

    def test_free_text_surgery_still_proves_the_operation_happened(self) -> None:
        guidance = " ".join(_guidance().split())
        assert "still proves the operation happened" in guidance
        assert "do NOT write 'no operative procedure documented'" in guidance

    def test_quote_or_deny_and_case_107_are_kept(self) -> None:
        guidance = " ".join(_guidance().split())
        assert "quote-or-deny requirement" in guidance
        assert "OR explicitly state why you reject it" in guidance
        assert "Case 107 / REQNO 68074627" in guidance
