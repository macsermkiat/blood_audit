"""Dev-test consortium helpers (scripts/pilot/dev_consortium).

The consortium reviews what the LLM API returned; it is run from Claude Code and
is not a gate. Code owns the two parts that need no judgment: checking that a
cited quote exists, and combining the reviewers' votes without averaging them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "pilot"
    / "dev_consortium"
    / "consortium.py"
)
_USER = (
    '<evidence id="E16" untrusted="true">CT brain: acute on top chronic subdural '
    "hematoma, 0.3-cm midline shift</evidence>"
)


@pytest.fixture(scope="module")
def consortium() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pilot_dev_consortium", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_verbatim_quote_in_the_cited_block_passes(consortium: ModuleType) -> None:
    cited = [{"source_id": "E16", "quote": "Acute on top chronic  subdural hematoma"}]
    assert consortium.citation_problems(_USER, cited) == []


def test_paraphrased_quote_is_a_citation_problem(consortium: ModuleType) -> None:
    # 68000711: the model "quoted" the CT finding in words the note never used.
    cited = [{"source_id": "E16", "quote": "acute subdural haemorrhage on CT"}]
    assert consortium.citation_problems(_USER, cited) == [
        "E16: quote not found verbatim in that block"
    ]


def test_unknown_evidence_id_is_a_citation_problem(consortium: ModuleType) -> None:
    cited = [{"source_id": "E99", "quote": "anything"}]
    assert consortium.citation_problems(_USER, cited) == ["E99: no such evidence id"]


def _answer(correct: bool, soundness: str = "sound") -> dict[str, object]:
    return {"model_classification_is_correct": correct, "soundness": soundness}


def test_unanimous_reviewers_certify_the_answer(consortium: ModuleType) -> None:
    answers = {"sonnet": _answer(True), "opus": _answer(True), "haiku": _answer(True)}
    assert consortium.consortium_outcome(answers) == "all agree: correct"


def test_a_majority_calling_the_verdict_wrong_is_reported_as_such(
    consortium: ModuleType,
) -> None:
    answers = {
        "sonnet": _answer(False, "unsound"),
        "opus": _answer(False, "unsound"),
        "haiku": _answer(True),
    }
    assert consortium.consortium_outcome(answers) == "majority: wrong verdict"


def test_one_dissenting_reviewer_is_surfaced_not_outvoted(
    consortium: ModuleType,
) -> None:
    # On the real run Haiku called nearly everything sound; a lenient reviewer
    # must not be able to hide what the others found, nor a strict one decide.
    answers = {
        "sonnet": _answer(True),
        "opus": _answer(False, "unsound"),
        "haiku": _answer(True),
    }
    assert consortium.consortium_outcome(answers) == "reviewers disagree"


def test_no_reviews_is_not_a_pass(consortium: ModuleType) -> None:
    assert consortium.consortium_outcome({}) == "no reviews"
