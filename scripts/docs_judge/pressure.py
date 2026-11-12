"""Refutation pass for non-passing verdicts (llm-as-judge: pressure-test).

For every block whose verdict is not ``reads_human`` Jev answers three
Nouls that try to refute the draft verdict. If all three hold, code
downgrades ``rewrite`` to ``light_edit``: the block is specific, fact-only,
and its only machine-looking traits are mechanical, so a rewrite is not
warranted. ``light_edit`` never changes because its fix is mechanical either
way. The question set is fixed so a re-judge after a fix asks the same
questions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from .judge import BlockJudgment, Client

REFUTE_THRESHOLD = 0.5

_QUESTIONS: Mapping[str, Mapping[str, Any]] = {
    "has_specifics": {
        "type": "noul",
        "instructions": (
            "Does `text` contain at least one concrete, checkable item: an "
            "identifier in backticks, a number with a unit, a file path, a "
            "command, a named flag, or a named rule?"
        ),
    },
    "fact_only": {
        "type": "noul",
        "instructions": (
            "Is every sentence in `text` a statement of fact, rule, behaviour, "
            "or instruction, with no praise of the subject and no claim about "
            "its importance?"
        ),
    },
    "mechanical_only": {
        "type": "noul",
        "instructions": (
            "If the boldface, dashes, and any triplet phrasing were removed from "
            "`text`, would the remaining sentences read as plain human technical "
            "writing with nothing left that looks machine-generated?"
        ),
    },
}


@dataclass(frozen=True)
class PressureResult:
    original_verdict: str
    final_verdict: str
    answers: Mapping[str, float]


def pressure_questions() -> Mapping[str, Mapping[str, Any]]:
    return _QUESTIONS


def _decide(original: str, answers: Mapping[str, float]) -> str:
    refuted = all(p >= REFUTE_THRESHOLD for p in answers.values())
    if original == "rewrite" and refuted:
        return "light_edit"
    return original


def pressure_test(
    client: Client, judgments: Sequence[BlockJudgment], max_workers: int = 8
) -> tuple[tuple[BlockJudgment, PressureResult | None], ...]:
    """Return each judgment (verdict possibly downgraded) with its refutation."""

    def one(j: BlockJudgment) -> tuple[BlockJudgment, PressureResult | None]:
        if j.verdict == "reads_human":
            return j, None
        response = client.ask({"text": j.block.text}, _QUESTIONS)
        answers = {q: float(response["answers"][q]["noul"]) for q in _QUESTIONS}
        final = _decide(j.verdict, answers)
        result = PressureResult(j.verdict, final, answers)
        return replace(j, verdict=final), result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return tuple(pool.map(one, judgments))
