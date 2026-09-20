"""Tests for scripts/docs_judge/pressure.py (refutation of non-passing verdicts).

WHY these tests exist:
- The downgrade rule is code-owned policy: only a fully refuted ``rewrite``
  moves to ``light_edit``; a single failed refutation keeps the draft
  verdict; passing blocks are never sent to the model at all. If any of
  those drift, the fix loop either rewrites plain prose or skips real slop.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from docs_judge.blocks import Block  # noqa: E402
from docs_judge.judge import BlockJudgment  # noqa: E402
from docs_judge.pressure import pressure_test  # noqa: E402


class FakeClient:
    def __init__(self, answers: Mapping[str, float]) -> None:
        self.answers = answers
        self.calls = 0

    def ask(
        self, state: Mapping[str, Any], questions: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.calls += 1
        return {
            "answers": {q: {"type": "noul", "noul": self.answers[q]} for q in questions}
        }


def _judgment(verdict: str) -> BlockJudgment:
    block = Block("en/x.mdx", 0, "paragraph", "H", 1, 1, "some text long enough")
    return BlockJudgment(
        block=block,
        language="en",
        audience="anyone",
        prechecks={},
        verdict=verdict,
        verdict_p=0.6,
        verdict_confidence=0.5,
        tells=(),
        nouls={},
        readability=1.0,
        readability_probs={},
        input_tokens=0,
    )


def test_fully_refuted_rewrite_is_downgraded_to_light_edit() -> None:
    client = FakeClient(
        {"has_specifics": 0.9, "fact_only": 0.8, "mechanical_only": 0.7}
    )
    ((judged, result),) = pressure_test(client, (_judgment("rewrite"),))
    assert judged.verdict == "light_edit"
    assert result is not None and result.original_verdict == "rewrite"


def test_one_failed_refutation_keeps_rewrite() -> None:
    client = FakeClient(
        {"has_specifics": 0.9, "fact_only": 0.2, "mechanical_only": 0.9}
    )
    ((judged, result),) = pressure_test(client, (_judgment("rewrite"),))
    assert judged.verdict == "rewrite"
    assert result is not None and result.answers["fact_only"] == 0.2


def test_light_edit_is_recorded_but_never_changed() -> None:
    client = FakeClient(
        {"has_specifics": 0.9, "fact_only": 0.9, "mechanical_only": 0.9}
    )
    ((judged, result),) = pressure_test(client, (_judgment("light_edit"),))
    assert judged.verdict == "light_edit"
    assert result is not None


def test_passing_blocks_are_not_sent_to_the_model() -> None:
    client = FakeClient(
        {"has_specifics": 0.0, "fact_only": 0.0, "mechanical_only": 0.0}
    )
    ((judged, result),) = pressure_test(client, (_judgment("reads_human"),))
    assert result is None and judged.verdict == "reads_human"
    assert client.calls == 0
