"""Tests for scripts/docs_judge/judge.py and report.py (orchestration + rollup).

WHY these tests exist:
- Audience is passed as state per section; a wrong mapping judges a
  clinician page as if for engineers, and the readability verdicts become
  meaningless. The mapping is pinned.
- Page rollup is code-owned (llm-as-judge: never let the model aggregate).
  Verdict counts, the tells list (Nouls at or above threshold), and the
  parity pairing must be deterministic given fixed answers.
- The EN/TH pairing must never pair blocks across different heading
  ordinals, otherwise a parity "No" points at the wrong paragraph.
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
from docs_judge.judge import (  # noqa: E402
    TELL_THRESHOLD,
    audience_for,
    judge_blocks,
    language_for,
    pair_blocks,
    rollup,
)
from docs_judge.questions import TELL_IDS  # noqa: E402
from docs_judge.report import render_markdown  # noqa: E402


class FakeClient:
    """Returns canned answers keyed by the block text; records states seen."""

    def __init__(self, canned: Mapping[str, Mapping[str, Any]]) -> None:
        self.canned = canned
        self.states: list[Mapping[str, Any]] = []

    def ask(
        self, state: Mapping[str, Any], questions: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.states.append(state)
        text = state.get("text", state.get("th", ""))
        answers = dict(self.canned[text])
        for qid in questions:
            answers.setdefault(qid, {"type": "noul", "noul": 0.0})
        return {"model": "fake", "answers": answers, "usage": {"input_tokens": 10}}


def _block(path: str, index: int, text: str, heading: str = "H") -> Block:
    return Block(path, index, "paragraph", heading, index + 1, index + 1, text)


def _answers(verdict: str, tells: Mapping[str, float], read: float) -> dict[str, Any]:
    probs = {"reads_human": 0.0, "light_edit": 0.0, "rewrite": 0.0}
    probs[verdict] = 1.0
    out: dict[str, Any] = {
        "verdict": {
            "type": "choice",
            "choice": verdict,
            "confidence": 0.9,
            "probabilities": probs,
        },
        "readability": {
            "type": "score",
            "score": read,
            "confidence": 0.5,
            "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0},
        },
    }
    for tell, p in tells.items():
        out[tell] = {"type": "noul", "noul": p}
    return out


def test_audience_and_language_follow_the_path() -> None:
    assert "clinic" in audience_for("docs/src/content/docs/en/clinical/x.mdx").lower()
    assert "IT" in audience_for("docs/src/content/docs/th/operators/x.mdx")
    assert "engineer" in audience_for("docs/src/content/docs/en/developers/x.mdx")
    assert audience_for("README.md") != ""
    assert language_for("docs/src/content/docs/th/index.mdx") == "th"
    assert language_for("docs/src/content/docs/en/index.mdx") == "en"
    assert language_for("README.md") == "en"


def test_judge_blocks_sends_audience_and_collects_tells_above_threshold() -> None:
    puff = "This groundbreaking system — a testament to innovation."
    plain = "The pipeline reads HOSxP exports after transfusion."
    client = FakeClient(
        {
            puff: _answers("rewrite", {"significance": 0.9, "promotional": 0.6}, 1.2),
            plain: _answers("reads_human", {"significance": 0.05}, 1.9),
        }
    )
    blocks = (
        _block("en/clinical/p.mdx", 0, puff),
        _block("en/clinical/p.mdx", 1, plain),
    )
    judged = judge_blocks(client, blocks, language="en", audience="QI clinicians")
    assert client.states[0]["audience"] == "QI clinicians"
    assert client.states[0]["language"] == "en"
    first, second = judged
    assert first.verdict == "rewrite"
    assert dict(first.tells) == {"significance": 0.9, "promotional": 0.6}
    assert all(p >= TELL_THRESHOLD for _, p in first.tells)
    assert first.prechecks["dash"] == 1 and first.prechecks["ai_vocab"] == 2
    assert second.verdict == "reads_human" and second.tells == ()
    assert second.readability == 1.9


def test_rollup_counts_verdicts_and_ranks_tells_in_code() -> None:
    a = "alpha text that is long enough to judge"
    b = "beta text that is long enough to judge"
    c = "gamma text that is long enough to judge"
    client = FakeClient(
        {
            a: _answers("rewrite", {"significance": 0.8, "rule_of_three": 0.7}, 1.0),
            b: _answers("light_edit", {"significance": 0.6}, 1.5),
            c: _answers("reads_human", {}, 2.0),
        }
    )
    blocks = tuple(_block("en/x.mdx", i, t) for i, t in enumerate((a, b, c)))
    judged = judge_blocks(client, blocks, language="en", audience="anyone")
    page = rollup("en/x.mdx", judged)
    assert page.counts == {"reads_human": 1, "light_edit": 1, "rewrite": 1}
    assert page.top_tells[0] == ("significance", 2)
    assert page.readability_mean == 1.5
    assert page.n_blocks == 3


def test_pair_blocks_pairs_by_heading_ordinal_and_reports_unpaired() -> None:
    en = (
        _block("en/p.mdx", 0, "en intro paragraph long enough", "Intro"),
        _block("en/p.mdx", 1, "en second para long enough", "Intro"),
        _block("en/p.mdx", 2, "en details paragraph long enough", "Details"),
    )
    th = (
        _block("th/p.mdx", 0, "th intro paragraph long enough", "บทนำ"),
        _block("th/p.mdx", 1, "th details paragraph long enough", "รายละเอียด"),
    )
    pairs, unpaired = pair_blocks(en, th)
    assert [(p.en.index, p.th.index) for p in pairs] == [(0, 0), (2, 1)]
    assert [b.index for b in unpaired] == [1]


def test_render_markdown_lists_pages_and_flagged_blocks() -> None:
    puff = "This groundbreaking system — a testament to innovation."
    client = FakeClient({puff: _answers("rewrite", {"significance": 0.9}, 1.0)})
    judged = judge_blocks(client, (_block("en/x.mdx", 0, puff),), "en", "anyone")
    page = rollup("en/x.mdx", judged)
    md = render_markdown(pages=(page,), judgments=judged, parity=(), calibration=None)
    assert "en/x.mdx" in md
    assert "rewrite" in md
    assert "significance" in md
    assert "groundbreaking" in md
    assert set(TELL_IDS) >= {"significance", "promotional", "rule_of_three"}


def test_pair_blocks_with_condensed_translation_pairs_only_the_intro() -> None:
    en = (
        _block("en/p.mdx", 0, "en intro paragraph long enough", ""),
        _block("en/p.mdx", 1, "en details paragraph long enough", "Details"),
        _block("en/p.mdx", 2, "en extra section long enough", "Extra"),
    )
    th = (
        _block("th/p.mdx", 0, "th intro paragraph long enough", ""),
        _block("th/p.mdx", 1, "th condensed body long enough", "รายละเอียด"),
    )
    pairs, unpaired = pair_blocks(en, th)
    assert [(p.en.index, p.th.index) for p in pairs] == [(0, 0)]
    assert sorted((b.path, b.index) for b in unpaired) == [
        ("en/p.mdx", 1),
        ("en/p.mdx", 2),
        ("th/p.mdx", 1),
    ]
