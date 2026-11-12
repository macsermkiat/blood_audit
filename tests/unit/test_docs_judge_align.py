"""Tests for Jev Choice-based EN/TH block alignment (scripts/docs_judge/align.py).

WHY these tests exist:
- Heading-ordinal pairing mispaired condensed Thai pages, so parity Nos
  pointed at the wrong paragraph. Alignment now asks Jev to SELECT the
  Thai counterpart from the page's blocks (select, don't guess), with an
  explicit 'none' option. Code owns the rest: the probability gate, the
  one-to-one rule (two EN blocks cannot share one TH block), and the
  unaligned list. Each test pins one of those code-owned rules.
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

from docs_judge.align import ALIGN_THRESHOLD, align_blocks  # noqa: E402
from docs_judge.blocks import Block  # noqa: E402


def _block(path: str, index: int, text: str) -> Block:
    return Block(path, index, "paragraph", "H", index + 1, index + 1, text)


class FakeClient:
    """Answers the alignment Choice from a {en_text: (choice, p)} table."""

    def __init__(self, table: Mapping[str, tuple[str, float]]) -> None:
        self.table = table
        self.states: list[Mapping[str, Any]] = []

    def ask(
        self, state: Mapping[str, Any], questions: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.states.append(state)
        choice, p = self.table[state["en"]]
        options = list(questions["match"]["criteria"])
        probs = {
            o: (p if o == choice else (1 - p) / (len(options) - 1)) for o in options
        }
        return {
            "answers": {
                "match": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": p,
                    "probabilities": probs,
                }
            }
        }


def test_state_carries_all_thai_candidates_and_a_none_option() -> None:
    en = (_block("en/p.mdx", 0, "english intro"),)
    th = (_block("th/p.mdx", 0, "thai intro"), _block("th/p.mdx", 1, "thai body"))
    client = FakeClient({"english intro": ("c0", 0.9)})
    aligned, _ = align_blocks(client, en, th)
    assert client.states[0]["candidates"] == {"c0": "thai intro", "c1": "thai body"}
    assert aligned[0].th is not None and aligned[0].th.index == 0


def test_none_choice_and_low_probability_leave_the_block_unaligned() -> None:
    en = (_block("en/p.mdx", 0, "only in english"), _block("en/p.mdx", 1, "weak match"))
    th = (_block("th/p.mdx", 0, "thai body"),)
    client = FakeClient({"only in english": ("none", 0.8), "weak match": ("c0", 0.3)})
    aligned, unaligned = align_blocks(client, en, th)
    assert all(a.th is None for a in aligned)
    assert [b.index for b in unaligned] == [0, 1]
    assert ALIGN_THRESHOLD > 0.3


def test_two_english_blocks_choosing_one_thai_block_keep_only_the_stronger() -> None:
    en = (_block("en/p.mdx", 0, "first"), _block("en/p.mdx", 1, "second"))
    th = (_block("th/p.mdx", 0, "thai"),)
    client = FakeClient({"first": ("c0", 0.6), "second": ("c0", 0.95)})
    aligned, unaligned = align_blocks(client, en, th)
    kept = [(a.en.index, a.th.index) for a in aligned if a.th is not None]
    assert kept == [(1, 0)]
    assert [b.index for b in unaligned] == [0]


def test_empty_thai_page_asks_nothing() -> None:
    en = (_block("en/p.mdx", 0, "english"),)
    client = FakeClient({})
    aligned, unaligned = align_blocks(client, en, ())
    assert aligned == () and [b.index for b in unaligned] == [0]
    assert client.states == []
