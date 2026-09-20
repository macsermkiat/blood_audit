"""Tests for scripts/docs_judge/prechecks.py (code-owned AI-writing tells).

WHY these tests exist:
- Jev cannot count (TypeSafe jaggedness page, "Math and Numbers"), so every
  countable Wikipedia sign lives here, not in a question. If a counter is
  wrong the report under- or over-states the mechanical tells, and the fix
  loop targets the wrong blocks. Thai has its own phrase tells that the
  English-only sloplint scanner cannot see; the Thai test pins that gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from docs_judge.prechecks import heading_title_case, precheck  # noqa: E402


def test_dashes_bold_and_curly_quotes_are_counted() -> None:
    text = "A — B – C -- D **bold** and **more** “quoted” ‘x’"
    result = precheck(text)
    assert result["dash"] == 3
    assert result["bold"] == 2
    assert result["curly_quote"] == 4


def test_tier1_ai_vocabulary_counts_whole_words_case_insensitively() -> None:
    text = "We Delve into a robust tapestry; delving is not counted."
    result = precheck(text)
    assert result["ai_vocab"] == 3


def test_english_negative_parallelism_is_detected() -> None:
    assert precheck("It is not only fast but also safe.")["neg_parallel"] == 1
    assert precheck("It is fast and safe.")["neg_parallel"] == 0


def test_thai_tells_and_thai_negative_parallelism_are_detected() -> None:
    text = "ในโลกยุคปัจจุบัน ระบบนี้ไม่เพียงแต่ยกระดับคุณภาพ แต่ยังสะท้อนถึงความมุ่งมั่นอีกด้วย"
    result = precheck(text)
    assert result["neg_parallel"] == 1
    assert result["thai_tell"] >= 3


def test_plain_thai_prose_has_no_thai_tells() -> None:
    text = "ระบบอ่านข้อมูลที่ส่งออกจาก HOSxP หลังการให้เลือดเสร็จสิ้นแล้ว"
    assert precheck(text)["thai_tell"] == 0


def test_inline_header_list_items_are_counted() -> None:
    text = "- **Performance:** Performance is better.\n- plain item\n- **Safety.** x"
    assert precheck(text)["inline_header"] == 2


def test_emoji_is_counted() -> None:
    assert precheck("Launch 🚀 now ✅")["emoji"] == 2
    assert precheck("no emoji here")["emoji"] == 0


def test_heading_title_case_flags_multiword_capitalised_headings() -> None:
    source = "## Strategic Negotiations And Partnerships\n## What this is\n# Hb\n"
    assert heading_title_case(source) == ["Strategic Negotiations And Partnerships"]
