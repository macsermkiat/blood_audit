"""Tests for scripts/docs_judge/blocks.py (MDX -> judgeable prose blocks).

WHY these tests exist:
- The judge only sees what the splitter hands it. If frontmatter, imports,
  JSX components, tables, or code fences leak into a block, Jev judges markup
  instead of prose and the verdicts are noise. If prose inside a Starlight
  aside is dropped, whole warnings go unjudged. Each test pins one of those
  boundaries.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from docs_judge.blocks import Block, split_blocks  # noqa: E402

SAMPLE = """---
title: What this is
description: A post-hoc audit pipeline.
---

import AtAGlance from "../../../../components/AtAGlance.astro";
import QiNotSamdBanner from "../../../../components/QiNotSamdBanner.astro";

<QiNotSamdBanner />

<AtAGlance
  items={[
    { label: "Audience", value: "QI committee, blood bank, and hospital IT." },
  ]}
/>

## In one paragraph

blood_audit reviews inpatient adult RBC transfusions at Chulalongkorn
Hospital, after they are complete.

Rows that the deterministic classifier routes to the LLM go through six
fail-closed quote-grounding layers.

## What it is not

- **Not Software as a Medical Device.** Nothing here influences an order.
- **Not real-time.** No subscriber emits clinician alerts.

:::caution[PHI]
Input is real encrypted patient data. Do not commit outputs.
:::

| Cohort | Threshold |
| --- | --- |
| DEFAULT | 7.0 |

```bash
uv sync
```

> **Status:** 30 modules under `src/bba/`.
> - **Phase 1 (RBC)** is feature-complete at module level.

1. First step happens here.
2. Second step follows, with detail
   continued on an indented line.

Short.
"""


def _texts(blocks: tuple[Block, ...]) -> list[str]:
    return [b.text for b in blocks]


def test_frontmatter_imports_jsx_tables_and_code_are_not_blocks() -> None:
    blocks = split_blocks(SAMPLE, path="en/clinical/what-this-is.mdx")
    joined = "\n".join(_texts(blocks))
    assert "title:" not in joined
    assert "import AtAGlance" not in joined
    assert "<AtAGlance" not in joined
    assert "label:" not in joined
    assert "| Cohort" not in joined
    assert "uv sync" not in joined


def test_paragraphs_lists_and_aside_prose_become_blocks_with_headings() -> None:
    blocks = split_blocks(SAMPLE, path="en/clinical/what-this-is.mdx")
    kinds = [(b.kind, b.heading) for b in blocks]
    assert ("paragraph", "In one paragraph") in kinds
    assert ("list", "What it is not") in kinds
    assert ("paragraph", "What it is not") in kinds  # aside body
    aside = next(b for b in blocks if b.text.startswith("Input is real"))
    assert ":::" not in aside.text


def test_paragraph_lines_are_joined_and_line_numbers_are_one_based() -> None:
    blocks = split_blocks(SAMPLE, path="p.mdx")
    first = next(b for b in blocks if b.text.startswith("blood_audit reviews"))
    assert "Chulalongkorn Hospital, after" in first.text
    lines = SAMPLE.splitlines()
    assert lines[first.line_start - 1].startswith("blood_audit reviews")
    assert lines[first.line_end - 1].startswith("Hospital, after")


def test_blockquote_prefix_is_stripped_and_nested_list_stays_one_block() -> None:
    blocks = split_blocks(SAMPLE, path="README.md")
    quote = next(b for b in blocks if "Status:" in b.text)
    assert not quote.text.startswith(">")
    numbered = next(b for b in blocks if b.text.startswith("1. First step"))
    assert numbered.kind == "list"
    assert "continued on an indented line" in numbered.text
    assert numbered.line_end - numbered.line_start == 2


def test_blocks_shorter_than_min_chars_are_dropped() -> None:
    blocks = split_blocks(SAMPLE, path="p.mdx", min_chars=20)
    assert "Short." not in _texts(blocks)
    kept = split_blocks(SAMPLE, path="p.mdx", min_chars=1)
    assert "Short." in _texts(kept)


def test_block_index_is_sequential_and_path_is_recorded() -> None:
    blocks = split_blocks(SAMPLE, path="en/x.mdx")
    assert [b.index for b in blocks] == list(range(len(blocks)))
    assert {b.path for b in blocks} == {"en/x.mdx"}
