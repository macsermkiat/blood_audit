"""Split a Markdown/MDX source into judgeable prose blocks.

A block is one paragraph or one list. Frontmatter, imports, JSX components,
tables, fenced code, headings, and aside markers are never blocks. Headings
are kept as context on the blocks that follow them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

BlockKind = Literal["paragraph", "list"]

_LIST_LINE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
_JSX_OPEN = re.compile(r"^<[A-Z]")
_JSX_SELF_CLOSED = re.compile(r"/>\s*$")
_JSX_CLOSE = re.compile(r"^</[A-Z][A-Za-z]*>\s*$")
_BLOCKQUOTE = re.compile(r"^>\s?")

DEFAULT_MIN_CHARS = 20


@dataclass(frozen=True)
class Block:
    path: str
    index: int
    kind: BlockKind
    heading: str
    line_start: int
    line_end: int
    text: str


@dataclass(frozen=True)
class _Open:
    kind: BlockKind
    heading: str
    line_start: int
    line_end: int
    lines: tuple[str, ...]


def _close(open_block: _Open | None, path: str, index: int) -> Block | None:
    if open_block is None:
        return None
    if open_block.kind == "paragraph":
        text = " ".join(line.strip() for line in open_block.lines)
    else:
        text = "\n".join(line.rstrip() for line in open_block.lines)
    return Block(
        path=path,
        index=index,
        kind=open_block.kind,
        heading=open_block.heading,
        line_start=open_block.line_start,
        line_end=open_block.line_end,
        text=text,
    )


def _extend(open_block: _Open, line: str, lineno: int) -> _Open:
    return _Open(
        kind=open_block.kind,
        heading=open_block.heading,
        line_start=open_block.line_start,
        line_end=lineno,
        lines=(*open_block.lines, line),
    )


def _strip_frontmatter(lines: list[str]) -> int:
    """Return the index of the first line after YAML frontmatter, else 0."""
    if not lines or lines[0].strip() != "---":
        return 0
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return i + 1
    return 0


def split_blocks(
    source: str, path: str, min_chars: int = DEFAULT_MIN_CHARS
) -> tuple[Block, ...]:
    """Return the prose blocks of ``source`` in document order."""
    lines = source.splitlines()
    start = _strip_frontmatter(lines)
    blocks: list[Block] = []
    open_block: _Open | None = None
    heading = ""
    in_code = False
    in_jsx = False

    def flush() -> None:
        nonlocal open_block
        closed = _close(open_block, path, len(blocks))
        open_block = None
        if closed is not None and len(closed.text) >= min_chars:
            blocks.append(closed)

    for offset, raw in enumerate(lines[start:], start=start):
        lineno = offset + 1
        line = _BLOCKQUOTE.sub("", raw) if raw.startswith(">") else raw
        stripped = line.strip()

        if stripped.startswith("```"):
            flush()
            in_code = not in_code
            continue
        if in_code:
            continue
        if in_jsx:
            if _JSX_SELF_CLOSED.search(stripped) or _JSX_CLOSE.match(stripped):
                in_jsx = False
            continue
        if _JSX_OPEN.match(stripped):
            flush()
            if not (_JSX_SELF_CLOSED.search(stripped) or _JSX_CLOSE.match(stripped)):
                in_jsx = True
            continue
        if not stripped:
            flush()
            continue
        if (
            stripped.startswith("import ")
            or stripped.startswith("|")
            or stripped.startswith(":::")
            or stripped.startswith("<")
        ):
            flush()
            continue
        heading_match = _HEADING.match(stripped)
        if heading_match:
            flush()
            heading = heading_match.group(1)
            continue

        is_list_line = bool(_LIST_LINE.match(line))
        if open_block is None:
            kind: BlockKind = "list" if is_list_line else "paragraph"
            open_block = _Open(kind, heading, lineno, lineno, (line,))
        elif open_block.kind == "list" or not is_list_line:
            open_block = _extend(open_block, line, lineno)
        else:
            flush()
            open_block = _Open("list", heading, lineno, lineno, (line,))
    flush()
    return tuple(blocks)
