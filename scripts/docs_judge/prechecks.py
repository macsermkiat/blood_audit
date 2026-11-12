"""Code-owned counts of mechanical AI-writing tells.

Jev does not count reliably, so everything a regex can settle is settled
here. The English word list is the tier-1 slice of the humanize kit's
vocabulary; the Thai phrase list is this project's own, built from the
puffery phrases that Thai LLM prose reuses.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

AI_VOCAB_TIER1: tuple[str, ...] = (
    "delve",
    "tapestry",
    "vibrant",
    "crucial",
    "meticulous",
    "seamless",
    "groundbreaking",
    "leverage",
    "synergy",
    "transformative",
    "paramount",
    "multifaceted",
    "myriad",
    "cornerstone",
    "empower",
    "catalyst",
    "nestled",
    "realm",
    "unpack",
    "actionable",
    "impactful",
    "learnings",
    "robust",
    "embark",
    "showcase",
    "foster",
    "garner",
    "interplay",
    "enduring",
    "pivotal",
    "intricate",
    "harness",
    "testament",
    "underscore",
)

THAI_TELLS: tuple[str, ...] = (
    "ในโลกยุคปัจจุบัน",
    "ในยุคปัจจุบัน",
    "อย่างไรก็ตาม",
    "ยกระดับ",
    "ก้าวสำคัญ",
    "สะท้อนถึง",
    "ความมุ่งมั่น",
    "ไม่หยุดยั้ง",
    "อย่างมีนัยสำคัญ",
    "เป็นที่ประจักษ์",
    "ครอบคลุมทุกมิติ",
    "อย่างครบวงจร",
    "มีบทบาทสำคัญ",
    "อย่างยั่งยืน",
    "ปฏิวัติวงการ",
    "เปลี่ยนโฉม",
    "ตอกย้ำ",
)

_DASH = re.compile(r"—|–| -- ")
_BOLD = re.compile(r"\*\*[^*\n]+\*\*")
_CURLY = re.compile(r"[“”‘’]")
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff☀-➿\U0001f900-\U0001f9ff]",
)
_VOCAB = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in AI_VOCAB_TIER1) + r")\b",
    re.IGNORECASE,
)
_NEG_PARALLEL_EN = re.compile(
    r"\bnot\s+(?:only|just|merely|simply)\b[^.\n]*?\bbut\b", re.IGNORECASE
)
_NEG_PARALLEL_TH = re.compile(r"ไม่(?:เพียง(?:แต่)?|ใช่แค่|ได้แค่)[^\n]*?แต่(?:ยัง)?")
_THAI = re.compile("|".join(re.escape(p) for p in THAI_TELLS))
_INLINE_HEADER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\*\*[^*\n]+[:.]\*\*", re.M)
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.M)
_STOPWORDS = frozenset({"a", "an", "and", "or", "the", "of", "to", "in", "for"})


def precheck(text: str) -> Mapping[str, int]:
    """Count each mechanical tell in ``text``; every key is always present."""
    return {
        "dash": len(_DASH.findall(text)),
        "bold": len(_BOLD.findall(text)),
        "curly_quote": len(_CURLY.findall(text)),
        "emoji": len(_EMOJI.findall(text)),
        "ai_vocab": len(_VOCAB.findall(text)),
        "neg_parallel": len(_NEG_PARALLEL_EN.findall(text))
        + len(_NEG_PARALLEL_TH.findall(text)),
        "thai_tell": len(_THAI.findall(text)),
        "inline_header": len(_INLINE_HEADER.findall(text)),
    }


def heading_title_case(source: str) -> list[str]:
    """Return headings where every non-stopword is capitalised (2+ words)."""
    flagged: list[str] = []
    for heading in _HEADING.findall(source):
        words = [w for w in re.split(r"\s+", heading) if w and w.isalpha()]
        content = [w for w in words if w.lower() not in _STOPWORDS]
        if len(words) >= 3 and content and all(w[0].isupper() for w in content):
            flagged.append(heading)
    return flagged
