"""Pre-LLM injection scanner.

PRD §38 + issue #21 scope: detect imperative verbs, fake-guideline
patterns, and bilingual Thai/EN jailbreaks inside redacted evidence text.
Flagged rows route to ``NEEDS_REVIEW`` without an LLM call.

Design contract:

* Pure function over post-redaction evidence text. No I/O, no model deps.
* Case-insensitive for ASCII letters; NFC-normalized for Thai before
  matching, so NFD-encoded adversarial inputs produce the same verdict.
* Returns ALL matches (not just the first) so monitoring can count
  per-pattern hit rates. Flagging is the OR over all patterns.
* Shipped catalog satisfies
  :data:`bba.prompt_builder.MIN_REQUIRED_INJECTION_PATTERNS` (>= 20)
  across imperative verbs (EN + TH), fake guidelines, bilingual
  jailbreaks, system-prompt exfiltration, and role-pretend.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import NamedTuple

from bba.prompt_builder.models import (
    EvidenceChunk,
    InjectionCategory,
    InjectionMatch,
    InjectionVerdict,
)


class InjectionPattern(NamedTuple):
    """One named injection-detection pattern."""

    pattern_id: str
    category: InjectionCategory
    regex: str
    language: str


INJECTION_PATTERNS: tuple[InjectionPattern, ...] = (
    # ---- IMPERATIVE_VERB_EN ------------------------------------------------
    # Target noun list includes ``prompt`` / ``system`` so direct system-
    # prompt-bypass attacks ("ignore the system prompt", "forget your
    # system instructions") trip the scanner. Codex review #21 round 1
    # caught the original noun list missing ``prompt``.
    InjectionPattern(
        pattern_id="imp_ignore_v1",
        category=InjectionCategory.IMPERATIVE_VERB_EN,
        regex=(
            r"\bignore\b.{0,40}?\b"
            r"(?:instructions?|rules?|directives?|guidelines?|policy|"
            r"policies|prompt|system)\b"
        ),
        language="en",
    ),
    InjectionPattern(
        pattern_id="imp_disregard_v1",
        category=InjectionCategory.IMPERATIVE_VERB_EN,
        regex=(
            r"\bdisregard\b.{0,40}?\b"
            r"(?:instructions?|rules?|directives?|guidelines?|policy|"
            r"policies|prompt|system)\b"
        ),
        language="en",
    ),
    InjectionPattern(
        pattern_id="imp_forget_v1",
        category=InjectionCategory.IMPERATIVE_VERB_EN,
        regex=(
            r"\bforget\b.{0,40}?\b"
            r"(?:instructions?|rules?|directives?|guidelines?|policy|"
            r"policies|prompt|system)\b"
        ),
        language="en",
    ),
    InjectionPattern(
        pattern_id="imp_from_now_on_v1",
        category=InjectionCategory.IMPERATIVE_VERB_EN,
        regex=r"\bfrom\s+now\s+on\b",
        language="en",
    ),
    InjectionPattern(
        pattern_id="imp_new_instructions_v1",
        category=InjectionCategory.IMPERATIVE_VERB_EN,
        regex=r"\bnew\s+(?:instructions?|rules?|directives?)\b\s*:",
        language="en",
    ),
    # ---- FAKE_GUIDELINE ----------------------------------------------------
    # Real KCMH guideline is "PR 17.2"; a version with 2+ nines, leading
    # 200+ century, or 3+ digit minor is a fabricated reference. Tolerates
    # ICD-10 codes (K92.2 etc.) because the prefix is "PR ".
    InjectionPattern(
        pattern_id="fake_pr_guideline_v1",
        category=InjectionCategory.FAKE_GUIDELINE,
        regex=r"\b(?:kcmh\s+)?pr\s+\d+\.(?:9{2,}|[2-9]\d{2,}|\d{3,})\b",
        language="en",
    ),
    InjectionPattern(
        pattern_id="fake_aabb_guideline_v1",
        category=InjectionCategory.FAKE_GUIDELINE,
        regex=(
            r"\baabb\s+(?:\d{4}\s+)?(?:guideline\s+)\d+\.\d+\b"
            r"|\baabb\s+(?:20[5-9]\d|2[1-9]\d{2})\b"
        ),
        language="en",
    ),
    InjectionPattern(
        pattern_id="fake_iso_guideline_v1",
        category=InjectionCategory.FAKE_GUIDELINE,
        regex=r"\biso\s+\d{5,}\b",
        language="en",
    ),
    InjectionPattern(
        pattern_id="fake_who_guideline_v1",
        category=InjectionCategory.FAKE_GUIDELINE,
        regex=r"\bwho\s+(?:[a-z]+\s+){0,3}guideline\s+\d+\.\d+\b",
        language="en",
    ),
    # Same fabricated-version criteria as the English ``fake_pr_guideline_v1``:
    # only flag minor versions of (a) 2+ trailing nines, (b) 200+ leading
    # century-style digit, or (c) 3+ total digits. The real KCMH guideline
    # is ``PR 17.2`` and must not flag in Thai prose either (codex review
    # #21 round 1 — original pattern matched any decimal version).
    InjectionPattern(
        pattern_id="fake_thai_pr_v1",
        category=InjectionCategory.FAKE_GUIDELINE,
        regex=r"แนวทาง\s*(?:KCMH\s+)?PR\s+\d+\.(?:9{2,}|[2-9]\d{2,}|\d{3,})",
        language="th",
    ),
    # ---- SYSTEM_PROMPT_EXFIL ----------------------------------------------
    InjectionPattern(
        pattern_id="exfil_reveal_prompt_v1",
        category=InjectionCategory.SYSTEM_PROMPT_EXFIL,
        regex=(
            r"\b(?:reveal|show|display|print|output|dump)\b"
            r".{0,30}?\b(?:system\s+)?prompt\b"
        ),
        language="en",
    ),
    InjectionPattern(
        pattern_id="exfil_repeat_initial_v1",
        category=InjectionCategory.SYSTEM_PROMPT_EXFIL,
        regex=(
            r"\brepeat\b.{0,30}?\b(?:initial|first|system|original)\b"
            r"\s+(?:instructions?|prompt)"
        ),
        language="en",
    ),
    InjectionPattern(
        pattern_id="exfil_override_system_v1",
        category=InjectionCategory.SYSTEM_PROMPT_EXFIL,
        regex=r"\boverride\s+(?:your\s+|the\s+)?system\s+prompt\b",
        language="en",
    ),
    # ---- ROLE_PRETEND ------------------------------------------------------
    InjectionPattern(
        pattern_id="role_you_are_now_v1",
        category=InjectionCategory.ROLE_PRETEND,
        regex=r"\byou\s+are\s+now\s+(?:a|an)\s+\w+",
        language="en",
    ),
    InjectionPattern(
        pattern_id="role_act_as_v1",
        category=InjectionCategory.ROLE_PRETEND,
        regex=(
            r"\bact\s+as\s+(?:a|an)\s+"
            r"(?:different|unfiltered|jailbroken|new|fictional|"
            r"unrestricted|uncensored)\b"
        ),
        language="en",
    ),
    InjectionPattern(
        pattern_id="role_pretend_v1",
        category=InjectionCategory.ROLE_PRETEND,
        regex=r"\bpretend\s+to\s+be\s+(?:a|an)\b",
        language="en",
    ),
    # ---- IMPERATIVE_VERB_TH -----------------------------------------------
    InjectionPattern(
        pattern_id="imp_th_ignore_v1",
        category=InjectionCategory.IMPERATIVE_VERB_TH,
        regex=r"ละเลย(?:คำสั่ง|กฎ|ระเบียบ)",
        language="th",
    ),
    InjectionPattern(
        pattern_id="imp_th_skip_v1",
        category=InjectionCategory.IMPERATIVE_VERB_TH,
        regex=r"ข้าม(?:กฎ|คำสั่ง|ระเบียบ)",
        language="th",
    ),
    InjectionPattern(
        pattern_id="imp_th_dont_listen_v1",
        category=InjectionCategory.IMPERATIVE_VERB_TH,
        regex=r"อย่าฟัง(?:คำสั่ง|กฎ|ระเบียบ|ระบบ)",
        language="th",
    ),
    # ---- JAILBREAK_TH ------------------------------------------------------
    InjectionPattern(
        pattern_id="jb_th_unrestricted_ai_v1",
        category=InjectionCategory.JAILBREAK_TH,
        regex=r"AI\s*ที่ไม่มีข้อจำกัด",
        language="th",
    ),
    InjectionPattern(
        pattern_id="jb_th_forget_original_v1",
        category=InjectionCategory.JAILBREAK_TH,
        regex=r"ลืมคำสั่งเดิม",
        language="th",
    ),
    # ---- JAILBREAK_EN ------------------------------------------------------
    InjectionPattern(
        pattern_id="jb_translate_then_comply_v1",
        category=InjectionCategory.JAILBREAK_EN,
        regex=(
            r"\btranslate\s+this\b.{0,80}?"
            r"(?:then\s+comply|ละเลย|ignore|comply\s+with)"
        ),
        language="en",
    ),
)


_COMPILED_PATTERNS: tuple[tuple[InjectionPattern, re.Pattern[str]], ...] = tuple(
    (p, re.compile(p.regex, re.IGNORECASE | re.DOTALL)) for p in INJECTION_PATTERNS
)


def scan_injection(*, evidence_id: str, text: str) -> tuple[InjectionMatch, ...]:
    """Scan one evidence chunk's text against :data:`INJECTION_PATTERNS`."""
    if not text:
        return ()
    nfc_text = unicodedata.normalize("NFC", text)
    matches: list[InjectionMatch] = []
    for pattern, compiled in _COMPILED_PATTERNS:
        for m in compiled.finditer(nfc_text):
            span_text = m.group(0)
            if not span_text:
                continue
            matches.append(
                InjectionMatch(
                    category=pattern.category,
                    pattern_id=pattern.pattern_id,
                    evidence_id=evidence_id,
                    span_text=span_text,
                    start=m.start(),
                    end=m.end(),
                )
            )
    return tuple(matches)


def scan_chunks(chunks: Sequence[EvidenceChunk]) -> InjectionVerdict:
    """Apply :func:`scan_injection` over every chunk and aggregate."""
    all_matches: list[InjectionMatch] = []
    for chunk in chunks:
        all_matches.extend(
            scan_injection(evidence_id=chunk.evidence_id, text=chunk.text)
        )
    return InjectionVerdict(
        flagged=bool(all_matches), matches=tuple(all_matches)
    )
