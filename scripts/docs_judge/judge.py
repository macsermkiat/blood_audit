"""Orchestrate block judgments and roll them up in code.

Jev answers per block; this module owns everything that must not drift:
audience mapping, threshold on tells, verdict counting, EN/TH pairing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import mean
from typing import Any, Protocol

from .blocks import Block
from .prechecks import precheck
from .questions import TELL_IDS, VERDICTS, block_questions, parity_questions

TELL_THRESHOLD = 0.5
PARITY_THRESHOLD = 0.5

_AUDIENCES: Mapping[str, str] = {
    "clinical": (
        "a clinician on the hospital's quality-improvement committee or blood "
        "bank staff, comfortable with transfusion medicine, not a programmer"
    ),
    "operators": (
        "hospital IT staff who run scripts and manage exports, not a clinician "
        "and not a developer of this codebase"
    ),
    "developers": (
        "a software engineer new to this repository, comfortable with Python, "
        "unfamiliar with transfusion medicine"
    ),
    "reference": (
        "any reader of the site (clinician, hospital IT, or engineer) looking "
        "up a definition or a field"
    ),
}
_DEFAULT_AUDIENCE = (
    "a mixed audience arriving at the project's front page: clinicians, "
    "hospital IT staff, and engineers"
)


class Client(Protocol):
    def ask(
        self, state: Mapping[str, Any], questions: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class BlockJudgment:
    block: Block
    language: str
    audience: str
    prechecks: Mapping[str, int]
    verdict: str
    verdict_p: float
    verdict_confidence: float
    tells: tuple[tuple[str, float], ...]
    nouls: Mapping[str, float]
    readability: float
    readability_probs: Mapping[str, float]
    input_tokens: int


@dataclass(frozen=True)
class PageSummary:
    path: str
    language: str
    n_blocks: int
    counts: Mapping[str, int]
    top_tells: tuple[tuple[str, int], ...]
    readability_mean: float
    precheck_totals: Mapping[str, int]
    title_case_headings: tuple[str, ...]


@dataclass(frozen=True)
class Pair:
    en: Block
    th: Block


@dataclass(frozen=True)
class ParityJudgment:
    pair: Pair
    parity: float


def audience_for(path: str) -> str:
    for section, audience in _AUDIENCES.items():
        if f"/{section}/" in path:
            return audience
    return _DEFAULT_AUDIENCE


def language_for(path: str) -> str:
    return "th" if "/th/" in path else "en"


def _state(block: Block, language: str, audience: str) -> Mapping[str, Any]:
    return {
        "text": block.text,
        "kind": block.kind,
        "heading": block.heading,
        "language": language,
        "audience": audience,
    }


def _judgment(
    block: Block, language: str, audience: str, response: Mapping[str, Any]
) -> BlockJudgment:
    answers = response["answers"]
    nouls = {tell: float(answers[tell]["noul"]) for tell in TELL_IDS}
    tells = tuple(
        sorted(
            ((t, p) for t, p in nouls.items() if p >= TELL_THRESHOLD),
            key=lambda item: -item[1],
        )
    )
    verdict = answers["verdict"]
    readability = answers["readability"]
    return BlockJudgment(
        block=block,
        language=language,
        audience=audience,
        prechecks=precheck(block.text),
        verdict=str(verdict["choice"]),
        verdict_p=float(verdict["probabilities"][verdict["choice"]]),
        verdict_confidence=float(verdict["confidence"]),
        tells=tells,
        nouls=nouls,
        readability=float(readability["score"]),
        readability_probs={
            k: float(v) for k, v in readability["probabilities"].items()
        },
        input_tokens=int(response.get("usage", {}).get("input_tokens", 0)),
    )


def judge_blocks(
    client: Client,
    blocks: Sequence[Block],
    language: str,
    audience: str,
    max_workers: int = 8,
) -> tuple[BlockJudgment, ...]:
    questions = block_questions()

    def one(block: Block) -> BlockJudgment:
        response = client.ask(_state(block, language, audience), questions)
        return _judgment(block, language, audience, response)

    if max_workers <= 1:
        return tuple(one(b) for b in blocks)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return tuple(pool.map(one, blocks))


def rollup(
    path: str,
    judgments: Sequence[BlockJudgment],
    title_case_headings: Sequence[str] = (),
) -> PageSummary:
    counts = Counter(j.verdict for j in judgments)
    tell_counter = Counter(t for j in judgments for t, _ in j.tells)
    precheck_totals = Counter[str]()
    for j in judgments:
        precheck_totals.update(j.prechecks)
    return PageSummary(
        path=path,
        language=judgments[0].language if judgments else language_for(path),
        n_blocks=len(judgments),
        counts={v: counts.get(v, 0) for v in VERDICTS},
        top_tells=tuple(tell_counter.most_common()),
        readability_mean=mean(j.readability for j in judgments) if judgments else 0.0,
        precheck_totals=dict(precheck_totals),
        title_case_headings=tuple(title_case_headings),
    )


def _heading_ordinals(blocks: Sequence[Block]) -> list[int]:
    ordinals: list[int] = []
    current = -1
    last_heading: str | None = None
    for block in blocks:
        if block.heading != last_heading:
            current += 1
            last_heading = block.heading
        ordinals.append(current)
    return ordinals


def pair_blocks(
    en_blocks: Sequence[Block], th_blocks: Sequence[Block]
) -> tuple[tuple[Pair, ...], tuple[Block, ...]]:
    """Pair EN and TH blocks by heading ordinal, then by position under it."""
    en_groups: dict[int, list[Block]] = {}
    for block, ordinal in zip(en_blocks, _heading_ordinals(en_blocks), strict=True):
        en_groups.setdefault(ordinal, []).append(block)
    th_groups: dict[int, list[Block]] = {}
    for block, ordinal in zip(th_blocks, _heading_ordinals(th_blocks), strict=True):
        th_groups.setdefault(ordinal, []).append(block)
    pairs: list[Pair] = []
    unpaired: list[Block] = []
    # A condensed translation has fewer heading groups; ordinals then stop
    # corresponding after the first divergence, so only the intro group
    # (before the first heading) is paired and everything else is unpaired.
    if len(en_groups) != len(th_groups):
        en_intro = (
            en_groups.get(0, []) if not en_blocks or not en_blocks[0].heading else []
        )
        th_intro = (
            th_groups.get(0, []) if not th_blocks or not th_blocks[0].heading else []
        )
        pairs.extend(Pair(e, t) for e, t in zip(en_intro, th_intro, strict=False))
        paired_ids = {id(p.en) for p in pairs} | {id(p.th) for p in pairs}
        unpaired.extend(b for b in (*en_blocks, *th_blocks) if id(b) not in paired_ids)
        return tuple(pairs), tuple(unpaired)
    for ordinal in sorted(set(en_groups) | set(th_groups)):
        en_group = en_groups.get(ordinal, [])
        th_group = th_groups.get(ordinal, [])
        pairs.extend(Pair(e, t) for e, t in zip(en_group, th_group, strict=False))
        shorter = min(len(en_group), len(th_group))
        unpaired.extend(en_group[shorter:])
        unpaired.extend(th_group[shorter:])
    return tuple(pairs), tuple(unpaired)


def judge_parity(
    client: Client, pairs: Sequence[Pair], max_workers: int = 8
) -> tuple[ParityJudgment, ...]:
    questions = parity_questions()

    def one(pair: Pair) -> ParityJudgment:
        response = client.ask({"en": pair.en.text, "th": pair.th.text}, questions)
        return ParityJudgment(pair, float(response["answers"]["parity"]["noul"]))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return tuple(pool.map(one, pairs))
