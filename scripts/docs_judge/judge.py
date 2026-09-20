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
    align_probability: float


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


def judge_parity(
    client: Client,
    pairs: Sequence[Pair],
    align_probabilities: Sequence[float] | None = None,
    max_workers: int = 8,
) -> tuple[ParityJudgment, ...]:
    """Judge fact parity for each aligned pair.

    ``align_probabilities`` carries how confidently each pair was aligned.
    A low-parity answer on a weakly aligned pair says the blocks do not
    correspond, not that the translation dropped a fact, so the report
    must be able to tell the two apart.
    """
    questions = parity_questions()
    probabilities = (
        list(align_probabilities)
        if align_probabilities is not None
        else [1.0] * len(pairs)
    )

    def one(item: tuple[Pair, float]) -> ParityJudgment:
        pair, align_p = item
        response = client.ask({"en": pair.en.text, "th": pair.th.text}, questions)
        return ParityJudgment(
            pair, float(response["answers"]["parity"]["noul"]), align_p
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return tuple(pool.map(one, zip(pairs, probabilities, strict=True)))
