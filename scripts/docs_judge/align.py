"""Align English blocks to their Thai counterparts with one Jev Choice each.

The Thai site is a condensed translation, so position-based pairing fails.
Instead each English block is shown the Thai page's blocks as named
candidates and Jev selects one or ``none``. Code owns the probability gate
and the one-to-one rule; the model never invents a pairing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .blocks import Block
from .judge import Client

ALIGN_THRESHOLD = 0.5
NONE = "none"


@dataclass(frozen=True)
class Alignment:
    en: Block
    th: Block | None
    probability: float


def _questions(candidate_ids: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
    criteria: dict[str, str] = {
        cid: f"The Thai block stored at `candidates.{cid}` covers the same subject "
        f"and the same facts as `en`, possibly condensed."
        for cid in candidate_ids
    }
    criteria[NONE] = (
        "No candidate corresponds to `en`; its content is absent from the Thai page."
    )
    return {
        "match": {
            "type": "choice",
            "instructions": (
                "`en` is one prose block from an English documentation page. "
                "`candidates` holds every prose block of the Thai translation of "
                "the same page, keyed by id. Which candidate is the Thai "
                "counterpart of `en`: the block that translates or condenses the "
                "same statements? Shared code identifiers and numbers are strong "
                "evidence. Choose `none` when no candidate covers `en`."
            ),
            "criteria": criteria,
        }
    }


def align_blocks(
    client: Client,
    en_blocks: Sequence[Block],
    th_blocks: Sequence[Block],
    max_workers: int = 8,
) -> tuple[tuple[Alignment, ...], tuple[Block, ...]]:
    """Return one Alignment per English block plus the blocks left unaligned."""
    if not th_blocks:
        return (), tuple(en_blocks)
    ids = [f"c{i}" for i in range(len(th_blocks))]
    candidates = {cid: b.text for cid, b in zip(ids, th_blocks, strict=True)}
    by_id = dict(zip(ids, th_blocks, strict=True))
    questions = _questions(ids)

    def one(en: Block) -> Alignment:
        answer = client.ask({"en": en.text, "candidates": candidates}, questions)
        match = answer["answers"]["match"]
        choice = str(match["choice"])
        p = float(match["probabilities"][choice])
        if choice == NONE or p < ALIGN_THRESHOLD:
            return Alignment(en, None, p)
        return Alignment(en, by_id[choice], p)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        raw = tuple(pool.map(one, en_blocks))

    best: dict[int, Alignment] = {}
    for a in raw:
        if a.th is None:
            continue
        current = best.get(a.th.index)
        if current is None or a.probability > current.probability:
            best[a.th.index] = a
    winners = {id(a) for a in best.values()}
    aligned = tuple(
        a if id(a) in winners else Alignment(a.en, None, a.probability) for a in raw
    )
    unaligned = tuple(a.en for a in aligned if a.th is None)
    return aligned, unaligned
