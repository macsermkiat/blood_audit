"""Typed questions sent to Jev for one prose block.

Each Noul is one family from Wikipedia's "Signs of AI writing" that needs a
semantic reading (the countable families are in prechecks.py). The Score is
readability for the audience named in state. The Choice is the one holistic
verdict; its options map 1:1 to what the fix loop does with the block.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VERDICTS: tuple[str, ...] = ("reads_human", "light_edit", "rewrite")

_TELLS: Mapping[str, Mapping[str, str]] = {
    "significance": {
        "q": (
            "Does `text` inflate the importance of its subject by calling it a "
            "milestone, a turning point, a testament, a vital or crucial role, or "
            "by tying it to a broader trend or evolving landscape, instead of "
            "stating what it is or does?"
        ),
        "true": (
            "The passage claims significance, legacy, or broader meaning "
            "(English: 'marks a pivotal moment', 'plays a crucial role'; Thai: "
            "'ก้าวสำคัญ', 'สะท้อนถึง', 'ยกระดับ')."
        ),
        "false": "The passage states facts, rules, or steps without claiming importance.",
    },
    "promotional": {
        "q": (
            "Does `text` praise its subject with advertisement-like language "
            "rather than describing it neutrally?"
        ),
        "true": (
            "Words like vibrant, rich, groundbreaking, world-class, seamless, "
            "state-of-the-art, or Thai equivalents such as 'ล้ำสมัย', "
            "'ครบวงจร', 'มุ่งมั่น', are used to sell the subject."
        ),
        "false": "The tone is neutral or plainly technical.",
    },
    "superficial_analysis": {
        "q": (
            "Does `text` end a sentence with a tacked-on participle clause that "
            "asserts meaning without adding information, such as 'highlighting', "
            "'underscoring', 'ensuring', 'reflecting', 'showcasing', or a Thai "
            "tail such as 'ซึ่งสะท้อนถึง', 'ตอกย้ำถึง'?"
        ),
        "true": "At least one sentence carries such a tail.",
        "false": "Sentences end on concrete content or have no such tail.",
    },
    "vague_attribution": {
        "q": (
            "Does `text` attribute a claim to an unnamed authority such as "
            "'experts agree', 'studies show', 'observers note', 'it is widely "
            "regarded', or Thai 'ผู้เชี่ยวชาญระบุว่า', 'เป็นที่ยอมรับกันว่า', "
            "without naming the source?"
        ),
        "true": "A claim rests on an unnamed authority.",
        "false": "Claims are unattributed plain statements or cite a named source.",
    },
    "copula_avoidance": {
        "q": (
            "Does `text` avoid plain 'is', 'has', or 'does' by using 'serves as', "
            "'stands as', 'boasts', 'features', 'represents a', 'acts as', where "
            "the plain verb would say the same thing?"
        ),
        "true": "At least one such substitute verb phrase is used decoratively.",
        "false": "The passage uses plain verbs, or the substitute carries real meaning.",
    },
    "neg_parallel": {
        "q": (
            "Does `text` build a contrast of the form 'not only X but Y', "
            "'not X, it's Y', 'Y rather than X', or Thai 'ไม่เพียงแต่ ... แต่ยัง', "
            "'ไม่ใช่แค่ ... แต่', used for rhetorical effect rather than to "
            "correct a real misconception?"
        ),
        "true": "A telegraphed contrast is used for emphasis.",
        "false": (
            "No such contrast, or the negation corrects a specific wrong "
            "assumption the reader would plausibly hold."
        ),
    },
    "rule_of_three": {
        "q": (
            "Does `text` group items in threes for rhythm (three adjectives, "
            "three examples, three clauses) where the number three is not "
            "dictated by the facts?"
        ),
        "true": "A triplet exists mainly for cadence.",
        "false": (
            "No triplet, or the three items are the actual complete set "
            "(for example three named options or three named files)."
        ),
    },
    "generic_conclusion": {
        "q": (
            "Does `text` end with a generic wrap-up: a summary that restates the "
            "section, an upbeat future outlook, 'challenges remain', 'in "
            "conclusion', 'overall', or Thai 'โดยสรุป', 'ในอนาคต', 'อย่างไรก็ตาม' "
            "used as a closing flourish?"
        ),
        "true": "The passage closes on a recap or an outlook instead of a concrete point.",
        "false": "The passage ends on a concrete fact, rule, step, or example.",
    },
    "chatbot_artifact": {
        "q": (
            "Does `text` contain phrasing addressed to a chat user or announcing "
            "what the writing will do: 'let's dive in', 'in this section we will', "
            "'here's what you need to know', 'I hope this helps', 'certainly', or "
            "Thai 'มาดูกันเลย', 'ในส่วนนี้เราจะ', 'หวังว่าจะเป็นประโยชน์'?"
        ),
        "true": "Chat or signposting phrasing is present.",
        "false": "The passage states its content directly.",
    },
    "filler_hedging": {
        "q": (
            "Does `text` pad sentences with filler or stacked hedges: 'it is "
            "important to note', 'in order to', 'due to the fact that', 'could "
            "potentially possibly', 'essentially', 'basically', or Thai 'ทั้งนี้', "
            "'โดยทั่วไปแล้ว', 'เป็นที่น่าสังเกตว่า', 'อาจจะสามารถ'?"
        ),
        "true": "At least one filler phrase or a stack of two hedges is present.",
        "false": "Sentences are direct; a single hedge that carries real uncertainty is fine.",
    },
}

TELL_IDS: tuple[str, ...] = tuple(_TELLS)

_READABILITY_LEVELS: tuple[str, ...] = (
    "Cannot follow: key terms are undefined for this reader, or the sentence "
    "structure is so tangled that the meaning is unclear on a second read.",
    "Follows with effort: needs a second read, or has to look up one or two "
    "terms that the passage uses without defining.",
    "Follows on first read: the reader knows what the passage says and what "
    "to do with it after one pass.",
)

_VERDICT_CRITERIA: Mapping[str, str] = {
    "reads_human": (
        "Plain, specific, and free of the AI-writing signs above. Leave the "
        "passage as it is."
    ),
    "light_edit": (
        "The content is fine but one or two mechanical tells remain (a "
        "decorative dash, one filler phrase, one boldface run, one triplet). "
        "A small edit fixes it without rewriting sentences."
    ),
    "rewrite": (
        "Several signs co-occur, or the passage puffs importance, praises "
        "its subject, or buries the point in filler. The sentences need to "
        "be rewritten."
    ),
}


def block_questions() -> Mapping[str, Mapping[str, Any]]:
    """Return the full question map for one block.

    State must carry ``text``, ``language``, ``audience``, ``kind``, and
    ``heading``. Question ids are stable so a re-judge after a rewrite asks
    the same set.
    """
    questions: dict[str, Mapping[str, Any]] = {
        tell: {
            "type": "noul",
            "instructions": spec["q"],
            "criteria": {"true": spec["true"], "false": spec["false"]},
        }
        for tell, spec in _TELLS.items()
    }
    questions["readability"] = {
        "type": "score",
        "instructions": (
            "How easily would the reader described in `audience` follow `text` "
            "on a first read? `text` is one block (kind given in `kind`) under "
            "the heading in `heading` of a documentation page written in the "
            "language given in `language`. Judge the block on its own; do not "
            "penalise it for terms the heading or an earlier block would define."
        ),
        "criteria": list(_READABILITY_LEVELS),
    }
    questions["verdict"] = {
        "type": "choice",
        "instructions": (
            "Judging `text` as a whole against Wikipedia's 'Signs of AI writing' "
            "guide (significance inflation, promotional tone, superficial "
            "participle tails, vague attribution, copula avoidance, negative "
            "parallelism, rule of three, generic conclusions, chatbot phrasing, "
            "filler and hedging), which action fits the block? Technical terms, "
            "code identifiers, and clinical vocabulary are not signs of AI writing."
        ),
        "criteria": dict(_VERDICT_CRITERIA),
    }
    return questions


def parity_questions() -> Mapping[str, Mapping[str, Any]]:
    """Question for one English/Thai block pair; state carries ``en`` and ``th``."""
    return {
        "parity": {
            "type": "noul",
            "instructions": (
                "Does the Thai passage `th` state the same facts, rules, numbers, "
                "and identifiers as the English passage `en`, with nothing "
                "material added or dropped? Wording, sentence order, and "
                "explanatory glosses of English terms may differ."
            ),
            "criteria": {
                "true": "Every material fact in `en` appears in `th` and vice versa.",
                "false": (
                    "A fact, number, identifier, condition, or warning present in "
                    "one passage is missing or contradicted in the other."
                ),
            },
        }
    }
