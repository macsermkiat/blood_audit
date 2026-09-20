"""Calibration set: does Jev separate known slop from the project's own prose?

The slop side is written here on purpose, in the register of Thai and
English LLM boilerplate. The "human" side is sampled from the existing
pages, which is an assumption, not ground truth: the pages were written
with agent help and are exactly what the main run judges. The calibration
therefore answers one narrow question: with the same question set, does
obvious slop land on ``rewrite`` and the corpus prose mostly not?
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

from .blocks import Block
from .judge import Client, judge_blocks

SLOP_TH: tuple[str, ...] = (
    "ในโลกยุคปัจจุบันที่เทคโนโลยีก้าวหน้าอย่างไม่หยุดยั้ง ระบบนี้ไม่เพียงแต่ช่วยยกระดับ"
    "คุณภาพการดูแลผู้ป่วยเท่านั้น แต่ยังเป็นก้าวสำคัญที่สะท้อนถึงความมุ่งมั่นของโรงพยาบาล"
    "ในการเป็นผู้นำด้านนวัตกรรมอีกด้วย",
    "การนำระบบออดิทมาใช้ถือเป็นจุดเปลี่ยนสำคัญที่จะปฏิวัติวงการการให้เลือด สร้างมาตรฐานใหม่"
    "ที่ครอบคลุมทุกมิติ และตอกย้ำถึงบทบาทของโรงพยาบาลในฐานะผู้นำด้านคุณภาพอย่างยั่งยืน",
    "อย่างไรก็ตาม แม้จะมีความท้าทายอยู่บ้าง แต่ด้วยความร่วมมือของทุกฝ่าย เชื่อมั่นได้ว่า"
    "อนาคตของระบบนี้จะสดใสและเปี่ยมไปด้วยโอกาสอย่างไม่มีที่สิ้นสุด",
    "โดยสรุปแล้ว ระบบนี้มีบทบาทสำคัญอย่างยิ่งในการเสริมสร้างความปลอดภัย ประสิทธิภาพ และ"
    "ความโปร่งใส ซึ่งล้วนเป็นรากฐานสำคัญของการดูแลผู้ป่วยที่มีคุณภาพระดับโลก",
    "ทั้งนี้ เป็นที่น่าสังเกตว่าผู้เชี่ยวชาญหลายท่านต่างเห็นพ้องต้องกันว่าแนวทางนี้อาจจะสามารถ"
    "ช่วยเสริมศักยภาพของบุคลากรได้อย่างมีนัยสำคัญในระยะยาว",
    "มาดูกันเลยว่าในส่วนนี้เราจะพาทุกท่านไปสำรวจโลกของการออดิทการให้เลือด ซึ่งเป็นการเดินทาง"
    "ที่น่าตื่นเต้นและเต็มไปด้วยข้อมูลเชิงลึกที่นำไปปฏิบัติได้จริง",
    "ระบบนี้ไม่ใช่แค่เครื่องมือ แต่เป็นพันธมิตรที่จะร่วมเดินทางไปกับทีมแพทย์ ช่วยปลดล็อก"
    "ศักยภาพ และเปลี่ยนโฉมการทำงานให้ราบรื่นไร้รอยต่ออย่างแท้จริง",
    "ด้วยสถาปัตยกรรมที่ล้ำสมัย แข็งแกร่ง และยืดหยุ่น ระบบจึงสามารถรองรับความต้องการที่"
    "หลากหลายได้อย่างครบวงจร สะท้อนถึงวิสัยทัศน์ที่มองไกลของทีมพัฒนา",
    "การให้เลือดอย่างเหมาะสมถือเป็นหัวใจสำคัญของการดูแลผู้ป่วยยุคใหม่ และระบบนี้ก็เปรียบเสมือน"
    "ประภาคารที่ส่องทางให้บุคลากรทางการแพทย์ก้าวไปข้างหน้าอย่างมั่นใจ",
    "หวังว่าข้อมูลนี้จะเป็นประโยชน์ หากมีข้อสงสัยเพิ่มเติม สามารถสอบถามได้ตลอดเวลา และขอให้"
    "ทุกท่านประสบความสำเร็จในการนำระบบไปใช้อย่างราบรื่น",
)

SLOP_EN: tuple[str, ...] = (
    "In today's rapidly evolving healthcare landscape, this groundbreaking "
    "pipeline stands as a testament to the hospital's unwavering commitment to "
    "excellence, not only streamlining transfusion review but also empowering "
    "clinicians to deliver world-class care.",
    "By leveraging cutting-edge technology, the audit serves as a robust "
    "cornerstone for quality improvement, fostering a vibrant culture of "
    "safety, transparency, and accountability across the institution.",
    "Experts widely agree that such systems play a pivotal role in modern "
    "medicine, highlighting the importance of data-driven decision making and "
    "underscoring the transformative potential of automation.",
    "Let's dive in! In this section we will explore the key features, "
    "unpack the architecture, and discover actionable insights that you can "
    "apply right away. Here's what you need to know.",
    "Despite these challenges, the future looks bright. With continued "
    "collaboration, the pipeline is poised to elevate patient outcomes and "
    "set a new paradigm for transfusion stewardship. Exciting times lie ahead.",
)


def _first_prose_blocks(
    blocks_by_page: Mapping[str, Sequence[Block]], language: str, n: int
) -> tuple[Block, ...]:
    picked: list[Block] = []
    for path in sorted(blocks_by_page):
        if f"/{language}/" not in path:
            continue
        paragraphs = [b for b in blocks_by_page[path] if b.kind == "paragraph"]
        if paragraphs:
            picked.append(paragraphs[0])
        if len(picked) == n:
            break
    return tuple(picked)


def _synthetic(texts: Sequence[str], language: str) -> tuple[Block, ...]:
    return tuple(
        Block(f"calibration/{language}/slop", i, "paragraph", "", 0, 0, t)
        for i, t in enumerate(texts)
    )


def _group_stats(
    client: Client, blocks: Sequence[Block], language: str
) -> dict[str, Any]:
    audience = "a hospital quality-improvement committee member"
    judged = judge_blocks(client, blocks, language=language, audience=audience)
    rewrite_p = [
        1.0 - j.verdict_p if j.verdict != "rewrite" else j.verdict_p for j in judged
    ]
    verdicts: dict[str, int] = {}
    for j in judged:
        verdicts[j.verdict] = verdicts.get(j.verdict, 0) + 1
    return {
        "n": len(judged),
        "verdicts": verdicts,
        "p_rewrite_min": round(min(rewrite_p), 3) if rewrite_p else None,
        "p_rewrite_median": round(median(rewrite_p), 3) if rewrite_p else None,
        "p_rewrite_max": round(max(rewrite_p), 3) if rewrite_p else None,
        "tells_per_block": round(sum(len(j.tells) for j in judged) / len(judged), 2)
        if judged
        else None,
    }


def run_calibration(
    client: Client, blocks_by_page: Mapping[str, Sequence[Block]], n_human: int = 10
) -> dict[str, Any]:
    """Judge the slop sets and a corpus sample; return per-group statistics."""
    return {
        "note": (
            "'corpus' groups are the first paragraph of each page, assumed "
            "human-edited; not independent ground truth."
        ),
        "th_slop": _group_stats(client, _synthetic(SLOP_TH, "th"), "th"),
        "th_corpus": _group_stats(
            client, _first_prose_blocks(blocks_by_page, "th", n_human), "th"
        ),
        "en_slop": _group_stats(client, _synthetic(SLOP_EN, "en"), "en"),
        "en_corpus": _group_stats(
            client, _first_prose_blocks(blocks_by_page, "en", n_human // 2), "en"
        ),
    }
