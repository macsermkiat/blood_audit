"""Dev-test consortium: review what the LLM API returned, from Claude Code.

NOT a pipeline step and NOT a gate (user ruling 2026-09-21). It answers "are the
answers the API gave us sound and correct?" while a prompt, schema or evidence
change is being tested. The reviewers are Claude Code subagents on different
models (and optionally the Codex CLI), so it runs on the subscription and makes
no Anthropic API call.

    # 1. bundles: evidence the model saw + its full answer, from the audit store
    python scripts/pilot/dev_consortium/consortium.py build [--reqno-file FILE]
    # 2. in Claude Code: one subagent per reviewer/batch, each told to read
    #    INSTRUCTIONS.md and write answers/<reviewer>/<reqno>.json
    # 3. combine the votes (code counts; dissent is shown, never averaged)
    python scripts/pilot/dev_consortium/consortium.py combine

Everything lands in ``$BBA_PILOT_WORK_DIR/dev_consortium``. The bundles hold the
de-identified notes the model saw: sending them to a reviewer outside Anthropic
(for example Codex) is a decision for the data owner, per run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bba.audit_store import AuditStore
from bba.audit_store.models import AuditStoreConfig

WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
OUT = WORK / "dev_consortium"
HERE = Path(__file__).resolve().parent
_EVIDENCE_RE = re.compile(r'<evidence id="(E\d+)"[^>]*>(.*?)</evidence>', re.S)


def _plain(value: Any) -> Any:
    """Audit-store payloads are read-only mappings; make them JSON-serialisable."""
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def citation_problems(user_text: str, indications: Sequence[Any]) -> list[str]:
    """Code check: every cited quote must appear verbatim in the block it cites."""
    evidence = {m.group(1): m.group(2) for m in _EVIDENCE_RE.finditer(user_text)}
    problems: list[str] = []
    for item in indications:
        if not isinstance(item, Mapping):
            problems.append("indication is not an object")
            continue
        source_id = str(item.get("source_id") or "")
        quote = _normalise(str(item.get("quote") or ""))
        if source_id not in evidence:
            problems.append(f"{source_id or '(no source_id)'}: no such evidence id")
        elif quote and quote not in _normalise(evidence[source_id]):
            problems.append(f"{source_id}: quote not found verbatim in that block")
    return problems


def consortium_outcome(answers: Mapping[str, Mapping[str, Any]]) -> str:
    """How the reviewers' independent answers combine. Never an average."""
    if not answers:
        return "no reviews"
    wrong = sum(
        a.get("model_classification_is_correct") is False for a in answers.values()
    )
    unsound = sum(a.get("soundness") == "unsound" for a in answers.values())
    if wrong == 0 and unsound == 0:
        return "all agree: correct"
    if wrong * 2 > len(answers):
        return "majority: wrong verdict"
    return "reviewers disagree"


def _text_blocks(blocks: Any) -> str:
    if isinstance(blocks, str):
        return blocks
    return "\n\n".join(
        str(b.get("text", "")) for b in blocks or () if isinstance(b, Mapping)
    )


def build(reqnos: set[str] | None) -> None:
    report = {
        r["audit_id"]: r
        for r in json.loads((WORK / "llm_report.json").read_text())
        if r.get("llm_final") and (reqnos is None or r["reqno"] in reqnos)
    }
    store = AuditStore(
        AuditStoreConfig(root_dir=WORK / "data" / "audit_store", code_version="dev")
    )
    rows: dict[str, Any] = {}
    for row in sorted(store.read_audit_results(), key=lambda r: r.run_timestamp):
        if row.audit_id in report:
            rows[row.audit_id] = row
    calls: dict[tuple[str, str], Any] = {}
    for call in store.read_llm_calls():
        key = (call.audit_id, call.run_id)
        if key not in calls or call.request_timestamp > calls[key].request_timestamp:
            calls[key] = call
    (OUT / "review").mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "INSTRUCTIONS.md", OUT / "INSTRUCTIONS.md")
    citations: dict[str, list[str]] = {}
    for audit_id, row in rows.items():
        call = calls.get((audit_id, row.run_id))
        if call is None:
            continue
        request, response = _plain(call.request_json), _plain(call.response_json)
        answer = next(
            (
                b["input"]
                for b in response.get("content", ())
                if b.get("type") == "tool_use"
            ),
            None,
        )
        if answer is None:
            continue
        user = _text_blocks(request["messages"][0]["content"])
        citations[row.reqno] = citation_problems(user, answer.get("indications") or ())
        (OUT / "review" / f"{row.reqno}.review.txt").write_text(
            "=== SYSTEM INSTRUCTIONS (follow exactly) ===\n"
            + _text_blocks(request.get("system"))
            + "\n\n=== USER MESSAGE ===\n"
            + user
            + "\n\n=== MODEL RESPONSE UNDER REVIEW (the auditing model's full answer;"
            + f" stop_reason: {response.get('stop_reason')}) ===\n"
            + json.dumps(answer, ensure_ascii=False, indent=1)
            + "\n\n=== PIPELINE FINAL VERDICT AFTER GUARDRAILS: "
            + f"{row.final_classification} (review_reason: {row.review_reason}) ===\n",
            encoding="utf-8",
        )
    (OUT / "citation_check.json").write_text(json.dumps(citations, indent=1))
    flagged = sum(bool(v) for v in citations.values())
    print(f"{len(citations)} review bundles in {OUT}; citation problems in {flagged}")


def combine() -> None:
    reviewers = sorted(p.name for p in (OUT / "answers").iterdir() if p.is_dir())
    citations = json.loads((OUT / "citation_check.json").read_text())
    outcomes: Counter[str] = Counter()
    combined = []
    print(f"reviewers: {', '.join(reviewers)}")
    for reqno in sorted(citations):
        answers: dict[str, Any] = {}
        for reviewer in reviewers:
            path = OUT / "answers" / reviewer / f"{reqno}.json"
            try:
                answers[reviewer] = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
        outcome = consortium_outcome(answers)
        outcomes[outcome] += 1
        votes = "/".join(
            str(answers.get(r, {}).get("correct_classification", "-"))[:13]
            for r in reviewers
        )
        cite = "CITATION" if citations[reqno] else ""
        print(f"{reqno:<10}{outcome:<26}{votes:<60}{cite}")
        combined.append({"reqno": reqno, "outcome": outcome, "answers": answers})
    (OUT / "combined.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=1)
    )
    print(dict(outcomes))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build_cmd = sub.add_parser(
        "build", help="write review bundles from the audit store"
    )
    build_cmd.add_argument(
        "--reqno-file", type=Path, help="comma- or line-separated REQNOs"
    )
    sub.add_parser("combine", help="combine the reviewers' answers")
    args = parser.parse_args(argv)
    if args.command == "build":
        wanted = None
        if args.reqno_file is not None:
            wanted = {x for x in re.split(r"[,\s]+", args.reqno_file.read_text()) if x}
        build(wanted)
    else:
        combine()


if __name__ == "__main__":
    main()
