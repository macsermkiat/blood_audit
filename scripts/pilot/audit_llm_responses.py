"""Audit the LLM leg's responses BEFORE build_review.py renders them.

Run order: ``run_pipeline.py`` -> ``run_llm_leg.py`` -> THIS -> ``build_review.py``.

Issue #239: on the first real-data platelet run 22 of 297 answers carried the
label APPROPRIATE while their own reasoning concluded INAPPROPRIATE or
NEEDS_REVIEW, and a clinician found it on case 1 of the review page. This step
reads every current record (``llm_report.json`` + the audit store's raw tool
payload) and reports what is wrong with the RESPONSES. The loop is: run the
audit, fix the cause, re-run the flagged REQNOs (``BBA_PILOT_ONLY_REQNO``), run
the audit again; build the page when it is clean.

Checks code can decide run always. Reading the conclusion out of free-text
reasoning is an extraction task for a CONSORTIUM of models (``--judge``): each
judge sees one summary only, never the label; code compares the votes with the
label and the English conclusion with the Thai one. The consortium certifies a
label only when unanimous; one dissenting judge is a split for a human, never
outvoted. The regex reading of the conclusion is a CANDIDATE finder: on the
real run the judges overruled 4 of its 10 hits and found 7 it missed, so use
``--judge all`` before a page goes to clinicians.

Patient-level text is sent only to the Anthropic API, the processor the LLM leg
already uses. Nothing here changes a verdict: it writes a findings file.

Environment variables:

* ``BBA_PILOT_WORK_DIR`` — sandbox directory (default ``/tmp/bba_mini``).
* ``ANTHROPIC_API_KEY`` — required only with ``--judge``.
* ``BBA_AUDIT_JUDGE_MODELS`` — comma-separated judge model ids.

Outputs in the work dir: ``llm_response_audit.json``, ``llm_response_audit.csv``
and ``llm_response_audit_reqnos.txt`` (REQNOs whose finding a re-run can fix,
ready for ``BBA_PILOT_ONLY_REQNO``). Exit status 2 when any HIGH finding exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from bba.audit_store import AuditStore
from bba.audit_store.models import AuditStoreConfig

WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
DEFAULT_JUDGE_MODELS = "claude-sonnet-5,claude-haiku-4-5-20251001,claude-opus-5"
JUDGE_WORKERS = 8
JUDGE_MAX_TOKENS = 200
MIN_STATED_VOTES = 2
LOW_CONFIDENCE = 0.5
# Thai inside the English summary is normally a verbatim quote from a Thai
# note; only a field that is substantially the wrong language is a mix-up.
THAI_SHARE_LIMIT = 0.2

Severity = Literal["HIGH", "MEDIUM", "LOW"]
CLASSES = (
    "POTENTIALLY_INAPPROPRIATE",
    "INSUFFICIENT_EVIDENCE",
    "INAPPROPRIATE",
    "NEEDS_REVIEW",
    "APPROPRIATE",
)
HARD_SIGNALS = (
    "active_bleeding",
    "procedure_indication",
    "prophylactic_marrow_failure",
    "intracranial_bleed_indication",
)
PARSE_FAILURE_REASONS = frozenset(
    {
        "malformed_json",
        "schema_mismatch",
        "classification_out_of_set",
        "empty_response",
        "tool_use_missing",
        "hallucination_suspect",
        "empty_reasoning",
    }
)
# Findings a fresh LLM call can fix; the rest need a code or data fix first.
RERUN_CODES = frozenset(
    {"parse_failure", "consortium_label_contradiction", "llm_result_missing"}
)

# Upper-case only: lower-case "appropriate" is ordinary prose ("appropriate
# threshold"); the consortium reads conclusions written in plain words.
_CLASS_RE = re.compile(
    r"\b(" + "|".join(c.replace("_", "[ _]") for c in CLASSES) + r")\b"
)
# A class is REJECTED when a cue sits right before it ("not X", "rather than a
# clean X") or when it continues a rejected list ("X or Y", "X, nor Y"). Only
# the mention is rejected, never the rest of the sentence: in "not APPROPRIATE
# but INAPPROPRIATE" the correction is the conclusion.
_FILLER = r"(?:[a-z][a-z-]*\s+){0,3}"
_CUE_RE = re.compile(
    r"(?:\brather than|\binstead of|\bas opposed to|\bnot|\bnor|\bnever)\s+"
    + _FILLER
    + r"$",
    re.IGNORECASE,
)
_CHAIN_RE = re.compile(
    r"\s*(?:,|/|\bor\b|\bnor\b|,\s*or\b)\s*" + _FILLER, re.IGNORECASE
)
_THAI_RE = re.compile("[฀-๿]")
_TAG_RE = re.compile(r"</?[a-z_]+\s*/?>", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ResponseRecord:
    reqno: str
    component: str
    label: str | None
    final: str
    review_reason: str | None
    field_order: tuple[str, ...]
    signals: dict[str, Any]
    reasoning_en: str
    reasoning_th: str
    indications: tuple[Any, ...]
    trigger_value: float | None


@dataclass(frozen=True, slots=True)
class Finding:
    reqno: str
    code: str
    severity: Severity
    detail: str
    judge_votes: tuple[str, ...] = ()


def concluded_class(reasoning: str) -> str | None:
    """The last class the reasoning names without rejecting it, else ``None``."""
    concluded: str | None = None
    previous_end = 0
    previous_rejected = False
    for match in _CLASS_RE.finditer(reasoning):
        between = reasoning[previous_end : match.start()]
        rejected = bool(_CUE_RE.search(between)) or (
            previous_rejected and _CHAIN_RE.fullmatch(between) is not None
        )
        if not rejected:
            concluded = match.group(1).replace(" ", "_")
        previous_end, previous_rejected = match.end(), rejected
    return concluded


def _thai_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if _THAI_RE.match(ch)) / len(letters)


def _trigger_value_stated(record: ResponseRecord) -> bool:
    value = record.trigger_value
    if value is None or value <= 0:
        return True
    text = record.reasoning_en.replace(",", "")
    if record.component == "platelet":
        per_ul = rf"(?<![\d.]){value * 1000:.0f}(?!\d)"
        per_k = rf"(?<![\d.]){value:g}(?:\.0)?\s*(?:k\b|x\s*10|×)"
        return re.search(f"{per_ul}|{per_k}", text, re.IGNORECASE) is not None
    return re.search(rf"(?<![\d.]){value:g}(?:0)?(?!\d)", text) is not None


def check_record(record: ResponseRecord) -> tuple[Finding, ...]:
    """Deterministic checks on one response. Pure; no I/O."""
    out: list[Finding] = []

    def add(code: str, severity: Severity, detail: str) -> None:
        out.append(Finding(record.reqno, code, severity, detail))

    if record.review_reason in PARSE_FAILURE_REASONS:
        add("parse_failure", "HIGH", f"{record.review_reason}: re-run this REQNO")
        return tuple(out)
    if record.label is None:
        add("label_missing", "HIGH", "tool payload carries no classification")
        return tuple(out)

    concluded = concluded_class(record.reasoning_en)
    if concluded and concluded != record.label:
        add(
            "label_contradicts_reasoning",
            "HIGH",
            f"label {record.label}, reasoning appears to conclude {concluded}",
        )
    if record.final != record.label and record.review_reason is None:
        add(
            "final_differs_without_reason",
            "HIGH",
            f"label {record.label} persisted as {record.final} with no review_reason",
        )
    order = record.field_order
    if (
        record.component == "platelet"
        and "classification" in order
        and "reasoning_summary_en" in order
        and order.index("classification") < order.index("reasoning_summary_en")
    ):
        add(
            "label_before_reasoning",
            "MEDIUM",
            "model wrote the label before its reasoning (schema asks for it last)",
        )
    true_signals = sorted(k for k, v in record.signals.items() if v is True)
    if record.label in ("INAPPROPRIATE", "INSUFFICIENT_EVIDENCE") and true_signals:
        add(
            "signal_contradicts_label",
            "HIGH",
            f"label {record.label} with true hard signal(s): {', '.join(true_signals)}",
        )
    cited = [i for i in record.indications if isinstance(i, Mapping)]
    if len(cited) != len(record.indications) or any(
        not str(i.get("quote") or "").strip()
        or not str(i.get("source_id") or "").strip()
        for i in cited
    ):
        add("malformed_indication", "MEDIUM", "indication without a quote or source_id")
    if record.label == "APPROPRIATE":
        if not cited:
            add("appropriate_without_citation", "MEDIUM", "APPROPRIATE cites nothing")
        elif all(
            isinstance(i.get("confidence"), (int, float))
            and i["confidence"] < LOW_CONFIDENCE
            for i in cited
        ):
            add(
                "low_confidence_appropriate",
                "MEDIUM",
                f"every cited indication has confidence below {LOW_CONFIDENCE}",
            )
    if not _trigger_value_stated(record):
        add(
            "trigger_value_not_in_reasoning",
            "LOW",
            f"trigger value {record.trigger_value:g} is never stated in the reasoning",
        )
    if (
        _thai_share(record.reasoning_en) > THAI_SHARE_LIMIT
        or _thai_share(record.reasoning_th) < THAI_SHARE_LIMIT
        or _TAG_RE.search(record.reasoning_en)
        or _TAG_RE.search(record.reasoning_th)
    ):
        add(
            "language_or_tag_leak",
            "MEDIUM",
            "a summary is in the wrong language or empty, or carries markup",
        )
    return tuple(out)


def missing_result_findings(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[Finding, ...]:
    """Report entries the LLM leg wrote with ``llm_final: null`` (batch row
    dropped or unparsable): the page shows "LLM verdict missing" for them."""
    return tuple(
        Finding(
            str(e.get("reqno", e.get("audit_id"))),
            "llm_result_missing",
            "HIGH",
            "report entry has no LLM result: re-run this REQNO",
        )
        for e in entries
        if "llm_final" in e and not e.get("llm_final")
    )


def unmatched_report_finding(reqno: str, why: str) -> Finding:
    return Finding(reqno, "response_not_auditable", "HIGH", why)


def _majority(votes: Sequence[str]) -> str | None:
    stated = [v for v in votes if v not in ("NONE_STATED", "JUDGE_ERROR")]
    if len(stated) < MIN_STATED_VOTES:
        return None
    top, n = Counter(stated).most_common(1)[0]
    return top if n * 2 > len(stated) else None


def consortium_verdict(label: str | None, votes: Sequence[str]) -> str:
    """Compare the judges' extracted conclusions with the label.

    ``contradiction``: a strict majority of stated votes names another class.
    ``consistent``: every stated vote names the label (dissent is never
    outvoted). ``unavailable``: too many judges failed to certify anything."""
    usable = [v for v in votes if v != "JUDGE_ERROR"]
    if len(usable) < MIN_STATED_VOTES:
        return "unavailable"
    stated = [v for v in usable if v != "NONE_STATED"]
    if len(stated) < MIN_STATED_VOTES:
        return "no_conclusion_stated"
    majority = _majority(stated)
    if majority is not None and majority != label:
        return "contradiction"
    return "consistent" if set(stated) == {label} else "split"


_JUDGE_TOOL = {
    "name": "report_conclusion",
    "description": "Report which verdict class the reasoning text concludes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "concluded_class": {"type": "string", "enum": [*CLASSES, "NONE_STATED"]}
        },
        "required": ["concluded_class"],
    },
}
_JUDGE_SYSTEM = (
    "You are given the reasoning summary (English or Thai) an auditor wrote for "
    "one blood-component order. Report the verdict class the reasoning FINALLY "
    "concludes. Classes named only to be rejected ('X rather than Y') are not "
    "the conclusion. If the text changes its mind, report where it ends up. "
    "'Flag for committee review' means NEEDS_REVIEW. If it never states a "
    "conclusion, report NONE_STATED. Do not judge whether the conclusion is right."
)


def _judge_one(client: Any, model: str, text: str) -> str:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=JUDGE_MAX_TOKENS,
            system=_JUDGE_SYSTEM,
            tools=[_JUDGE_TOOL],
            tool_choice={"type": "tool", "name": _JUDGE_TOOL["name"]},
            messages=[{"role": "user", "content": text}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                vote = (block.input or {}).get("concluded_class")
                return vote if vote in (*CLASSES, "NONE_STATED") else "JUDGE_ERROR"
    except Exception as exc:  # noqa: BLE001 - one judge failing must not sink the audit
        print(f"  WARN judge {model} failed: {exc}", file=sys.stderr)
    return "JUDGE_ERROR"


def judge_records(
    records: Sequence[ResponseRecord], models: Sequence[str]
) -> tuple[Finding, ...]:
    import anthropic  # lazy: the deterministic audit needs no SDK or key

    client = anthropic.Anthropic()
    # Keyed by position, not REQNO: two patients can share a REQNO.
    jobs = [
        (index, language, model)
        for index in range(len(records))
        for language in ("en", "th")
        for model in models
    ]

    def run(job: tuple[int, str, str]) -> str:
        index, language, model = job
        record = records[index]
        text = record.reasoning_en if language == "en" else record.reasoning_th
        return _judge_one(client, model, text) if text.strip() else "NONE_STATED"

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        votes = list(pool.map(run, jobs))
    cast: dict[tuple[int, str], list[str]] = {}
    for (index, language, _model), vote in zip(jobs, votes, strict=True):
        cast.setdefault((index, language), []).append(vote)

    out: list[Finding] = []
    for index, record in enumerate(records):
        en, th = tuple(cast[(index, "en")]), tuple(cast[(index, "th")])
        verdict = consortium_verdict(record.label, en)
        told = f"label {record.label}; judges read {', '.join(en)}"
        if verdict == "contradiction":
            out.append(
                Finding(
                    record.reqno, "consortium_label_contradiction", "HIGH", told, en
                )
            )
        elif verdict == "split":
            out.append(Finding(record.reqno, "consortium_split", "MEDIUM", told, en))
        elif verdict == "unavailable":
            out.append(
                Finding(record.reqno, "consortium_unavailable", "HIGH", told, en)
            )
        en_majority, th_majority = _majority(en), _majority(th)
        if en_majority and th_majority and en_majority != th_majority:
            out.append(
                Finding(
                    record.reqno,
                    "en_th_conclusion_mismatch",
                    "HIGH",
                    f"English concludes {en_majority}, Thai concludes {th_majority}",
                    (*en, *th),
                )
            )
    return tuple(out)


def _tool_input(response_json: Any) -> dict[str, Any]:
    for block in (response_json or {}).get("content", ()) or ():
        if isinstance(block, Mapping) and block.get("type") == "tool_use":
            return dict(block.get("input") or {})
    return {}


def load_records(work: Path) -> tuple[tuple[ResponseRecord, ...], tuple[Finding, ...]]:
    """Current LLM records joined to their raw tool payload, plus a finding for
    every report entry that could not be audited (never a silent skip)."""
    entries = json.loads((work / "llm_report.json").read_text())
    report = {r["audit_id"]: r for r in entries if r.get("llm_final")}
    store = AuditStore(
        AuditStoreConfig(root_dir=work / "data" / "audit_store", code_version="audit")
    )
    calls: dict[tuple[str, str], Any] = {}
    for call in store.read_llm_calls():
        key = (call.audit_id, call.run_id)
        if key not in calls or call.request_timestamp > calls[key].request_timestamp:
            calls[key] = call
    # The store row that produced the report entry: same final verdict and
    # reason, newest first. A newer row that disagrees with the report would
    # pair one run's label with another run's reasoning.
    rows: dict[str, Any] = {}
    for row in sorted(store.read_audit_results(), key=lambda r: r.run_timestamp):
        final = report.get(row.audit_id, {}).get("llm_final")
        if (
            final is not None
            and row.final_classification == final.get("final_classification")
            and row.review_reason == final.get("review_reason")
        ):
            rows[row.audit_id] = row

    records: list[ResponseRecord] = []
    skipped: list[Finding] = [*missing_result_findings(entries)]
    for audit_id, entry in report.items():
        final = entry["llm_final"]
        reqno = str(entry.get("reqno", audit_id))
        row = rows.get(audit_id)
        if row is None:
            skipped.append(
                unmatched_report_finding(reqno, "no audit-store row matches the report")
            )
            continue
        call = calls.get((audit_id, row.run_id))
        tool = _tool_input(call.response_json) if call is not None else {}
        from_model = str(final.get("model") or "").startswith("claude")
        if not tool and final.get("review_reason") not in PARSE_FAILURE_REASONS:
            if from_model:
                skipped.append(
                    unmatched_report_finding(
                        reqno, "no stored tool payload for the row"
                    )
                )
            continue  # deterministic marker rows carry no model response
        component = getattr(row, "component", "red_cell") or "red_cell"
        records.append(
            ResponseRecord(
                reqno=reqno,
                component=component,
                label=tool.get("classification"),
                final=str(final.get("final_classification")),
                review_reason=final.get("review_reason"),
                field_order=tuple(tool),
                signals={k: tool[k] for k in HARD_SIGNALS if k in tool},
                # As persisted (after replay's tag salvage): what the page shows.
                reasoning_en=str(final.get("reasoning_en") or ""),
                reasoning_th=str(final.get("reasoning_th") or ""),
                indications=tuple(tool.get("indications") or ()),
                trigger_value=(
                    row.platelet_value if component == "platelet" else row.hb_value
                ),
            )
        )
    return tuple(records), tuple(skipped)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--judge",
        choices=("off", "candidates", "all"),
        default="off",
        help="consortium judge: off; only records the regex could not settle "
        "(contradiction candidates + no stated conclusion); or all (recommended)",
    )
    args = parser.parse_args(argv)

    records, skipped = load_records(WORK)
    findings = [*skipped, *(f for r in records for f in check_record(r))]
    if args.judge != "off":
        models = [
            m.strip()
            for m in os.environ.get(
                "BBA_AUDIT_JUDGE_MODELS", DEFAULT_JUDGE_MODELS
            ).split(",")
            if m.strip()
        ]
        if len(models) < MIN_STATED_VOTES:
            parser.error(f"--judge needs at least {MIN_STATED_VOTES} judge models")
        candidates = {
            f.reqno for f in findings if f.code == "label_contradicts_reasoning"
        }
        to_judge = [
            r
            for r in records
            if r.label is not None
            and r.review_reason not in PARSE_FAILURE_REASONS
            and (
                args.judge == "all"
                or r.reqno in candidates
                or concluded_class(r.reasoning_en) is None
            )
        ]
        print(
            f"consortium: {len(to_judge)} records x 2 summaries x {len(models)} judges"
        )
        judged = {r.reqno for r in to_judge}
        # The judges settle what the regex only suspected.
        findings = [
            replace(f, severity="LOW", detail=f.detail + " (regex; see consortium)")
            if f.code == "label_contradicts_reasoning" and f.reqno in judged
            else f
            for f in findings
        ]
        findings.extend(judge_records(to_judge, models))

    high = sorted({f.reqno for f in findings if f.severity == "HIGH"})
    rerun = sorted({f.reqno for f in findings if f.code in RERUN_CODES})
    (WORK / "llm_response_audit.json").write_text(
        json.dumps([asdict(f) for f in findings], ensure_ascii=False, indent=1)
    )
    with (WORK / "llm_response_audit.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["reqno", "severity", "code", "detail"])
        writer.writerows([f.reqno, f.severity, f.code, f.detail] for f in findings)
    (WORK / "llm_response_audit_reqnos.txt").write_text(",".join(rerun))

    print(
        f"audited {len(records)} LLM responses in {WORK} "
        f"({len(skipped)} report entries could not be audited)"
    )
    for (severity, code), n in sorted(
        Counter((f.severity, f.code) for f in findings).items()
    ):
        print(f"  {severity:<6} {code:<34} {n}")
    print(f"REQNOs with a HIGH finding: {len(high)}")
    print(f"REQNOs a re-run can fix: {len(rerun)} -> llm_response_audit_reqnos.txt")
    return 2 if high else 0


if __name__ == "__main__":
    raise SystemExit(main())
