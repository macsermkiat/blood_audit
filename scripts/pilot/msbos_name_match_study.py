"""Offline Tier-1 operation-name match study for MSBOS reservation rows.

This script is annotation-only. It reads the existing pilot report and bundle,
writes only ``<work>/msbos_name_match_study.csv``, and never changes a verdict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, cast

from bba.cohort_detector import OperativeEvent  # type: ignore[import-untyped]
from bba.preop_reservation.name_match import (  # type: ignore[import-untyped]
    OperationNameIndex,
    _index_from_rows,
    match_operation_names,
    would_be_verdict,
)
from bba.preop_reservation.reference import (  # type: ignore[import-untyped]
    MSBOS_REFERENCE_FILENAME,
)


class _RunPipelineHelpers(Protocol):
    def _build_op_events(
        self,
        iptsumoprt: list[dict[str, str]],
        ipddchsumoprt: list[dict[str, str]],
        incpt: list[dict[str, str]],
        optract_dict: dict[str, dict[str, str]],
        icd9_dict: dict[str, dict[str, str]],
        an: str,
    ) -> tuple[OperativeEvent, ...]: ...

    def _icd9_dict_from_rows(
        self, rows: list[dict[str, str]]
    ) -> dict[str, dict[str, str]]: ...

    def _normalize_incpt(self, raw: list[dict[str, str]]) -> list[dict[str, str]]: ...

    def _normalize_iptsumoprt(
        self, raw: list[dict[str, str]]
    ) -> list[dict[str, str]]: ...

    def _normalize_optract(
        self, raw: list[dict[str, str]]
    ) -> dict[str, dict[str, str]]: ...


_pipeline = cast(_RunPipelineHelpers, importlib.import_module("run_pipeline"))
_build_op_events = _pipeline._build_op_events
_icd9_dict_from_rows = _pipeline._icd9_dict_from_rows
_normalize_incpt = _pipeline._normalize_incpt
_normalize_iptsumoprt = _pipeline._normalize_iptsumoprt
_normalize_optract = _pipeline._normalize_optract

csv.field_size_limit(sys.maxsize)

WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
BUNDLE = WORK / "bundle"
REPORT = WORK / "report.csv"
OUTPUT = WORK / "msbos_name_match_study.csv"

_STUDY_REASON_ORDER = (
    "unresolved_code",
    "no_planned_op",
    "ambiguous_planned_op",
    "ambiguous_code",
)
STUDY_REASONS = frozenset(_STUDY_REASON_ORDER)
CONTROL_REASONS = frozenset(
    {
        "within_recommendation",
        "type_and_screen_screen_only",
        "over_none",
        "over_gm_excess",
        "over_type_and_screen_crossmatched",
    }
)
_RETURNS_TERMINAL_CLASSIFICATIONS = frozenset(
    {"RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"}
)
_REPORT_REQUIRED = frozenset(
    {
        "reqno",
        "an",
        "order_datetime_utc",
        "component",
        "classification",
        "msbos_reason",
        "msbos_reserved_units",
        "msbos_token",
        "msbos_recommended_units",
        "msbos_resolved_icd9",
        "msbos_reference_hash",
    }
)
OUTPUT_FIELDS = (
    "row_kind",
    "reason",
    "source_icd9",
    "reqno",
    "an",
    "order_datetime_utc",
    "reserved_units",
    "icd10_diagnosis",
    "events_scope",
    "event_names",
    "tier",
    "match_status",
    "representative_operation",
    "matched_operations",
    "matched_event_name",
    "matched_event_datetime",
    "matched_event_hours_from_order",
    "matched_specialty",
    "matched_procedure_group",
    "recommendation_token",
    "recommendation_units",
    "would_be_reason",
    "would_be_is_over",
    "distinct_recommendation_count",
    "code_recommendation",
    "control_score",
    "tier2_confidence",
    "tier2_raw_suggestion",
    "reference_hash",
)
_STATUS_KEYS = ("matched", "no_match", "conflict")
ControlScore = Literal["agree", "disagree", "no_match", "conflict"]


class StudyPreflightError(ValueError):
    """The report or bundle cannot be scored coherently."""


@dataclass(frozen=True)
class OperationMetadata:
    specialty: str
    procedure_group: str


@dataclass(frozen=True)
class StudyReference:
    index: OperationNameIndex
    content_hash: str
    metadata: Mapping[str, OperationMetadata]

    def __post_init__(self) -> None:
        if not self.content_hash.strip():
            raise ValueError("StudyReference content_hash must be non-blank")
        if self.index.content_hash != self.content_hash:
            raise ValueError("StudyReference index/content_hash mismatch")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class StudyRun:
    rows: tuple[Mapping[str, str], ...]
    reference_hash: str
    study_bucket_counts: Mapping[str, int]
    study_status_counts: Mapping[str, Mapping[str, int]]
    control_counts: Mapping[str, int]
    agreement_rate: float | None
    gate_line: str


@dataclass(frozen=True)
class BundleRows:
    iptsumoprt: list[dict[str, str]]
    ipddchsumoprt: list[dict[str, str]]
    incpt: list[dict[str, str]]
    optract: dict[str, dict[str, str]]
    icd9: dict[str, dict[str, str]]
    diagnosis_by_an: Mapping[str, tuple[str, ...]]


def _read_csv(name: str) -> list[dict[str, str]]:
    with (BUNDLE / name).open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_optional_csv(name: str) -> list[dict[str, str]]:
    path = BUNDLE / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_preferred_optional_csv(*names: str) -> list[dict[str, str]]:
    for name in names:
        if (BUNDLE / name).exists():
            return _read_optional_csv(name)
    return []


def _metadata_from_rows(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, OperationMetadata]:
    values: dict[str, tuple[set[str], set[str]]] = {}
    for row in rows:
        operation = (row.get("operation") or "").strip()
        if not operation:
            continue
        specialty = (row.get("specialty") or "").strip()
        group = (row.get("procedure_group") or "").strip()
        specialties, groups = values.setdefault(operation, (set(), set()))
        if specialty:
            specialties.add(specialty)
        if group:
            groups.add(group)
    return {
        operation: OperationMetadata(
            specialty="; ".join(sorted(specialties)),
            procedure_group="; ".join(sorted(groups)),
        )
        for operation, (specialties, groups) in sorted(values.items())
    }


def _load_study_reference(
    reference: StudyReference | None = None,
) -> StudyReference:
    if reference is not None:
        return reference
    path = resources.files("bba.preop_reservation").joinpath(
        "data", MSBOS_REFERENCE_FILENAME
    )
    raw = path.read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    index = _index_from_rows(rows, content_hash=content_hash)
    metadata = MappingProxyType(_metadata_from_rows(rows))
    return StudyReference(index=index, content_hash=content_hash, metadata=metadata)


def _in_scope(row: Mapping[str, str]) -> bool:
    component = (row.get("component") or "").strip()
    classification = (row.get("classification") or "").strip()
    reason = (row.get("msbos_reason") or "").strip()
    return (
        component in {"", "red_cell"}
        and classification in _RETURNS_TERMINAL_CLASSIFICATIONS
        and reason in STUDY_REASONS | CONTROL_REASONS
    )


def _csv_header(path: Path, label: str) -> list[str]:
    if not path.exists():
        raise StudyPreflightError(f"missing {label}: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            fieldnames = csv.DictReader(handle).fieldnames
    except (csv.Error, UnicodeDecodeError) as exc:
        raise StudyPreflightError(f"malformed {label}: {exc}") from exc
    if not fieldnames:
        raise StudyPreflightError(f"{label} has an empty header")
    return list(fieldnames)


def _require_columns(path: Path, label: str, required: frozenset[str]) -> None:
    present = {field.casefold() for field in _csv_header(path, label)}
    missing = sorted(field for field in required if field.casefold() not in present)
    if missing:
        raise StudyPreflightError(
            f"{label} missing required columns: {', '.join(missing)}"
        )


def _parse_bundle_csv(path: Path) -> None:
    """Fully parse a bundle CSV so body corruption fails loud (not silent no_match).

    A header-only check passes an unterminated quoted field that the lenient
    readers would silently drop; parsing the whole file surfaces it here.
    """
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for _ in csv.DictReader(handle):
                pass
    except (csv.Error, UnicodeDecodeError) as exc:
        raise StudyPreflightError(f"malformed bundle file {path}: {exc}") from exc


def _preflight_bundle() -> None:
    if not BUNDLE.is_dir():
        raise StudyPreflightError(f"missing bundle directory: {BUNDLE}")
    _require_columns(
        BUNDLE / "IPTSUMOPRT.csv",
        "IPTSUMOPRT.csv",
        frozenset({"An", "Icd9cm", "Indate"}),
    )
    _require_columns(
        BUNDLE / "ICD9CM.csv",
        "ICD9CM.csv",
        frozenset({"Icd9cm", "Name", "Orflag"}),
    )
    _parse_bundle_csv(BUNDLE / "IPTSUMOPRT.csv")
    _parse_bundle_csv(BUNDLE / "ICD9CM.csv")
    diagnosis = BUNDLE / "Diagnosis.csv"
    if diagnosis.exists():
        _require_columns(diagnosis, "Diagnosis.csv", frozenset({"AN", "ICD10"}))
        _parse_bundle_csv(diagnosis)


def _row_label(row: Mapping[str, str]) -> str:
    reqno = (row.get("reqno") or "").strip()
    an = (row.get("an") or "").strip()
    return f"reqno={reqno!r}, an={an!r}"


def _preflight(
    report_rows: Sequence[Mapping[str, str]],
    report_fieldnames: Sequence[str],
    *,
    expected_hash: str,
) -> None:
    if not report_fieldnames:
        raise StudyPreflightError("report.csv has an empty header")
    missing = sorted(_REPORT_REQUIRED.difference(report_fieldnames))
    if missing:
        raise StudyPreflightError(
            f"report.csv missing required columns: {', '.join(missing)}"
        )
    scoped = [row for row in report_rows if _in_scope(row)]
    blanks = [
        _row_label(row)
        for row in scoped
        if not (row.get("msbos_reference_hash") or "").strip()
    ]
    if blanks:
        raise StudyPreflightError(
            "missing provenance: blank msbos_reference_hash for " + "; ".join(blanks)
        )
    stale = sorted(
        {
            (row.get("msbos_reference_hash") or "").strip()
            for row in scoped
            if (row.get("msbos_reference_hash") or "").strip() != expected_hash
        }
    )
    if stale:
        raise StudyPreflightError(
            f"stale reference hash(es): {', '.join(stale)}; expected {expected_hash}"
        )
    for row in scoped:
        _validate_order_datetime(row)
    _preflight_bundle()


def _validate_order_datetime(row: Mapping[str, str]) -> None:
    raw = (row.get("order_datetime_utc") or "").strip()
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise StudyPreflightError(
            f"invalid order_datetime_utc {raw!r} for {_row_label(row)}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise StudyPreflightError(
            f"order_datetime_utc must be timezone-aware UTC for {_row_label(row)}: {raw!r}"
        )


def _read_report() -> tuple[list[dict[str, str]], list[str]]:
    fieldnames = _csv_header(REPORT, "report.csv")
    with REPORT.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return rows, fieldnames


def _diagnoses_by_an(
    rows: Sequence[Mapping[str, str]],
) -> Mapping[str, tuple[str, ...]]:
    codes: dict[str, set[str]] = {}
    for row in rows:
        folded = {key.casefold(): value for key, value in row.items()}
        an = (folded.get("an") or "").strip()
        code = (folded.get("icd10") or "").strip()
        if an and code:
            codes.setdefault(an, set()).add(code)
    return MappingProxyType(
        {an: tuple(sorted(values)) for an, values in sorted(codes.items())}
    )


def _canonical_icd9_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    canonical: list[dict[str, str]] = []
    for row in rows:
        folded = {key.casefold(): value for key, value in row.items()}
        canonical.append(
            {
                "Icd9cm": folded.get("icd9cm") or "",
                "Name": folded.get("name") or "",
                "Orflag": folded.get("orflag") or "",
            }
        )
    return canonical


def _load_bundle_rows() -> BundleRows:
    iptsumoprt = _normalize_iptsumoprt(_read_csv("IPTSUMOPRT.csv"))
    ipddchsumoprt = _normalize_iptsumoprt(_read_optional_csv("IPDDCHSUMOPRT.csv"))
    incpt = _normalize_incpt(
        _read_preferred_optional_csv("INCPT_OPRTACT.csv", "INCPT.csv")
    )
    optract = _normalize_optract(_read_optional_csv("OPRTACT.csv"))
    icd9 = _icd9_dict_from_rows(_canonical_icd9_rows(_read_csv("ICD9CM.csv")))
    diagnoses = _diagnoses_by_an(_read_optional_csv("Diagnosis.csv"))
    return BundleRows(
        iptsumoprt=iptsumoprt,
        ipddchsumoprt=ipddchsumoprt,
        incpt=incpt,
        optract=optract,
        icd9=icd9,
        diagnosis_by_an=diagnoses,
    )


def _aggregate_metadata(
    operations: Sequence[str], metadata: Mapping[str, OperationMetadata]
) -> tuple[str, str]:
    specialties: set[str] = set()
    groups: set[str] = set()
    for operation in operations:
        item = metadata.get(operation)
        if item is None:
            continue
        specialties.update(value for value in item.specialty.split("; ") if value)
        groups.update(value for value in item.procedure_group.split("; ") if value)
    return "; ".join(sorted(specialties)), "; ".join(sorted(groups))


def _matched_event_fields(
    events: Sequence[OperativeEvent], event_name: str, order: datetime
) -> tuple[str, str]:
    candidates = [event for event in events if event.name == event_name]
    if not candidates:
        return "", ""
    chosen = min(
        candidates,
        key=lambda event: (
            abs((event.operative_datetime - order).total_seconds()),
            event.operative_datetime,
        ),
    )
    hours = (chosen.operative_datetime - order).total_seconds() / 3600
    return chosen.operative_datetime.isoformat(), str(round(hours, 2))


def _control_score(
    status: str,
    recommendation: tuple[str, int] | None,
    code_recommendation: tuple[str, int],
) -> ControlScore:
    if status == "no_match":
        return "no_match"
    if status == "conflicting_recommendations":
        return "conflict"
    if recommendation == code_recommendation:
        return "agree"
    return "disagree"


def _study_case(
    row: Mapping[str, str],
    events: Sequence[OperativeEvent],
    diagnosis_codes: Sequence[str],
    *,
    reference: StudyReference,
) -> dict[str, str]:
    reason = row["msbos_reason"].strip()
    order = datetime.fromisoformat(row["order_datetime_utc"].strip())
    # no_planned_op means no uniquely-selected non-blank upcoming operative code;
    # scan all named events so oddly dated candidates remain visible for review.
    scope = "all_events" if reason == "no_planned_op" else "upcoming"
    scanned = [
        event
        for event in events
        if event.name and (scope == "all_events" or event.operative_datetime >= order)
    ]
    event_names = tuple(sorted({event.name for event in scanned if event.name}))
    result = match_operation_names(reference.index, event_names)
    matched_operations = result.matched_operations
    recommendation = result.recommendation
    matched = result.status == "matched" and recommendation is not None
    representative = matched_operations[0] if matched else ""
    specialty, group = (
        _aggregate_metadata(matched_operations, reference.metadata)
        if matched
        else ("", "")
    )
    event_datetime, event_hours = (
        _matched_event_fields(scanned, result.matched_event_name, order)
        if matched
        else ("", "")
    )
    reserved = int(row["msbos_reserved_units"])
    verdict = (
        would_be_verdict(
            row=recommendation,
            reserved_units=reserved,
            reference_hash=reference.content_hash,
        )
        if recommendation is not None and matched
        else None
    )
    row_kind = "study" if reason in STUDY_REASONS else "control"
    code_tuple = (
        (row["msbos_token"].strip(), int(row["msbos_recommended_units"]))
        if row_kind == "control"
        else ("", 0)
    )
    recommendation_tuple = (
        (recommendation.msbos, recommendation.recommended_units)
        if recommendation is not None and matched
        else None
    )
    score = (
        _control_score(result.status, recommendation_tuple, code_tuple)
        if row_kind == "control"
        else ""
    )
    return {
        "row_kind": row_kind,
        "reason": reason,
        "source_icd9": row["msbos_resolved_icd9"].strip(),
        "reqno": row["reqno"],
        "an": row["an"],
        "order_datetime_utc": row["order_datetime_utc"],
        "reserved_units": str(reserved),
        "icd10_diagnosis": "|".join(sorted(set(diagnosis_codes))),
        "events_scope": scope,
        "event_names": "|".join(event_names),
        "tier": "1",
        "match_status": result.status,
        "representative_operation": representative,
        "matched_operations": "|".join(matched_operations),
        "matched_event_name": result.matched_event_name if matched else "",
        "matched_event_datetime": event_datetime,
        "matched_event_hours_from_order": event_hours,
        "matched_specialty": specialty,
        "matched_procedure_group": group,
        "recommendation_token": recommendation.msbos
        if recommendation is not None
        else "",
        "recommendation_units": (
            str(recommendation.recommended_units) if recommendation is not None else ""
        ),
        "would_be_reason": verdict.reason if verdict is not None else "",
        "would_be_is_over": str(verdict.is_over) if verdict is not None else "",
        "distinct_recommendation_count": str(result.distinct_recommendation_count),
        "code_recommendation": (
            f"{code_tuple[0]} {code_tuple[1]}" if row_kind == "control" else ""
        ),
        "control_score": score,
        "tier2_confidence": "",
        "tier2_raw_suggestion": "",
        "reference_hash": reference.content_hash,
    }


def _status_key(raw_status: str) -> str:
    if raw_status == "conflicting_recommendations":
        return "conflict"
    return raw_status


def control_gate_line(counts: Mapping[str, int]) -> str:
    matched = counts.get("agree", 0) + counts.get("disagree", 0)
    if matched == 0:
        return "GATE: N/A (0 name-matched control rows to score)"
    rate = counts.get("agree", 0) / matched
    verdict = "PASS" if rate >= 0.95 else "FAIL"
    line = f"GATE: {verdict} (rate={rate:.2%}, n={matched})"
    if matched < 10:
        return line + " -- LOW SAMPLE, not a robust gate"
    return line


def run_study(
    *,
    reference: StudyReference | None = None,
    reasons: frozenset[str] | None = None,
    limit: int | None = None,
) -> StudyRun:
    selected_reference = _load_study_reference(reference)
    selected_reasons = STUDY_REASONS if reasons is None else reasons
    report_rows, fieldnames = _read_report()
    _preflight(report_rows, fieldnames, expected_hash=selected_reference.content_hash)
    bundle = _load_bundle_rows()
    output_rows: list[Mapping[str, str]] = []
    study_counts = {reason: 0 for reason in sorted(STUDY_REASONS)}
    status_counts = {
        reason: {status: 0 for status in _STATUS_KEYS}
        for reason in sorted(STUDY_REASONS)
    }
    control_counts = {
        score: 0 for score in ("agree", "conflict", "disagree", "no_match")
    }
    included_study = 0
    for report_row in report_rows:
        if not _in_scope(report_row):
            continue
        reason = report_row["msbos_reason"].strip()
        if reason in STUDY_REASONS:
            if reason not in selected_reasons or (
                limit is not None and included_study >= limit
            ):
                continue
            included_study += 1
        an = report_row["an"]
        events = _build_op_events(
            bundle.iptsumoprt,
            bundle.ipddchsumoprt,
            bundle.incpt,
            bundle.optract,
            bundle.icd9,
            an,
        )
        output = _study_case(
            report_row,
            events,
            bundle.diagnosis_by_an.get(an, ()),
            reference=selected_reference,
        )
        output_rows.append(MappingProxyType(output))
        if reason in STUDY_REASONS:
            study_counts[reason] += 1
            status_counts[reason][_status_key(output["match_status"])] += 1
        else:
            control_counts[output["control_score"]] += 1
    matched_controls = control_counts["agree"] + control_counts["disagree"]
    agreement_rate = (
        control_counts["agree"] / matched_controls if matched_controls else None
    )
    frozen_status = MappingProxyType(
        {reason: MappingProxyType(counts) for reason, counts in status_counts.items()}
    )
    frozen_control = MappingProxyType(control_counts)
    return StudyRun(
        rows=tuple(output_rows),
        reference_hash=selected_reference.content_hash,
        study_bucket_counts=MappingProxyType(study_counts),
        study_status_counts=frozen_status,
        control_counts=frozen_control,
        agreement_rate=agreement_rate,
        gate_line=control_gate_line(frozen_control),
    )


def _validate_output_path(path: Path) -> Path:
    resolved_work = WORK.resolve()
    resolved = path.resolve()
    forbidden = {REPORT.resolve(), (WORK / "review.html").resolve()}
    if not resolved.is_relative_to(resolved_work):
        raise StudyPreflightError(f"output path escapes work directory: {resolved}")
    if resolved in forbidden or resolved.is_relative_to(BUNDLE.resolve()):
        raise StudyPreflightError(f"refusing protected output path: {resolved}")
    return resolved


def write_study_csv(rows: Sequence[Mapping[str, str]]) -> Path:
    _validate_output_path(OUTPUT)
    output = OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def format_summary(result: StudyRun, *, output_path: Path) -> str:
    lines = [
        f"Preflight OK: reference_hash={result.reference_hash}",
        f"Study: total rows={sum(result.study_bucket_counts.values())}",
    ]
    for reason in _STUDY_REASON_ORDER:
        counts = result.study_status_counts[reason]
        lines.append(
            f"Study bucket {reason}: total={result.study_bucket_counts[reason]} "
            f"matched={counts['matched']} no_match={counts['no_match']} "
            f"conflict={counts['conflict']}"
        )
    unresolved = result.study_status_counts["unresolved_code"]
    actual = (unresolved["matched"], unresolved["conflict"], unresolved["no_match"])
    lines.append(
        "unresolved_code probe: "
        f"actual matched={actual[0]} conflict={actual[1]} no_match={actual[2]}; "
        "baseline ~22 matched / 1 conflict / ~150 no_match"
    )
    if actual != (22, 1, 150):
        lines.append(
            "unresolved_code probe note: actuals deviate from the stated baseline"
        )
    control_total = sum(result.control_counts.values())
    lines.append(f"Control: total={control_total}")
    lines.append(
        "Control scores: "
        f"agree={result.control_counts['agree']} "
        f"disagree={result.control_counts['disagree']} "
        f"no_match={result.control_counts['no_match']} "
        f"conflict={result.control_counts['conflict']}"
    )
    denominator = result.control_counts["agree"] + result.control_counts["disagree"]
    rate = "N/A" if result.agreement_rate is None else f"{result.agreement_rate:.2%}"
    lines.extend(
        [
            f"agreement_rate: {result.control_counts['agree']}/{denominator} ({rate})",
            result.gate_line,
            "Tier-1 only (Tier-2 is #189, not run)",
            f"Output CSV: {output_path}",
        ]
    )
    return "\n".join(lines)


def _parse_reasons(raw: str) -> frozenset[str]:
    values = frozenset(value.strip() for value in raw.split(",") if value.strip())
    invalid = sorted(values.difference(STUDY_REASONS))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"invalid study reason(s): {', '.join(invalid)}"
        )
    return values


def _nonnegative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("limit must be non-negative")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reasons",
        type=_parse_reasons,
        default=STUDY_REASONS,
        help="comma-separated subset of study reasons",
    )
    parser.add_argument("--limit", type=_nonnegative_int, default=None)
    return parser


def main(
    argv: Sequence[str] | None = None, *, reference: StudyReference | None = None
) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    try:
        result = run_study(reference=reference, reasons=args.reasons, limit=args.limit)
        output = write_study_csv(result.rows)
    except StudyPreflightError as exc:
        print(f"Study preflight failed: {exc}", file=sys.stderr)
        return 1
    print(format_summary(result, output_path=output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
