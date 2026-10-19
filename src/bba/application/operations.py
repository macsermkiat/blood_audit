"""Evaluation, monitoring and report adapters over explicit persisted inputs."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from bba.audit_store.models import SafeId
from bba.audit_store import AuditRow, AuditStore, Classification
from bba.eval_harness import agreement_with_metrics
from bba.monitoring import (
    GoldenSetEntry,
    SentinelManifest,
    evaluate_golden_set_drift,
    evaluate_sentinel_run,
)
from bba.report_generator import ReportInputs, generate_monthly_report
from bba.report_generator.builder import default_indication_codes_extractor
from bba.secure_files import read_private_bytes, write_private_bytes


def run_rows(store: AuditStore, run_id: str, version: str) -> tuple[AuditRow, ...]:
    rows = store.read_audit_results(run_id=run_id, code_version=version)
    if not rows:
        raise ValueError(
            "no committed audit rows for the selected run and code version"
        )
    if len({r.audit_id for r in rows}) != len(rows):
        raise ValueError("ambiguous duplicate audit IDs in selected version")
    return rows


def evaluate(rows: tuple[AuditRow, ...], labels_path: Path) -> dict[str, object]:
    label_bytes = labels_path.read_bytes()
    with io.StringIO(label_bytes.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["audit_id", "classification"]:
            raise ValueError(
                "reviewed labels CSV requires audit_id,classification columns"
            )
        labels: tuple[tuple[str, Classification], ...] = tuple(
            (
                r["audit_id"],
                TypeAdapter(Classification).validate_python(r["classification"]),
            )
            for r in reader
        )
    if not labels or len({aid for aid, _ in labels}) != len(labels):
        raise ValueError(
            "reviewed labels must be nonempty and contain unique audit IDs"
        )
    actual = {r.audit_id: r.final_classification for r in rows}
    if any(aid not in actual for aid, _ in labels):
        raise ValueError(
            "reviewed label has no audit row in the selected run and version"
        )
    metrics = agreement_with_metrics(
        tuple(label for _, label in labels), tuple(actual[aid] for aid, _ in labels)
    )
    return {
        "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "audit_rows_sha256": hashlib.sha256(
            json.dumps(
                [
                    r.model_dump(mode="json")
                    for r in sorted(rows, key=lambda r: r.audit_id)
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "reviewed_count": len(labels),
        "run_row_count": len(rows),
        "unreviewed_count": len(rows) - len(labels),
        "agreement": metrics.model_dump(mode="json"),
    }


def persist_analysis(root: Path, result: dict[str, object]) -> Path:
    payload = json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2).encode()
    target = root / f"{hashlib.sha256(payload).hexdigest()}.json"
    try:
        write_private_bytes(target, payload)
    except FileExistsError:
        if read_private_bytes(target) != payload:
            raise ValueError(
                "analysis artifact conflicts with its content hash"
            ) from None
    return target.resolve()


class MonitoringManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    baseline_run: SafeId
    current_run: SafeId
    baseline_code_version: str = Field(min_length=1)
    current_code_version: str = Field(min_length=1)
    sentinel: SentinelManifest


def sentinel(
    store: AuditStore, manifest: MonitoringManifest, *, quarterly: bool
) -> BaseModel:
    if (manifest.baseline_run, manifest.baseline_code_version) == (
        manifest.current_run,
        manifest.current_code_version,
    ):
        raise ValueError(
            "baseline and current must identify distinct committed runs or code versions"
        )
    baseline = {
        r.audit_id: r
        for r in run_rows(store, manifest.baseline_run, manifest.baseline_code_version)
    }
    current = {
        r.audit_id: r
        for r in run_rows(store, manifest.current_run, manifest.current_code_version)
    }
    ids = manifest.sentinel.audit_ids
    if not ids or len(ids) != manifest.sentinel.size or len(set(ids)) != len(ids):
        raise ValueError("sentinel manifest must contain exactly size unique audit IDs")
    if any(aid not in baseline or aid not in current for aid in ids):
        raise ValueError("every manifest audit ID must exist in both selected runs")
    if quarterly:

        def entry(row: AuditRow) -> GoldenSetEntry:
            return GoldenSetEntry(
                audit_id=row.audit_id,
                classification=row.final_classification,
                indications=default_indication_codes_extractor(row),
            )

        return evaluate_golden_set_drift(
            baseline=tuple(entry(baseline[aid]) for aid in ids),
            current=tuple(entry(current[aid]) for aid in ids),
        )
    return evaluate_sentinel_run(
        manifest=manifest.sentinel,
        previous={aid: baseline[aid].final_classification for aid in ids},
        current={aid: current[aid].final_classification for aid in ids},
    )


def write_report(inputs: ReportInputs, report_format: str) -> Path:
    if report_format == "pdf":
        return generate_monthly_report(inputs).pdf_path
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    target = inputs.output_dir / f"report.{report_format}"
    if report_format == "json":
        target.write_text(inputs.model_dump_json(indent=2), encoding="utf-8")
    elif report_format == "html":
        columns = (
            "audit_id",
            "order_datetime",
            "ward_id",
            "physician_id",
            "final_classification",
            "cohort_applied",
            "needs_human_review",
        )
        header = "".join(f'<th scope="col">{html.escape(c)}</th>' for c in columns)
        body = "".join(
            "<tr>"
            + "".join(f"<td>{html.escape(str(getattr(row, c)))}</td>" for c in columns)
            + "</tr>"
            for row in inputs.rows
        )
        metadata = html.escape(
            json.dumps(
                {
                    "footer": inputs.footer.model_dump(mode="json"),
                    "provenance": [
                        p.model_dump(mode="json") for p in inputs.provenance
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        target.write_text(
            f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Blood audit {inputs.month}</title><h1>Blood audit {inputs.month}</h1><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table><h2>Provenance</h2><pre>{metadata}</pre></html>',
            encoding="utf-8",
        )
    else:
        raise ValueError("unsupported report format")
    return target.resolve()
