"""RedactorBackend over the owner's thai-medical-deid scripts.

The approved entry point is ``deidentify_nun.py``. It returns redacted text and
detections whose offsets index its own regex-preprocessed text, so this adapter
derives its own spans by aligning that output with the original note. Spans
carry PHI. They go only to ``redact_bundle`` for role-cue context and never
into a RedactionResult, a log line, or an error message.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from bba.deid_redactor.exceptions import BackendRedactionError
from bba.deid_redactor.models import (
    BackendRedactionResult,
    RedactionSpan,
    RedactorVersion,
)

ENTRY_RELPATH = Path("deidentify_nun.py")
MODEL_FILENAME = "model.safetensors"
_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")


class ThaiDeidSetupError(RuntimeError):
    """The redactor checkout or model directory cannot be trusted."""


class ThaiMedicalDeidBackend:
    def __init__(self, module: Any, ner: Any) -> None:
        self._module = module
        self._ner = ner

    def redact(self, text: str) -> BackendRedactionResult:
        try:
            final = str(self._module.anonymize_batch(self._ner, [text])[0][0])
        except Exception as exc:
            # The exception text can quote the note; keep only its type.
            raise BackendRedactionError(
                f"thai-medical-deid failed ({type(exc).__name__})"
            ) from None
        # Thai digits map 1:1, so offsets in the normalized text are offsets in
        # the original.
        normalized = text.translate(self._module._THAI_DIGIT_TABLE)
        return BackendRedactionResult(text=final, spans=_align_spans(normalized, final))


def _align_spans(original: str, final: str) -> tuple[RedactionSpan, ...]:
    opcodes = difflib.SequenceMatcher(
        None, original, final, autojunk=False
    ).get_opcodes()
    spans: list[RedactionSpan] = []
    for match in _PLACEHOLDER.finditer(final):
        touched = [
            (tag, i1, i2, j1, j2)
            for tag, i1, i2, j1, j2 in opcodes
            if (j1 < match.end() and match.start() < j2)
            or (j1 == j2 and match.start() < j1 < match.end())
        ]
        # A token already in the note (the export was masked once before) is
        # still a span: redact_bundle needs one span per [PERSON] and [DATE]
        # token. A token that shares letters with the PHI it replaced
        # (PERSON/SOMCHAI) can straddle blocks; its span covers every touched
        # original range.
        starts: list[int] = []
        ends: list[int] = []
        for tag, i1, i2, j1, j2 in touched:
            if tag == "equal":
                lo = max(match.start(), j1)
                hi = min(match.end(), j2)
                starts.append(i1 + lo - j1)
                ends.append(i1 + hi - j1)
            else:
                starts.append(i1)
                ends.append(i2)
        start, end = min(starts), max(ends)
        spans.append(
            RedactionSpan(
                start=start,
                end=end,
                entity_type=match.group()[1:-1],
                original_text=original[start:end],
            )
        )
    return tuple(spans)


def _git(deid_repo: Path, *args: str) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(deid_repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ThaiDeidSetupError(f"git {args[0]} failed in {deid_repo}") from exc
    return done.stdout.strip()


def _pipeline_version(entry: Path) -> str:
    """Read PIPELINE_VERSION from source so the policy needs no torch import."""
    for node in ast.parse(entry.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PIPELINE_VERSION" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value:
                return value
    raise ThaiDeidSetupError(f"PIPELINE_VERSION not found in {entry}")


def redactor_version(deid_repo: Path, model_dir: Path) -> RedactorVersion:
    if _git(deid_repo, "status", "--porcelain"):
        raise ThaiDeidSetupError(
            "thai-medical-deid checkout has uncommitted changes; the version "
            "would not identify the code that redacts"
        )
    head = _git(deid_repo, "rev-parse", "HEAD")
    model = model_dir / MODEL_FILENAME
    try:
        with model.open("rb") as stream:
            model_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise ThaiDeidSetupError(f"cannot read {model}") from exc
    try:
        import pythainlp
    except ImportError as exc:
        raise ThaiDeidSetupError("pythainlp is required to pin the gazetteer") from exc
    return RedactorVersion(
        version=f"thai-medical-deid@{head}+{_pipeline_version(deid_repo / ENTRY_RELPATH)}",
        model_sha=model_sha,
        gazetteer_version=f"pythainlp-{pythainlp.__version__}",
    )


def _import_entry(entry: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("deidentify_nun", entry)
    if spec is None or spec.loader is None:
        raise ThaiDeidSetupError(f"cannot import {entry}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_backend(
    deid_repo: Path, model_dir: Path
) -> tuple[ThaiMedicalDeidBackend, RedactorVersion]:
    version = redactor_version(deid_repo, model_dir)
    # Both must precede the import: the redactor imports transformers at load,
    # and a bytecode cache would dirty the checkout the version vouches for.
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.dont_write_bytecode = True
    module = _import_entry(deid_repo / ENTRY_RELPATH)
    if not module._THAI_FIRST_NAMES or not module._THAI_FAMILY_NAMES:
        raise ThaiDeidSetupError(
            "Thai name gazetteer is empty (PyThaiNLP corpora missing); refusing "
            "to redact without it"
        )
    if not version.version.endswith(f"+{module.PIPELINE_VERSION}"):
        raise ThaiDeidSetupError("PIPELINE_VERSION differs from the checked source")
    module.MODEL_ID = str(model_dir)
    return ThaiMedicalDeidBackend(module, module.load_ner_pipeline()), version
