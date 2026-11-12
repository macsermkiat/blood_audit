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
import importlib
import importlib.metadata
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
_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")
# The tokenizer truncates at 510 tokens; without overlapping windows the NER
# never sees names past that point in a long note.
NER_STRIDE = 128
# Measured on 12 real long notes: NER over 400-char pieces found 10 names, the
# strided pass over the whole chunk 2. Windows overlap by 150 chars, so a name
# shorter than that lies whole in some window instead of being cut in two.
NER_WINDOW_CHARS = 400
NER_WINDOW_STEP = 250


def _window_starts(length: int) -> list[int]:
    starts = [0]
    while starts[-1] + NER_WINDOW_CHARS < length:
        starts.append(starts[-1] + NER_WINDOW_STEP)
    return starts


class ThaiDeidSetupError(RuntimeError):
    """The redactor checkout or model directory cannot be trusted."""


def _with_stride(ner: Any) -> Any:
    def strided(texts: Any, **kwargs: Any) -> Any:
        return ner(texts, **{**kwargs, "stride": NER_STRIDE})

    return strided


def windowed_person_ner(ner: Any) -> Any:
    """PERSON entities from overlapping windows, offsets mapped to the full text."""

    def persons(texts: Any, **kwargs: Any) -> Any:
        results: list[list[dict[str, Any]]] = []
        for text in texts:
            starts = _window_starts(len(text))
            windows = [text[s : s + NER_WINDOW_CHARS] for s in starts]
            found: list[dict[str, Any]] = []
            for offset, entities in zip(starts, ner(windows, **kwargs)):
                found.extend(
                    {**e, "start": e["start"] + offset, "end": e["end"] + offset}
                    for e in entities
                    if e.get("entity_group") == "PERSON"
                )
            results.append(sorted(found, key=lambda e: e["start"]))
        return results

    return persons


class ThaiMedicalDeidBackend:
    def __init__(self, module: Any, ner: Any) -> None:
        self._module = module
        self._ner = ner

    def redact(self, text: str) -> BackendRedactionResult:
        try:
            ner = _with_stride(self._ner)
            # Pass 1 sees the whole text so regex patterns never straddle a
            # cut. On a long note, pass 2 re-runs NER over overlapping windows
            # of pass 1's output, where name recall holds. It adds names only:
            # on pass-1 output the model also tags vitals, dates and diagnoses
            # (seen on real notes), and deid's placeholder overlap check keeps
            # pass 1's tokens intact.
            final = str(self._module.anonymize_batch(ner, [text])[0][0])
            if len(final) > NER_WINDOW_CHARS:
                final = str(
                    self._module.anonymize_batch(windowed_person_ner(ner), [final])[0][
                        0
                    ]
                )
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


def _read_entry(deid_repo: Path) -> bytes:
    try:
        return (deid_repo / ENTRY_RELPATH).read_bytes()
    except OSError as exc:
        raise ThaiDeidSetupError(f"cannot read {deid_repo / ENTRY_RELPATH}") from exc


def _pipeline_version(source: bytes) -> str:
    """Read PIPELINE_VERSION from source so the policy needs no torch import."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PIPELINE_VERSION" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value:
                return value
    raise ThaiDeidSetupError("PIPELINE_VERSION not found in the entry file")


def model_manifest(model_dir: Path) -> bytes:
    """One line per file, so config.json and tokenizer edits change the pin too."""
    lines: list[str] = []
    for root, _, names in os.walk(model_dir, followlinks=True):
        for name in names:
            path = Path(root) / name
            relative = path.relative_to(model_dir).as_posix()
            try:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError as exc:
                raise ThaiDeidSetupError(f"cannot read {path}") from exc
            lines.append(f"{relative}  {digest}\n")
    if not lines:
        raise ThaiDeidSetupError(f"no model files under {model_dir}")
    return "".join(sorted(lines)).encode("utf-8")


def _dist_version(distribution: str) -> str:
    # Read from metadata so the policy command needs no torch import.
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ThaiDeidSetupError(f"{distribution} is not installed") from exc


ADAPTER_REDACTOR_MODULES = ("redactor", "roles", "date_shift", "semantic", "canonical")


def _adapter_files() -> tuple[Path, ...]:
    """This adapter plus the in-repo modules whose output is the signed text."""
    modules = [
        importlib.import_module(f"bba.deid_redactor.{name}")
        for name in ADAPTER_REDACTOR_MODULES
    ]
    return (Path(__file__), *(Path(str(m.__file__)) for m in modules))


def _adapter_digest() -> str:
    lines = "".join(
        f"{path.name}  {hashlib.sha256(path.read_bytes()).hexdigest()}\n"
        for path in _adapter_files()
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()[:16]


def redactor_version(deid_repo: Path, model_dir: Path) -> RedactorVersion:
    return _version_for(deid_repo, model_dir, _read_entry(deid_repo))


def _version_for(deid_repo: Path, model_dir: Path, source: bytes) -> RedactorVersion:
    if _git(deid_repo, "status", "--porcelain"):
        raise ThaiDeidSetupError(
            "thai-medical-deid checkout has uncommitted changes; the version "
            "would not identify the code that redacts"
        )
    head = _git(deid_repo, "rev-parse", "HEAD")
    try:
        import pythainlp
    except ImportError as exc:
        raise ThaiDeidSetupError("pythainlp is required to pin the gazetteer") from exc
    return RedactorVersion(
        version=(
            f"thai-medical-deid@{head}"
            f"+{_pipeline_version(source)}"
            f"+stride{NER_STRIDE}"
            f"+win{NER_WINDOW_CHARS}-{NER_WINDOW_STEP}"
            f"+entry-{hashlib.sha256(source).hexdigest()[:16]}"
            f"+transformers-{_dist_version('transformers')}"
            f"+tokenizers-{_dist_version('tokenizers')}"
            f"+torch-{_dist_version('torch')}"
            f"+adapter-{_adapter_digest()}"
        ),
        model_sha=hashlib.sha256(model_manifest(model_dir)).hexdigest(),
        gazetteer_version=f"pythainlp-{pythainlp.__version__}",
    )


def _exec_entry(entry: Path, source: bytes) -> ModuleType:
    # Compiling the bytes that were hashed, not importing by path, keeps a
    # stale or planted bytecode cache from running unpinned code.
    module = ModuleType("deidentify_nun")
    module.__file__ = str(entry)
    exec(compile(source, str(entry), "exec"), module.__dict__)
    return module


def load_backend(
    deid_repo: Path, model_dir: Path
) -> tuple[ThaiMedicalDeidBackend, RedactorVersion]:
    source = _read_entry(deid_repo)
    version = _version_for(deid_repo, model_dir, source)
    # Both must precede the import: the redactor imports transformers at load,
    # and a bytecode cache would dirty the checkout the version vouches for.
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.dont_write_bytecode = True
    module = _exec_entry(deid_repo / ENTRY_RELPATH, source)
    if not module._THAI_FIRST_NAMES or not module._THAI_FAMILY_NAMES:
        raise ThaiDeidSetupError(
            "Thai name gazetteer is empty (PyThaiNLP corpora missing); refusing "
            "to redact without it"
        )
    if f"+{module.PIPELINE_VERSION}+stride" not in version.version:
        raise ThaiDeidSetupError("PIPELINE_VERSION differs from the checked source")
    module.MODEL_ID = str(model_dir)
    return ThaiMedicalDeidBackend(module, module.load_ner_pipeline()), version
