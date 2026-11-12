"""The thai-medical-deid adapter hands redact_bundle spans that point at the PHI."""

from __future__ import annotations

import hashlib
import importlib.util
import py_compile
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from bba.deid_redactor import (
    NoteInput,
    RedactionRequest,
    RedactorVersion,
    redact_bundle,
)
from bba.deid_redactor.exceptions import BackendRedactionError
from thai_deid_backend import (
    _windows,
    NER_WINDOW_CHARS,
    NER_WINDOW_STEP,
    ThaiDeidSetupError,
    ThaiMedicalDeidBackend,
    load_backend,
    model_manifest,
    redactor_version,
    windowed_person_ner,
)
import thai_deid_backend
from thai_deid_fakes import (
    ContextShiftNer,
    LengthLimitedNer,
    WholeTextIdNer,
    WindowedNer,
    fake_module,
    ner_driven_module,
)

LIBRARY_VERSIONS = {"transformers": "5.1", "tokenizers": "0.2", "torch": "2.3"}
VERSION = RedactorVersion(version="v", model_sha="a" * 64, gazetteer_version="g")


def _backend(replacements: dict[str, str], **kwargs: str) -> ThaiMedicalDeidBackend:
    return ThaiMedicalDeidBackend(fake_module(replacements, **kwargs), None)


def _redact(backend: ThaiMedicalDeidBackend, text: str) -> str:
    result = redact_bundle(
        RedactionRequest(
            notes=(NoteInput(note_id="E1", text=text),),
            anchor_date=date(2026, 5, 1),
            redactor_version=VERSION,
        ),
        backend=backend,
    )
    return result.notes[0].redacted_text


def test_titled_name_span_points_at_the_name_in_the_original():
    text = "Hb 6.8 ordered by Dr. Somchai Jaidee at night"
    result = _backend({"Somchai Jaidee": "[PERSON]"}).redact(text)

    (span,) = result.spans
    assert span.entity_type == "PERSON"
    assert text[span.start : span.end] == "Somchai Jaidee"
    assert span.original_text == "Somchai Jaidee"


def test_thai_digits_next_to_a_name_do_not_widen_its_span():
    # Without aligning against the digit-normalized text, the digits and the
    # name merge into one replaced block and the span swallows the room number.
    text = "ห้อง๑๒๓สมชาย"
    (span,) = _backend({"สมชาย": "[PERSON]"}).redact(text).spans

    assert text[span.start : span.end] == "สมชาย"


def test_token_sharing_letters_with_the_phi_still_covers_all_of_it():
    text = "Dr. SOMCHAI saw him"
    (span,) = _backend({"SOMCHAI": "[PERSON]"}).redact(text).spans

    assert text[span.start : span.end] == "SOMCHAI"


def test_nurse_and_physician_cues_survive_into_role_tokens():
    text = "RN Alice reported to Dr. Bob that Hb is 6."
    redacted = _redact(_backend({"Alice": "[PERSON]", "Bob": "[PERSON]"}), text)

    assert redacted == "RN [NURSE] reported to Dr. [ATTENDING] that Hb is 6."


def test_tokens_already_in_the_note_are_spans_and_get_roles():
    # The HOSxP export was masked by thai-medical-deid once already, so notes
    # arrive with [PERSON]/[DATE] tokens; each needs a span or redact_bundle
    # refuses the note.
    text = "nurse [PERSON] called on [DATE]; Hb 6 seen by Dr. Bob"
    backend = _backend({"Bob": "[PERSON]"})

    spans = backend.redact(text).spans
    assert [s.entity_type for s in spans] == ["PERSON", "DATE", "PERSON"]
    assert text[spans[0].start : spans[0].end] == "[PERSON]"
    assert _redact(backend, text) == (
        "nurse [NURSE] called on [DATE]; Hb 6 seen by Dr. [ATTENDING]"
    )


def test_model_failure_message_never_quotes_the_note():
    backend = _backend({}, fail_on="SECRET-NAME")
    with pytest.raises(BackendRedactionError) as caught:
        _redact(backend, "Hb 6 SECRET-NAME")

    assert "SECRET-NAME" not in str(caught.value)
    assert "RuntimeError" in str(caught.value)


def test_name_past_the_ner_truncation_point_is_masked_through_redact_bundle():
    name = "Somchai Jaidee"
    text = "Hb 6.8 stable. " * 40 + f"Called by Dr. {name} at night."
    ner = WindowedNer((name,), window=200)
    assert ner([text], batch_size=1) == [[]]

    backend = ThaiMedicalDeidBackend(ner_driven_module(), ner)
    redacted = _redact(backend, text)

    assert name not in redacted
    assert redacted.endswith("Called by Dr. [ATTENDING] at night.")


def test_name_the_model_misses_in_a_long_note_is_masked_by_the_window_pass():
    name = "Somchai Jaidee"
    text = "Hb 6.8 stable. " * 40 + f"Called by Dr. {name} at night."
    ner = LengthLimitedNer((), (name,), limit=NER_WINDOW_CHARS)
    assert len(text) > NER_WINDOW_CHARS

    redacted = _redact(ThaiMedicalDeidBackend(ner_driven_module(), ner), text)

    assert name not in redacted
    assert redacted.endswith("Called by Dr. [ATTENDING] at night.")


def test_window_pass_leaves_a_pass_one_placeholder_as_one_token():
    text = "RN Alice noted. " + "Hb 6.8 stable. " * 40 + "Dr. Bob agreed."
    ner = LengthLimitedNer(("Alice",), ("Bob",), limit=NER_WINDOW_CHARS)
    backend = ThaiMedicalDeidBackend(ner_driven_module(), ner)

    result = backend.redact(text)

    assert result.text.startswith("RN [PERSON] noted.")
    assert result.text.endswith("Dr. [PERSON] agreed.")
    assert "[[" not in result.text
    assert [s.original_text for s in result.spans] == ["Alice", "Bob"]
    assert _redact(backend, text).startswith("RN [NURSE] noted.")


def _filler(length: int) -> str:
    return ("Hb 6.8 stable. " * (length // 15 + 1))[:length]


def test_name_straddling_a_non_overlapping_window_edge_is_masked_whole():
    # The name spans the 400 mark: neither [0,400) nor a [400,800) window holds
    # it, but the overlapping window starting at 250 does.
    name = "Somchai Jaidee"
    text = _filler(395) + name + _filler(600)
    ner = LengthLimitedNer((), (name,), limit=NER_WINDOW_CHARS)
    assert text.index(name) < NER_WINDOW_CHARS < text.index(name) + len(name)

    redacted = ThaiMedicalDeidBackend(ner_driven_module(), ner).redact(text).text

    assert "Somchai" not in redacted
    assert "Jaidee" not in redacted
    assert redacted.count("[PERSON]") == 1


def test_entity_offsets_map_back_for_a_name_in_the_third_window():
    name = "Somchai Jaidee"
    text = _filler(700) + name + _filler(300)
    position = text.index(name)
    second_end = NER_WINDOW_STEP + NER_WINDOW_CHARS
    third_start = 2 * NER_WINDOW_STEP
    assert second_end <= position
    assert position + len(name) <= third_start + NER_WINDOW_CHARS
    ner = LengthLimitedNer((), (name,), limit=NER_WINDOW_CHARS)

    result = ThaiMedicalDeidBackend(ner_driven_module(), ner).redact(text)

    assert result.text == _filler(700) + "[PERSON]" + _filler(300)
    (span,) = result.spans
    assert text[span.start : span.end] == name


def test_window_pass_runs_on_pass_one_output_not_the_original_text():
    code = "MRN-4471"
    name = "Somchai"
    text = _filler(300) + f" {code} " + _filler(300) + f" Dr. {name} agreed."
    ner = WholeTextIdNer(code, name, limit=NER_WINDOW_CHARS)

    redacted = ThaiMedicalDeidBackend(ner_driven_module(), ner).redact(text).text

    assert code not in redacted
    assert name not in redacted
    assert "[HOSPITAL_IDS]" in redacted


def test_windowed_ner_gives_one_sorted_person_list_per_text_with_scores_kept():
    seen: list[list[int]] = []

    def ner(windows, **kwargs):
        seen.append([len(w) for w in windows])
        return [
            [
                {"start": 5, "end": 9, "entity_group": "PERSON", "score": 0.9},
                {"start": 1, "end": 3, "entity_group": "DATE", "score": 0.8},
            ]
            for _ in windows
        ]

    out = windowed_person_ner(ner)(["x" * 1000, "short"], batch_size=1)

    assert seen == [[400, 400, 400, 250], [5]]
    assert [e["start"] for e in out[0]] == [5, 255, 505, 755]
    assert all(e["score"] == 0.9 and e["entity_group"] == "PERSON" for e in out[0])
    assert out[1] == [{"start": 5, "end": 9, "entity_group": "PERSON", "score": 0.9}]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-C",
            str(repo),
            *args,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def deid_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "pythainlp", SimpleNamespace(__version__="9.9"))
    monkeypatch.setattr(
        thai_deid_backend, "_dist_version", lambda name: LIBRARY_VERSIONS[name]
    )
    repo = tmp_path / "tmd"
    repo.mkdir(parents=True)
    entry = repo / "deidentify_nun.py"
    entry.write_text(
        'PIPELINE_VERSION = "pv1"\n'
        'MODEL_ID = "hub/model"\n'
        "_THAI_FIRST_NAMES = {'somchai'}\n"
        "_THAI_FAMILY_NAMES = {'jaidee'}\n"
        "_THAI_DIGIT_TABLE = {}\n"
        "def load_ner_pipeline():\n    return MODEL_ID\n"
        "def anonymize_batch(ner, texts):\n    return [(t, []) for t in texts]\n"
    )
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    return repo, model_dir, entry


def test_version_names_the_commit_pipeline_weights_and_gazetteer(deid_checkout):
    repo, model_dir, entry = deid_checkout
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    version = redactor_version(repo, model_dir)

    entry_sha = hashlib.sha256(entry.read_bytes()).hexdigest()[:16]
    assert version.version == (
        f"thai-medical-deid@{head}+pv1+stride128+win400-250+entry-{entry_sha}"
        f"+transformers-5.1+tokenizers-0.2+torch-2.3"
        f"+adapter-{thai_deid_backend._adapter_digest()}"
    )
    assert version.model_sha == hashlib.sha256(model_manifest(model_dir)).hexdigest()
    assert version.gazetteer_version == "pythainlp-9.9"


def test_editing_any_model_file_changes_the_pinned_model_sha(deid_checkout):
    repo, model_dir, _ = deid_checkout
    (model_dir / "config.json").write_text('{"id2label": {"1": "B-PERSON"}}')
    pinned = redactor_version(repo, model_dir).model_sha

    (model_dir / "config.json").write_text('{"id2label": {"1": "O"}}')

    assert redactor_version(repo, model_dir).model_sha != pinned


def test_manifest_lists_every_file_including_those_behind_symlinks(tmp_path: Path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "abc").write_bytes(b"weights")
    snapshot = tmp_path / "snapshot"
    (snapshot / "sub").mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to(blobs / "abc")
    (snapshot / "sub" / "vocab.txt").write_bytes(b"v")

    manifest = model_manifest(snapshot).decode()

    assert manifest == (
        f"model.safetensors  {hashlib.sha256(b'weights').hexdigest()}\n"
        f"sub/vocab.txt  {hashlib.sha256(b'v').hexdigest()}\n"
    )


def test_version_changes_when_a_library_version_changes(deid_checkout, monkeypatch):
    repo, model_dir, _ = deid_checkout
    pinned = redactor_version(repo, model_dir).version

    monkeypatch.setitem(LIBRARY_VERSIONS, "transformers", "5.2")

    assert redactor_version(repo, model_dir).version != pinned


def test_dirty_checkout_is_refused(deid_checkout):
    repo, model_dir, entry = deid_checkout
    entry.write_text(entry.read_text() + "# local tweak\n")

    with pytest.raises(ThaiDeidSetupError, match="uncommitted"):
        redactor_version(repo, model_dir)


def test_load_backend_pins_the_hashed_weights_offline(deid_checkout):
    import os

    repo, model_dir, _ = deid_checkout
    backend, version = load_backend(repo, model_dir)

    assert backend._ner == str(model_dir)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert version == redactor_version(repo, model_dir)


def test_empty_name_gazetteer_is_refused(deid_checkout):
    repo, model_dir, entry = deid_checkout
    entry.write_text(entry.read_text().replace("{'somchai'}", "set()"))
    _git(repo, "commit", "-q", "-am", "empty gazetteer")

    with pytest.raises(ThaiDeidSetupError, match="gazetteer is empty"):
        load_backend(repo, model_dir)


def test_window_pass_adds_names_but_not_ids_or_vitals():
    # On real pass-1 output the model tagged a BP range and a date as
    # HOSPITAL_IDS; pass 2 exists for names only.
    text = "BP 79-111/42 stable. " + "Hb 6.8 stable. " * 40 + "Dr. Bob agreed."
    ner = LengthLimitedNer(
        (), ("Bob",), limit=NER_WINDOW_CHARS, short_only_ids=("79-111/42",)
    )

    redacted = _redact(ThaiMedicalDeidBackend(ner_driven_module(), ner), text)

    assert redacted.startswith("BP 79-111/42 stable.")
    assert redacted.endswith("Dr. [ATTENDING] agreed.")


def test_short_chunk_is_not_redacted_twice():
    # Real case: "ICD-10 I481: Melaena" became "...: [PERSON]" when pass 2 saw
    # pass 1's placeholders; a chunk that fits one segment gets one pass.
    text = "ICD-10 I481: Melaena"
    ner = ContextShiftNer(code="I481", word="Melaena")

    result = ThaiMedicalDeidBackend(ner_driven_module(), ner).redact(text)

    assert result.text == "ICD-10 [HOSPITAL_IDS]: Melaena"


def test_a_changed_entry_file_changes_the_version(deid_checkout):
    repo, model_dir, entry = deid_checkout

    def entry_pin() -> str:
        return next(
            part
            for part in redactor_version(repo, model_dir).version.split("+")
            if part.startswith("entry-")
        )

    pinned = entry_pin()

    entry.write_text(entry.read_text() + "# reviewed edit\n")
    _git(repo, "commit", "-q", "-am", "edit")

    assert entry_pin() != pinned


def test_load_backend_runs_the_bytes_it_pinned_not_a_planted_bytecode_cache(
    deid_checkout, tmp_path: Path
):
    repo, model_dir, entry = deid_checkout
    (repo / ".gitignore").write_text("__pycache__/\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "ignore caches")
    planted = tmp_path / "planted.py"
    planted.write_text(entry.read_text() + 'PLANTED = "yes"\n')
    cache = Path(importlib.util.cache_from_source(str(entry)))
    cache.parent.mkdir()
    py_compile.compile(
        str(planted),
        cfile=str(cache),
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )

    backend, _ = load_backend(repo, model_dir)

    assert not hasattr(backend._module, "PLANTED")
    assert backend._module.__name__ == "deidentify_nun"
    assert backend._module.__file__ == str(entry)


def test_load_backend_refuses_a_module_whose_pipeline_version_differs(deid_checkout):
    repo, model_dir, entry = deid_checkout
    entry.write_text(entry.read_text() + 'PIPELINE_VERSION = "pv2"\n')
    _git(repo, "commit", "-q", "-am", "late reassignment")

    with pytest.raises(ThaiDeidSetupError, match="PIPELINE_VERSION differs"):
        load_backend(repo, model_dir)


def test_adapter_digest_covers_the_adapter_and_the_post_processing_modules():
    names = [path.name for path in thai_deid_backend._adapter_files()]

    assert names == [
        "thai_deid_backend.py",
        "redactor.py",
        "roles.py",
        "date_shift.py",
        "semantic.py",
        "canonical.py",
    ]


@pytest.mark.parametrize("index", range(6))
def test_adapter_digest_changes_when_any_covered_file_changes(
    index: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    copies = []
    for original in thai_deid_backend._adapter_files():
        copy = tmp_path / original.name
        copy.write_bytes(original.read_bytes())
        copies.append(copy)
    monkeypatch.setattr(thai_deid_backend, "_adapter_files", lambda: tuple(copies))
    pinned = thai_deid_backend._adapter_digest()

    copies[index].write_bytes(copies[index].read_bytes() + b"\n# edit\n")

    assert thai_deid_backend._adapter_digest() != pinned


def test_manifest_follows_a_symlinked_directory(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "vocab.txt").write_bytes(b"v")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"w")
    (snapshot / "tokenizer").symlink_to(outside, target_is_directory=True)

    manifest = model_manifest(snapshot).decode()

    assert f"tokenizer/vocab.txt  {hashlib.sha256(b'v').hexdigest()}\n" in manifest


def test_window_edges_land_on_whitespace_and_windows_still_overlap():
    # A window ending inside a name could let the model tag only part of it.
    text = " ".join(f"word{i:03d}" for i in range(200))
    spans = _windows(text)

    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for start, end in spans:
        assert start == 0 or text[start - 1].isspace()
        assert end == len(text) or text[end].isspace()
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start < prev_end
