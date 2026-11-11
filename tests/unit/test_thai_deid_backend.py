"""The thai-medical-deid adapter hands redact_bundle spans that point at the PHI."""

from __future__ import annotations

import hashlib
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
    ThaiDeidSetupError,
    ThaiMedicalDeidBackend,
    load_backend,
    model_manifest,
    redactor_version,
)
import thai_deid_backend
from thai_deid_fakes import WindowedNer, fake_module, ner_driven_module

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
    repo, model_dir, _ = deid_checkout
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    version = redactor_version(repo, model_dir)

    adapter = hashlib.sha256(Path(thai_deid_backend.__file__).read_bytes())
    assert version.version == (
        f"thai-medical-deid@{head}+pv1+stride128+transformers-5.1+tokenizers-0.2"
        f"+torch-2.3+adapter-{adapter.hexdigest()[:16]}"
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
