"""The issuer turns pilot prompt requests into artifacts the pilot admits."""

from __future__ import annotations

import hashlib
import stat
from datetime import date
from pathlib import Path

import pytest

from bba.deid_redactor import RedactorVersion
from bba.deid_redactor.admission import (
    PrivacyAdmissionError,
    PrivacyAdmissionPolicy,
    load_privacy_artifacts,
)
from bba.llm_client.models import BatchSubmissionRequest
from bba.prompt_builder import EvidenceChunk, PromptBuildRequest, build_prompt
from issue_privacy_artifacts import (
    issue_artifacts,
    keygen,
    write_policy,
)
from privacy_requests import (
    ARTIFACTS_ENV,
    PREPARE_ENV,
    PrivacyCase,
    PrivacyPrepareError,
    PrivacyRequests,
    bangkok_anchor,
    load_requests,
    prepare_path,
    write_private,
    write_requests,
)
from thai_deid_backend import ThaiMedicalDeidBackend
from thai_deid_fakes import fake_module

NOTE = (
    "Dr. Somchai Jaidee ordered PRC for Hb 6.8. Call 081-234-5678. "
    "Repeat Hb on 2026-05-03."
)
REPLACEMENTS = {"Somchai Jaidee": "[PERSON]", "081-234-5678": "[PHONE]"}


def _request(text: str = NOTE) -> PromptBuildRequest:
    return PromptBuildRequest(
        task_mode="HB_7_10_REVIEW",
        cohort_threshold=7.0,
        evidence_chunks=(
            EvidenceChunk(evidence_id="E1", source="IPDADMPROGRESS", text=text),
        ),
        few_shot_examples=(),
    )


def _requests(**cases: str) -> PrivacyRequests:
    return PrivacyRequests(
        run_id="run1",
        cases={
            audit_id: PrivacyCase(
                anchor_date=date(2026, 5, 1), prompt_request=_request(text)
            )
            for audit_id, text in cases.items()
        },
    )


@pytest.fixture
def issuer(tmp_path: Path):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"weights")
    version = RedactorVersion(
        version="thai-medical-deid@abc+pv1",
        model_sha=hashlib.sha256(b"weights").hexdigest(),
        gazetteer_version="pythainlp-9.9",
    )
    private_path, public_path = keygen("test-key", tmp_path / "keys")

    def issue(requests: PrivacyRequests, **module_kwargs: str):
        return issue_artifacts(
            requests,
            backend=ThaiMedicalDeidBackend(
                fake_module(REPLACEMENTS, **module_kwargs), None
            ),
            version=version,
            model_artifact=model,
            key_id="test-key",
            signing_key=private_path.read_bytes(),
        )

    policy_path = tmp_path / "policy.json"
    write_policy(
        key_id="test-key", public_key=public_path, version=version, out=policy_path
    )
    return issue, PrivacyAdmissionPolicy.from_file(policy_path)


def _submission(requests: PrivacyRequests, outcome, audit_id: str, run_id: str):
    artifact = outcome.artifacts[audit_id]
    prompt = build_prompt(
        requests.cases[audit_id].prompt_request.model_copy(
            update={"evidence_chunks": artifact.evidence_chunks}
        )
    )
    return BatchSubmissionRequest(
        audit_id=audit_id,
        run_id=run_id,
        task_mode=prompt.task_mode,
        prompt=prompt,
        privacy_attestation=artifact.attestation,
    )


def test_issued_artifact_is_clean_and_admitted_for_its_own_run_only(issuer):
    issue, policy = issuer
    requests = _requests(a1=NOTE)
    outcome = issue(requests)

    text = outcome.artifacts["a1"].evidence_chunks[0].text
    assert "Somchai" not in text and "081-234" not in text
    assert "[ATTENDING]" in text and "[PHONE]" in text
    assert "Day +2" in text and "2026-05-03" not in text

    policy.verify(_submission(requests, outcome, "a1", "run1"))
    with pytest.raises(PrivacyAdmissionError, match="exact prompt"):
        policy.verify(_submission(requests, outcome, "a1", "run2"))


def test_unsafe_and_failing_cases_are_refused_by_reason_without_note_text(issuer):
    issue, _ = issuer
    outcome = issue(
        _requests(ok=NOTE, dense="[NURSE]" * 5, boom="Hb 6 CRASH-TRIGGER-NAME"),
        fail_on="CRASH-TRIGGER-NAME",
    )

    assert set(outcome.artifacts) == {"ok"}
    assert sum(outcome.refused.values()) == 2
    assert any(k.startswith("PrivacyAdmissionError") for k in outcome.refused)
    assert any(k.startswith("BackendRedactionError") for k in outcome.refused)
    assert not any("CRASH-TRIGGER-NAME" in k for k in outcome.refused)


def test_artifact_map_round_trips_through_the_pilot_loader(issuer, tmp_path: Path):
    import json

    issue, _ = issuer
    outcome = issue(_requests(a1=NOTE))
    path = tmp_path / "artifacts.json"
    write_private(
        path,
        json.dumps(
            {k: v.model_dump(mode="json") for k, v in outcome.artifacts.items()}
        ),
    )

    assert load_privacy_artifacts(path) == outcome.artifacts
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_request_file_round_trips_and_is_owner_only(tmp_path: Path):
    requests = _requests(a1=NOTE)
    path = tmp_path / "requests.json"
    path.write_text("stale")
    path.chmod(0o644)

    write_requests(path, requests)

    assert load_requests(path) == requests
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_prepare_with_artifacts_is_refused_because_chunks_would_be_redacted(
    tmp_path: Path,
):
    with pytest.raises(PrivacyPrepareError, match=ARTIFACTS_ENV):
        prepare_path({PREPARE_ENV: str(tmp_path / "r.json"), ARTIFACTS_ENV: "a.json"})
    assert prepare_path({ARTIFACTS_ENV: "a.json"}) is None
    assert prepare_path({PREPARE_ENV: "r.json"}) == Path("r.json")


def test_anchor_is_the_bangkok_calendar_date_not_the_utc_date():
    from datetime import datetime, timezone

    # 20:00 UTC on May 1 is 03:00 on May 2 in Bangkok.
    assert bangkok_anchor(datetime(2026, 5, 1, 20, 0, tzinfo=timezone.utc)) == date(
        2026, 5, 2
    )
    with pytest.raises(PrivacyPrepareError):
        bangkok_anchor(datetime(2026, 5, 1, 20, 0))


def test_keygen_refuses_to_overwrite_either_key(tmp_path: Path):
    private_path, public_path = keygen("k1", tmp_path)
    assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(public_path.stat().st_mode) == 0o644
    assert len(private_path.read_bytes()) == len(public_path.read_bytes()) == 32

    before = private_path.read_bytes()
    with pytest.raises(FileExistsError):
        keygen("k1", tmp_path)
    public_path.unlink()
    with pytest.raises(FileExistsError):
        keygen("k1", tmp_path)
    assert private_path.read_bytes() == before
    with pytest.raises(ValueError):
        keygen("../escape", tmp_path)


def test_policy_file_loads_and_pins_the_approved_redactor(issuer):
    _, policy = issuer

    assert policy.key_id == "test-key"
    assert policy.approved_redactor.version == "thai-medical-deid@abc+pv1"
