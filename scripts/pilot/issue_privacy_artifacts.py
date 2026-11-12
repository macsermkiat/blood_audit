"""Issue signed privacy artifacts for the pilot LLM leg.

Runs on the operator's machine with the approved redactor (the owner's
``thai-medical-deid`` scripts). It holds the signing key; the pilot leg gets
only the artifacts and the public-key policy.

Order of work, with the same ``RUN_ID`` throughout::

    ISSUER="uv run --with 'transformers[torch]==5.22.2' --with tokenizers==0.23.3 \\
        --with torch==2.15.0 --with sentencepiece==0.2.3 --with pythainlp==5.4.0 \\
        --with pandas==3.1.1 --with tqdm==4.70.1 --with openpyxl==3.1.5 \\
        python scripts/pilot/issue_privacy_artifacts.py"

    # 1. once: a key pair outside the repo
    $ISSUER keygen --key-id kcmh-2026 --out-dir ~/.bba-keys
    # 2. once per redactor version: the trust-root policy the pilot reads
    $ISSUER policy --key-id kcmh-2026 --public-key ~/.bba-keys/kcmh-2026.pub \\
        --deid-repo ~/thai-medical-deid --model-dir ~/models/no-name-ner-th \\
        --out ~/.bba-keys/policy.json
    # 3. pilot, prepare mode: export the exact prompt requests (raw notes, 0600)
    BBA_PILOT_RUN_ID=run1 BBA_PILOT_PRIVACY_PREPARE=/secure/requests.json \\
        uv run python scripts/pilot/run_llm_leg.py
    # 4. issue: redact, sign, then check every artifact against the policy from
    #    step 2 and write nothing if any check fails
    $ISSUER issue --requests /secure/requests.json \\
        --signing-key ~/.bba-keys/kcmh-2026.ed25519 --key-id kcmh-2026 \\
        --policy ~/.bba-keys/policy.json \\
        --deid-repo ~/thai-medical-deid --model-dir ~/models/no-name-ner-th \\
        --out /secure/artifacts.json
    # 5. pilot, real run: same run id, so the signatures bind
    BBA_PILOT_RUN_ID=run1 BBA_PRIVACY_POLICY=~/.bba-keys/policy.json \\
        BBA_PRIVACY_ARTIFACTS=/secure/artifacts.json \\
        uv run python scripts/pilot/run_llm_leg.py

The heavy dependencies are supplied by ``uv run --with``; they are not project
dependencies. Their versions are part of the approved version, so use the same
pins for ``policy`` and ``issue``. The approved model SHA-256 is the hash of a
manifest of every file in the model directory (``model_manifest``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bba.deid_redactor import NoteInput, RedactionRequest
from bba.deid_redactor.admission import (
    PrivacyAdmissionError,
    PrivacyAdmissionPolicy,
    PrivacyArtifact,
    RedactionAuthority,
)
from bba.deid_redactor.exceptions import BackendRedactionError
from bba.deid_redactor.models import RedactorBackend, RedactorVersion
from bba.llm_client import BatchSubmissionRequest
from bba.prompt_builder import build_prompt
from privacy_requests import PrivacyRequests, load_requests, write_private

KEY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
KEY_BYTES = 32


@dataclass(frozen=True)
class IssueOutcome:
    artifacts: dict[str, PrivacyArtifact]
    refused: Counter[str]


def issue_artifacts(
    requests: PrivacyRequests,
    *,
    backend: RedactorBackend,
    version: RedactorVersion,
    model_artifact: Path,
    key_id: str,
    signing_key: bytes,
) -> IssueOutcome:
    authority = RedactionAuthority(
        backend=backend,
        model_artifact=model_artifact,
        version=version,
        key_id=key_id,
        signing_key=signing_key,
    )
    artifacts: dict[str, PrivacyArtifact] = {}
    refused: Counter[str] = Counter()
    for audit_id, case in requests.cases.items():
        request = RedactionRequest(
            notes=tuple(
                NoteInput(note_id=chunk.evidence_id, text=chunk.text)
                for chunk in case.prompt_request.evidence_chunks
            ),
            anchor_date=case.anchor_date,
            redactor_version=version,
        )
        try:
            artifacts[audit_id] = authority.prepare(
                request, case.prompt_request, audit_id=audit_id, run_id=requests.run_id
            )
        except (PrivacyAdmissionError, BackendRedactionError) as exc:
            # Messages are fixed strings or counts; none quote note text.
            refused[f"{type(exc).__name__}: {exc}"] += 1
    return IssueOutcome(artifacts, refused)


def _write_new(path: Path, data: bytes, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "wb") as stream:
        os.fchmod(stream.fileno(), mode)
        stream.write(data)


def keygen(key_id: str, out_dir: Path) -> tuple[Path, Path]:
    if not KEY_ID_PATTERN.fullmatch(key_id):
        raise ValueError("key id may contain only letters, digits, '.', '_' and '-'")
    private_path = out_dir / f"{key_id}.ed25519"
    public_path = out_dir / f"{key_id}.pub"
    for path in (private_path, public_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.from_private_bytes(os.urandom(KEY_BYTES))
    _write_new(private_path, key.private_bytes_raw(), 0o600)
    _write_new(public_path, key.public_key().public_bytes_raw(), 0o644)
    return private_path, public_path


def write_policy(
    *, key_id: str, public_key: Path, version: RedactorVersion, out: Path
) -> None:
    document = {
        "key_id": key_id,
        "public_key_file": str(public_key.resolve()),
        "approved_redactor": version.model_dump(mode="json"),
    }
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    out.chmod(0o644)
    # Fail now, not at the first pilot run, if the policy is unusable.
    PrivacyAdmissionPolicy.from_file(out)


def _read_signing_key(path: Path) -> bytes:
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"signing key {path} must not be accessible to others")
    key = path.read_bytes()
    if len(key) != KEY_BYTES:
        raise ValueError(f"signing key {path} must hold exactly {KEY_BYTES} raw bytes")
    return key


def _cmd_keygen(args: argparse.Namespace) -> int:
    private_path, public_path = keygen(args.key_id, args.out_dir)
    print(f"private key: {private_path}\npublic key: {public_path}")
    return 0


def _cmd_policy(args: argparse.Namespace) -> int:
    from thai_deid_backend import redactor_version

    version = redactor_version(args.deid_repo, args.model_dir)
    write_policy(
        key_id=args.key_id, public_key=args.public_key, version=version, out=args.out
    )
    print(f"policy: {args.out}\nredactor: {version.version}")
    return 0


def verify_against_policy(
    requests: PrivacyRequests,
    artifacts: dict[str, PrivacyArtifact],
    policy: PrivacyAdmissionPolicy,
) -> None:
    """Rebuild each prompt as run_llm_leg.py does and run the pilot's own check."""
    for audit_id, artifact in artifacts.items():
        prompt_request = requests.cases[audit_id].prompt_request
        prompt = build_prompt(
            prompt_request.model_copy(
                update={"evidence_chunks": artifact.evidence_chunks}
            )
        )
        policy.verify(
            BatchSubmissionRequest(
                audit_id=audit_id,
                run_id=requests.run_id,
                task_mode=prompt_request.task_mode,
                prompt=prompt,
                privacy_attestation=artifact.attestation,
            )
        )


def issue_to_file(
    requests: PrivacyRequests,
    *,
    backend: RedactorBackend,
    version: RedactorVersion,
    manifest: bytes,
    key_id: str,
    signing_key: bytes,
    policy: PrivacyAdmissionPolicy,
    out: Path,
) -> int:
    with tempfile.TemporaryDirectory() as scratch:
        # The authority hashes one file; the manifest is that file.
        manifest_path = Path(scratch) / "model.manifest"
        manifest_path.write_bytes(manifest)
        outcome = issue_artifacts(
            requests,
            backend=backend,
            version=version,
            model_artifact=manifest_path,
            key_id=key_id,
            signing_key=signing_key,
        )
    print(f"issued: {len(outcome.artifacts)}, refused: {sum(outcome.refused.values())}")
    for reason, count in sorted(outcome.refused.items()):
        print(f"  {count} x {reason}")
    if not outcome.artifacts:
        return 1
    try:
        verify_against_policy(requests, outcome.artifacts, policy)
    except PrivacyAdmissionError as exc:
        # The pilot would reject every row and burn the run id; this is a
        # key, policy or redactor mismatch, not a bad case, so write nothing.
        print(f"policy check failed, nothing written: {exc}")
        return 1
    write_private(
        out,
        json.dumps(
            {k: v.model_dump(mode="json") for k, v in outcome.artifacts.items()},
            ensure_ascii=False,
            indent=2,
        ),
    )
    print(f"artifacts: {out}")
    return 0


def _cmd_issue(args: argparse.Namespace) -> int:
    from thai_deid_backend import load_backend, model_manifest

    requests = load_requests(args.requests)
    policy = PrivacyAdmissionPolicy.from_file(args.policy)
    backend, version = load_backend(args.deid_repo, args.model_dir)
    return issue_to_file(
        requests,
        backend=backend,
        version=version,
        manifest=model_manifest(args.model_dir),
        key_id=args.key_id,
        signing_key=_read_signing_key(args.signing_key),
        policy=policy,
        out=args.out,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler: object) -> argparse.ArgumentParser:
        command = sub.add_parser(name)
        command.set_defaults(handler=handler)
        return command

    keygen_cmd = add("keygen", _cmd_keygen)
    keygen_cmd.add_argument("--key-id", required=True)
    keygen_cmd.add_argument("--out-dir", required=True, type=Path)

    policy_cmd = add("policy", _cmd_policy)
    policy_cmd.add_argument("--key-id", required=True)
    policy_cmd.add_argument("--public-key", required=True, type=Path)
    policy_cmd.add_argument("--out", required=True, type=Path)

    issue_cmd = add("issue", _cmd_issue)
    issue_cmd.add_argument("--requests", required=True, type=Path)
    issue_cmd.add_argument("--signing-key", required=True, type=Path)
    issue_cmd.add_argument("--key-id", required=True)
    issue_cmd.add_argument("--policy", required=True, type=Path)
    issue_cmd.add_argument("--out", required=True, type=Path)

    for command in (policy_cmd, issue_cmd):
        command.add_argument("--deid-repo", required=True, type=Path)
        command.add_argument("--model-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handler = args.handler
    result: int = handler(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
