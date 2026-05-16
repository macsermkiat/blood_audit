"""RED-phase failing tests for issue #29 (``bba.cli``).

Each ``class`` maps to one acceptance criterion in the issue body OR one
of the user-supplied operational constraints:

* AC ① "Implementation in src/bba/cli.py"
    → :class:`TestModulePublicSurface`, :class:`TestCommandInputModels`,
      :class:`TestPhiSurface`
* AC ② "All 6 subcommands invokeable with --help returning usable text"
    → :class:`TestHelpListsAllSubcommands`,
      :class:`TestSubcommandHelpExitsZero`,
      :class:`TestAuditHelpExposesForceFlag`
* AC ③ "Run-level idempotency: bba audit twice on same input is a no-op"
    → :class:`TestAuditIdempotencyNoop`,
      :class:`TestRunIdRecipe`,
      :class:`TestRunIdDeterminism`,
      :class:`TestRunIdSensitivity`,
      :class:`TestCodeVersionFromMetadata`
* AC ④ "--force flag overrides idempotency with audit-log entry"
    → :class:`TestAuditForceOverride`,
      :class:`TestAuditForceLogsOverridePerInvocation`
* AC ⑤ "Exception scrubbing: injected exception with `bundle` in scope
        → scrubbed traceback in logs"
    → :class:`TestPhiScrubbingInTraceback`,
      :class:`TestScrubTracebackRedactsBundleLocals`,
      :class:`TestScrubTracebackRedactsHnDigitRuns`,
      :class:`TestScrubTracebackRedactsThaiHonorifics`
* AC ⑥ "Coverage ≥ 60%; ruff + mypy clean" — verified by the build, not
        by a behavioural test.

User-supplied boundary constraints (from the top-of-prompt MUST-NOT
list):

* "No ``bba migrate`` / ``bba init-db`` / ``bba serve-api`` / ``--dry-run``"
    → :class:`TestNoForbiddenSubcommands`
* "No retry / backoff in CLI — it lives in bba.llm_client"
    → :class:`TestNoRetryBackoffInCli`
* "No env vars beyond BBA_DATA_DIR, BBA_DB_URL, ANTHROPIC_API_KEY"
    → :class:`TestEnvVarSurfaceIsTight`
* "Subcommands are thin glue — no business logic in cli/main.py"
    → :class:`TestMainModuleIsThin`

Tests assert *contracts* (the WHY), not implementation choices. In this
RED scaffold:

* Every behavioural test FAILS — calls into the scaffold raise
  ``NotImplementedError`` or attempt to invoke an un-attached
  subcommand, which click reports as ``UsageError`` with exit_code 2.
* Public-surface / constant / model-validation / boundary-regression
  tests PASS — they encode the *interface* contract (frozen models,
  constant shape, "must not contain these substrings"). The convention
  matches ``test_audit_store``, ``test_eval_harness``, and
  ``test_monitoring``.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

# Public-surface check: a missing re-export fails collection before any
# test runs. Mirrors the test_monitoring convention.
from bba.cli import (
    PHI_LOCAL_NAME_REGEX,
    PHI_REGEXES,
    RUN_ID_LENGTH,
    AuditCommandInput,
    AuditRunStore,
    CliError,
    CodeVersion,
    EvaluateCommandInput,
    IdempotencyError,
    IngestCommandInput,
    InputCsvHash,
    MutuallyExclusiveOptionError,
    ReportCommandInput,
    ReportFormat,
    RunId,
    RunNotFoundError,
    SchemaFingerprint,
    SchemaVersion,
    SentinelCadence,
    SentinelCommandInput,
    ServeDashboardInput,
    cli,
    code_version,
    compute_run_id,
    install_excepthook,
    scrub_traceback,
)

if TYPE_CHECKING:
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PACKAGE_DIR = REPO_ROOT / "src" / "bba" / "cli"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner() -> CliRunner:
    """A click CliRunner for invoking subcommands in tests."""
    return CliRunner()


@pytest.fixture(autouse=True)
def _restore_excepthook() -> Iterator[None]:
    """Snapshot ``sys.excepthook`` before each test and restore on tear-down.

    The CLI's root callback installs a PHI-scrubbing excepthook; without
    a fixture-scoped restore, that hook leaks into subsequent tests
    (and into pytest's own crash reporter) and pollutes diagnostics."""
    original = sys.excepthook
    try:
        yield
    finally:
        sys.excepthook = original


@pytest.fixture
def fake_csv(tmp_path: Path) -> Path:
    """A HOSxP-named CSV that satisfies ``click.Path(exists=True)``.

    File stem is ``BDVST`` (a known HOSxP table) so the bundle-aware
    :func:`compute_run_id` picks it up — see
    :class:`TestRunIdBundleAwareness` for why a non-HOSxP stem would
    raise."""
    p = tmp_path / "BDVST.csv"
    p.write_bytes(b"HN,AN,REQNO\n00000001,A1,R1\n")
    return p


@pytest.fixture
def stub_store() -> MagicMock:
    """A MagicMock implementing the :class:`AuditRunStore` Protocol.

    Tests must set ``run_complete``, ``run_count``,
    ``record_idempotency_override``, ``audit_log_entries`` explicitly —
    spec=AuditRunStore catches typos in method names at attribute-access
    time."""
    return MagicMock(spec=AuditRunStore)


# =============================================================================
# AC ① — Module exists with the expected public surface + model contracts
# =============================================================================


class TestModulePublicSurface:
    """The imports at the top of this file already cover most of the
    surface. This class asserts a few invariants whose VALUES are part
    of the contract — a silent change here is a regression even when
    the public surface still resolves."""

    def test_public_all_includes_every_expected_export(self) -> None:
        """``bba.cli.__all__`` lists every name this test module imports
        from the package — a regression guard so a future refactor
        cannot quietly drop one from the public surface even if the
        top-of-file import line happens to still resolve.

        The expected-names list is the union of all names referenced in
        the import block at the top of this file. Touching each name
        below also satisfies ruff's F401 unused-import check, making
        the import block its own public-surface contract."""
        import bba.cli as cli_pkg

        # Touch each name so ruff considers them used. Side-effect-free.
        _referenced: tuple[object, ...] = (
            AuditCommandInput,
            AuditRunStore,
            CliError,
            CodeVersion,
            EvaluateCommandInput,
            IdempotencyError,
            IngestCommandInput,
            InputCsvHash,
            MutuallyExclusiveOptionError,
            PHI_LOCAL_NAME_REGEX,
            PHI_REGEXES,
            RUN_ID_LENGTH,
            ReportCommandInput,
            ReportFormat,
            RunId,
            RunNotFoundError,
            SchemaFingerprint,
            SchemaVersion,
            SentinelCadence,
            SentinelCommandInput,
            ServeDashboardInput,
            cli,
            code_version,
            compute_run_id,
            install_excepthook,
            scrub_traceback,
        )
        assert _referenced  # exercise the tuple so it cannot be elided.

        expected_names: frozenset[str] = frozenset({
            "AuditCommandInput",
            "AuditRunStore",
            "CliError",
            "CodeVersion",
            "EvaluateCommandInput",
            "IdempotencyError",
            "IngestCommandInput",
            "InputCsvHash",
            "MutuallyExclusiveOptionError",
            "PHI_LOCAL_NAME_REGEX",
            "PHI_REGEXES",
            "RUN_ID_LENGTH",
            "ReportCommandInput",
            "ReportFormat",
            "RunId",
            "RunNotFoundError",
            "SchemaFingerprint",
            "SchemaVersion",
            "SentinelCadence",
            "SentinelCommandInput",
            "ServeDashboardInput",
            "cli",
            "code_version",
            "compute_run_id",
            "install_excepthook",
            "scrub_traceback",
        })
        missing = expected_names - set(cli_pkg.__all__)
        assert missing == set(), (
            f"names referenced in tests but missing from bba.cli.__all__: "
            f"{sorted(missing)}"
        )

    def test_cli_is_a_click_group(self) -> None:
        assert isinstance(cli, click.Group)

    def test_cli_name_is_bba(self) -> None:
        assert cli.name == "bba"

    def test_run_id_length_is_sixteen(self) -> None:
        """RUN_ID_LENGTH is part of the persisted-row identity contract.

        Changing it invalidates every audit_log row keyed on a prior
        run_id, so it's pinned here as a regression guard."""
        assert RUN_ID_LENGTH == 16

    def test_phi_regexes_nonempty_tuple_of_patterns(self) -> None:
        assert isinstance(PHI_REGEXES, tuple)
        assert len(PHI_REGEXES) >= 3
        for r in PHI_REGEXES:
            assert isinstance(r, re.Pattern)

    def test_phi_local_name_regex_matches_six_kinds(self) -> None:
        """The frame-local name pattern must catch the six PHI-bearing
        kinds the issue body calls out: bundle / patient / note / hn /
        an / encounter (case-insensitive prefix)."""
        for name in (
            "bundle", "patient", "note", "hn", "an", "encounter",
            "Bundle", "PATIENT", "Hn_value", "AN_NUMBER",
            "encounter_id", "noteText",
        ):
            assert PHI_LOCAL_NAME_REGEX.match(name), name

    def test_phi_local_name_regex_rejects_unrelated_names(self) -> None:
        """The pattern must NOT match unrelated locals that share a
        prefix (the regex is anchored with ``^``)."""
        for name in ("config", "result", "ward", "audit_id"):
            assert PHI_LOCAL_NAME_REGEX.match(name) is None, name


class TestCommandInputModels:
    """Pydantic v2 models are frozen + ``extra='forbid'`` + validated."""

    def test_ingest_input_frozen(self) -> None:
        m = IngestCommandInput(input_csv=Path("/x.csv"))
        with pytest.raises(ValidationError):
            m.input_csv = Path("/y.csv")  # type: ignore[misc]

    def test_ingest_input_rejects_unknown_kwarg(self) -> None:
        with pytest.raises(ValidationError):
            IngestCommandInput(input_csv=Path("/x.csv"), unknown=1)  # type: ignore[call-arg]

    def test_audit_input_xor_neither(self) -> None:
        with pytest.raises(ValidationError):
            AuditCommandInput()

    def test_audit_input_xor_both(self) -> None:
        with pytest.raises(ValidationError):
            AuditCommandInput(input_csv=Path("/x.csv"), run_id="abc123")

    def test_audit_input_only_input(self) -> None:
        m = AuditCommandInput(input_csv=Path("/x.csv"))
        assert m.input_csv == Path("/x.csv")
        assert m.run_id is None
        assert m.force is False

    def test_audit_input_only_run_id(self) -> None:
        m = AuditCommandInput(run_id="abc123def4567890")
        assert m.input_csv is None
        assert m.run_id == "abc123def4567890"

    def test_evaluate_input_requires_nonempty_run_id(self) -> None:
        with pytest.raises(ValidationError):
            EvaluateCommandInput(run_id="")

    def test_report_input_default_format_html(self) -> None:
        m = ReportCommandInput(run_id="abc")
        assert m.format == "html"

    def test_report_input_rejects_unknown_format(self) -> None:
        with pytest.raises(ValidationError):
            ReportCommandInput(run_id="abc", format="docx")  # type: ignore[arg-type]

    def test_serve_dashboard_port_range_low(self) -> None:
        with pytest.raises(ValidationError):
            ServeDashboardInput(port=0)

    def test_serve_dashboard_port_range_high(self) -> None:
        with pytest.raises(ValidationError):
            ServeDashboardInput(port=70_000)

    def test_sentinel_input_requires_cadence(self) -> None:
        with pytest.raises(ValidationError):
            SentinelCommandInput()  # type: ignore[call-arg]


class TestPhiSurface:
    """Spot-check that the PHI regex set matches the most common
    in-the-wild PHI shapes the issue body calls out."""

    def test_hn_digit_run_matches(self) -> None:
        """An 8-digit HN must be matched by at least one PHI regex."""
        assert any(r.search("HN 12345678 admitted today") for r in PHI_REGEXES)

    def test_western_honorific_name_matches(self) -> None:
        assert any(r.search("Mr. Smith was transferred") for r in PHI_REGEXES)

    def test_thai_honorific_matches(self) -> None:
        assert any(r.search("นาย สมชาย แสงทอง") for r in PHI_REGEXES)

    def test_thai_honorific_redacts_full_name_not_just_honorific(self) -> None:
        """Regression guard for PR-56 codex P1 #1: the Thai-honorific
        pattern must consume the trailing given+family name so
        ``pattern.sub('<REDACTED:phi>', "นายสมชาย ใจดี")`` collapses to
        ``'<REDACTED:phi>'`` — not ``'<REDACTED:phi>สมชาย ใจดี'`` which
        would leak the patient's name into the operator log."""
        from bba.cli.phi_scrubber import _redact_phi_in_string

        for original in (
            "นายสมชาย ใจดี",
            "นาย สมชาย แสงทอง",
            "นางสาวสุดา ปัญญา",
            "เด็กชายภูมิ ใจกล้า",
            "นาย Somchai Saengthong",
        ):
            scrubbed = _redact_phi_in_string(original)
            assert "<REDACTED" in scrubbed, (original, scrubbed)
            # No Thai-script name tokens survive the redaction.
            assert not any(
                ord(ch) in range(0x0E00, 0x0E7F) for ch in scrubbed
            ), (original, scrubbed)
            # No Latin name token survives either.
            for leak in ("Somchai", "Saengthong"):
                assert leak not in scrubbed, (original, scrubbed)

    def test_innocuous_strings_unmatched(self) -> None:
        """Sanity guard: non-PHI text must not trigger redaction."""
        for s in ("the patient was discharged", "ward 7", "transfusion ordered"):
            assert all(r.search(s) is None for r in PHI_REGEXES), s


# =============================================================================
# AC ② — All 6 subcommands invokeable with --help
# =============================================================================


SUBCOMMANDS: tuple[str, ...] = (
    "ingest",
    "audit",
    "evaluate",
    "report",
    "serve-dashboard",
    "sentinel",
)


class TestHelpListsAllSubcommands:
    """``bba --help`` must list every one of the six subcommands."""

    def test_root_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0, result.output

    @pytest.mark.parametrize("sub", SUBCOMMANDS)
    def test_root_help_mentions_each_subcommand(
        self, runner: CliRunner, sub: str
    ) -> None:
        """Scope the substring search to the ``Commands:`` section of
        click's help output — otherwise the docstring "Blood Audit"
        makes ``"audit"`` a spurious substring match."""
        result = runner.invoke(cli, ["--help"])
        commands_section = (
            result.output.split("Commands:", 1)[1]
            if "Commands:" in result.output
            else ""
        )
        assert sub in commands_section, (
            f"--help missing '{sub}' in Commands: section; "
            f"full output:\n{result.output}"
        )


class TestSubcommandHelpExitsZero:
    """Every subcommand's own ``--help`` page must render."""

    @pytest.mark.parametrize("sub", SUBCOMMANDS)
    def test_subcommand_help(self, runner: CliRunner, sub: str) -> None:
        result = runner.invoke(cli, [sub, "--help"])
        assert result.exit_code == 0, (
            f"`bba {sub} --help` returned exit_code "
            f"{result.exit_code}; output:\n{result.output}"
        )


class TestAuditHelpExposesForceFlag:
    """``bba audit --help`` must document ``--force``, ``--input``, and
    ``--run-id`` — the three options the issue body names."""

    @pytest.mark.parametrize("flag", ["--input", "--run-id", "--force"])
    def test_audit_help_documents_flag(
        self, runner: CliRunner, flag: str
    ) -> None:
        result = runner.invoke(cli, ["audit", "--help"])
        assert result.exit_code == 0, result.output
        assert flag in result.output, (
            f"`bba audit --help` missing '{flag}'; got:\n{result.output}"
        )


# =============================================================================
# AC ③ — Run-level idempotency (run_id recipe + no-op contract)
# =============================================================================


class TestRunIdRecipe:
    """``run_id = sha256(input_csv_hash + schema_fp + code_version)[:16]``."""

    def test_run_id_is_16_char_hex(self, fake_csv: Path) -> None:
        rid = compute_run_id(
            input_csv=fake_csv,
            schema_fingerprint=SchemaFingerprint("schema-fp-v1"),
            code_version_str=CodeVersion("0.1.0"),
        )
        assert len(rid) == RUN_ID_LENGTH == 16
        assert all(c in "0123456789abcdef" for c in rid), rid

    def test_run_id_matches_spec_formula(self, fake_csv: Path) -> None:
        """The CLI's run_id is the 16-char prefix of
        :class:`bba.ingest.RunIdentity` computed over every HOSxP-named
        CSV in ``input_csv.parent``. ``fake_csv`` is named ``BDVST.csv``
        and is the only HOSxP-named file in its parent dir, so the
        expected identity uses a one-entry ``per_file_hashes`` map."""
        from bba.ingest import RunIdentity, content_hash

        per_file_hashes = {"BDVST": content_hash(fake_csv)}
        expected_full = RunIdentity.from_inputs(
            per_file_hashes,  # type: ignore[arg-type]
            "schema-fp-v1",
            "0.1.0",
        ).run_id
        actual = compute_run_id(
            input_csv=fake_csv,
            schema_fingerprint=SchemaFingerprint("schema-fp-v1"),
            code_version_str=CodeVersion("0.1.0"),
        )
        assert actual == expected_full[:RUN_ID_LENGTH]


class TestRunIdDeterminism:
    """Same triple in → same run_id out, byte for byte."""

    def test_same_inputs_same_id(self, fake_csv: Path) -> None:
        a = compute_run_id(
            input_csv=fake_csv,
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        b = compute_run_id(
            input_csv=fake_csv,
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        assert a == b

    def test_identical_bundles_in_different_directories_same_id(
        self, tmp_path: Path
    ) -> None:
        """Two parent directories with the same HOSxP-named bundle
        contents produce the same run_id. The recipe is content-based
        (per-table file hash), not path-based."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        for d in (dir_a, dir_b):
            d.mkdir()
            (d / "BDVST.csv").write_bytes(b"HN,AN\n1,2\n")
        run_a = compute_run_id(
            input_csv=dir_a / "BDVST.csv",
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        run_b = compute_run_id(
            input_csv=dir_b / "BDVST.csv",
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        assert run_a == run_b


class TestRunIdSensitivity:
    """Changing any component of the triple must change the run_id."""

    def test_different_input_bytes_different_id(
        self, tmp_path: Path
    ) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "BDVST.csv").write_bytes(b"a,b\n1,2\n")
        (dir_b / "BDVST.csv").write_bytes(b"a,b\n1,3\n")
        a = compute_run_id(
            input_csv=dir_a / "BDVST.csv",
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        b = compute_run_id(
            input_csv=dir_b / "BDVST.csv",
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        assert a != b

    def test_sibling_table_edit_changes_run_id(
        self, tmp_path: Path
    ) -> None:
        """Regression for PR-56 codex P1 #2: editing a sibling HOSxP
        table (not the one passed via ``--input``) must change the
        run_id. Otherwise ``bba audit`` would treat the bundle as
        already-audited and no-op against stale outputs."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "BDVST.csv").write_bytes(b"HN,AN\n1,2\n")
        (bundle / "Diagnosis.csv").write_bytes(b"HN,AN\n1,A01\n")
        before = compute_run_id(
            input_csv=bundle / "BDVST.csv",
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        # The user only edits Diagnosis.csv — BDVST.csv stays the same.
        (bundle / "Diagnosis.csv").write_bytes(b"HN,AN\n1,B02\n")
        after = compute_run_id(
            input_csv=bundle / "BDVST.csv",
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        assert before != after, (
            "sibling-table edit must invalidate run_id; otherwise the "
            "idempotency guard reuses stale audit results"
        )

    def test_empty_bundle_raises_cli_error(self, tmp_path: Path) -> None:
        """A directory with no HOSxP-named CSVs cannot anchor an audit
        run; ``compute_run_id`` raises early rather than persist an
        idempotency marker for an unanchored bundle."""
        non_hosxp = tmp_path / "unknown.csv"
        non_hosxp.write_bytes(b"a,b\n1,2\n")
        with pytest.raises(CliError, match="no HOSxP-named CSV"):
            compute_run_id(
                input_csv=non_hosxp,
                schema_fingerprint=SchemaFingerprint("x"),
                code_version_str=CodeVersion("0.1.0"),
            )

    def test_different_schema_different_id(self, fake_csv: Path) -> None:
        a = compute_run_id(
            input_csv=fake_csv,
            schema_fingerprint=SchemaFingerprint("v1-fp"),
            code_version_str=CodeVersion("0.1.0"),
        )
        b = compute_run_id(
            input_csv=fake_csv,
            schema_fingerprint=SchemaFingerprint("v2-fp"),
            code_version_str=CodeVersion("0.1.0"),
        )
        assert a != b

    def test_different_version_different_id(self, fake_csv: Path) -> None:
        a = compute_run_id(
            input_csv=fake_csv,
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        b = compute_run_id(
            input_csv=fake_csv,
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.2.0"),
        )
        assert a != b

    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=25,
    )
    @given(payload=st.binary(min_size=1, max_size=256))
    def test_property_bytewise_identical_bundles_same_id(
        self, tmp_path: Path, payload: bytes
    ) -> None:
        """Property: two HOSxP-named bundles with identical bytes
        produce identical run_ids regardless of parent path."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        for d in (dir_a, dir_b):
            d.mkdir(exist_ok=True)
            (d / "BDVST.csv").write_bytes(payload)
        r1 = compute_run_id(
            input_csv=dir_a / "BDVST.csv",
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        r2 = compute_run_id(
            input_csv=dir_b / "BDVST.csv",
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        assert r1 == r2


class TestCodeVersionFromMetadata:
    """``code_version()`` must single-source from ``importlib.metadata``."""

    def test_returns_pep440_string(self) -> None:
        v = code_version()
        assert isinstance(v, str)
        assert re.match(r"^\d+\.\d+(\.\d+)?", v), v

    def test_matches_pyproject_version(self) -> None:
        """The version returned must match the wheel metadata so a
        manually-edited ``__version__`` cannot drift from
        ``pyproject.toml``."""
        from importlib.metadata import version as _pkg_version
        assert code_version() == _pkg_version("blood-audit")


class TestAuditIdempotencyNoop:
    """``bba audit`` twice on the same input is a no-op the second time."""

    def test_second_invocation_does_not_invoke_pipeline(
        self,
        runner: CliRunner,
        fake_csv: Path,
        stub_store: MagicMock,
    ) -> None:
        # First call: run not complete; pipeline runs; store marks complete.
        # Second call: run_complete → True; pipeline NOT invoked.
        stub_store.run_complete.side_effect = [False, True]
        stub_store.run_count.return_value = 1
        stub_store.audit_log_entries.return_value = ()

        with patch(
            "bba.cli.main._get_audit_run_store", return_value=stub_store
        ), patch(
            "bba.cli.main._run_audit_pipeline", return_value=None
        ) as fake_pipeline:
            r1 = runner.invoke(cli, ["audit", "--input", str(fake_csv)])
            r2 = runner.invoke(cli, ["audit", "--input", str(fake_csv)])

        assert r1.exit_code == 0, r1.output
        assert r2.exit_code == 0, r2.output
        assert fake_pipeline.call_count == 1, (
            f"pipeline invoked {fake_pipeline.call_count} times; "
            "second --input call must be a no-op"
        )

    def test_run_count_remains_one_after_two_invocations(
        self,
        runner: CliRunner,
        fake_csv: Path,
        stub_store: MagicMock,
    ) -> None:
        """``audit_store.run_count(run_id) == 1`` after two identical
        invocations is the issue's explicit acceptance assertion."""
        stub_store.run_complete.side_effect = [False, True]
        stub_store.run_count.return_value = 1

        with patch(
            "bba.cli.main._get_audit_run_store", return_value=stub_store
        ), patch("bba.cli.main._run_audit_pipeline", return_value=None):
            runner.invoke(cli, ["audit", "--input", str(fake_csv)])
            runner.invoke(cli, ["audit", "--input", str(fake_csv)])

        # The run_id used for the lookup must be the same both times.
        recorded_ids = {
            call.args[0] for call in stub_store.run_count.call_args_list
        }
        assert len(recorded_ids) == 1, (
            f"expected one distinct run_id queried; got {recorded_ids}"
        )
        assert stub_store.run_count.return_value == 1


# =============================================================================
# AC ④ — --force overrides idempotency and records compliance entries
# =============================================================================


class TestAuditForceOverride:
    """``--force`` runs the pipeline even when the run is already
    marked complete."""

    def test_force_invokes_pipeline_even_when_complete(
        self,
        runner: CliRunner,
        fake_csv: Path,
        stub_store: MagicMock,
    ) -> None:
        stub_store.run_complete.return_value = True

        with patch(
            "bba.cli.main._get_audit_run_store", return_value=stub_store
        ), patch(
            "bba.cli.main._run_audit_pipeline", return_value=None
        ) as fake_pipeline:
            result = runner.invoke(
                cli, ["audit", "--input", str(fake_csv), "--force"]
            )

        assert result.exit_code == 0, result.output
        assert fake_pipeline.call_count == 1


class TestAuditForceLogsOverridePerInvocation:
    """Each ``--force`` invocation must write one ``idempotency_override``
    row to ``audit_log`` for compliance."""

    def test_two_force_invocations_log_two_overrides(
        self,
        runner: CliRunner,
        fake_csv: Path,
        stub_store: MagicMock,
    ) -> None:
        stub_store.run_complete.return_value = True
        recorded: list[tuple[str, str]] = []

        def _record(run_id: str, /, *, reason: str) -> None:
            recorded.append((run_id, reason))

        stub_store.record_idempotency_override.side_effect = _record

        with patch(
            "bba.cli.main._get_audit_run_store", return_value=stub_store
        ), patch("bba.cli.main._run_audit_pipeline", return_value=None):
            r1 = runner.invoke(
                cli, ["audit", "--input", str(fake_csv), "--force"]
            )
            r2 = runner.invoke(
                cli, ["audit", "--input", str(fake_csv), "--force"]
            )

        assert r1.exit_code == 0, r1.output
        assert r2.exit_code == 0, r2.output
        assert len(recorded) == 2, (
            f"expected 2 audit_log idempotency_override rows; got {recorded}"
        )


# =============================================================================
# AC ⑤ — PHI scrubbing in exception traces
# =============================================================================


class TestPhiScrubbingInTraceback:
    """An uncaught exception with ``bundle`` in scope must produce a
    scrubbed traceback in the structured log surface — the explicit
    acceptance criterion in the issue body."""

    def test_bundle_locals_scrubbed_in_caplog(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        install_excepthook()

        def fail() -> None:
            bundle = {  # noqa: F841 — captured by the excepthook frame walk
                "hn": "12345678",
                "note": "Mr. Smith was admitted to Bangkok General",
            }
            patient = "Somchai Saengthong"  # noqa: F841
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR):
            try:
                fail()
            except RuntimeError:
                # Invoke the hook directly — pytest installs its own
                # excepthook that would otherwise swallow ours.
                exc_type, exc_value, exc_tb = sys.exc_info()
                assert exc_type is not None and exc_value is not None
                sys.excepthook(exc_type, exc_value, exc_tb)

        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "<REDACTED" in joined, (
            f"expected '<REDACTED' marker in scrubbed log; got:\n{joined}"
        )
        for leak in ("12345678", "Smith", "Somchai", "Saengthong"):
            assert leak not in joined, (
                f"PHI substring '{leak}' leaked into traceback log:\n{joined}"
            )


class TestScrubTracebackRedactsBundleLocals:
    """``scrub_traceback`` redacts any frame-local whose name matches
    :data:`PHI_LOCAL_NAME_REGEX`, regardless of value type."""

    def test_redacts_dict_named_bundle(self) -> None:
        def boom() -> None:
            bundle = {"hn": "00012345", "note": "x"}  # noqa: F841
            raise RuntimeError("boom")

        try:
            boom()
        except RuntimeError as exc:
            scrubbed = scrub_traceback(type(exc), exc, exc.__traceback__)

        assert "00012345" not in scrubbed, scrubbed
        assert "<REDACTED" in scrubbed, scrubbed

    def test_redacts_str_named_patient(self) -> None:
        def boom() -> None:
            patient = "นาย สมชาย"  # noqa: F841
            raise RuntimeError("boom")

        try:
            boom()
        except RuntimeError as exc:
            scrubbed = scrub_traceback(type(exc), exc, exc.__traceback__)

        assert "สมชาย" not in scrubbed
        assert "<REDACTED" in scrubbed


class TestScrubTracebackRedactsHnDigitRuns:
    """Even when a PHI value lives in a *non-PHI-named* local, the regex
    sweep over string values must still redact HN-shaped digit runs."""

    def test_digit_run_in_random_local(self) -> None:
        def boom() -> None:
            some_id = "00012345"  # noqa: F841 — name does not match PHI_LOCAL_NAME_REGEX
            raise RuntimeError("boom")

        try:
            boom()
        except RuntimeError as exc:
            scrubbed = scrub_traceback(type(exc), exc, exc.__traceback__)

        assert "00012345" not in scrubbed, scrubbed


class TestScrubTracebackRedactsThaiHonorifics:
    """Thai honorifics in any string value must be redacted regardless
    of the surrounding local name."""

    def test_thai_honorific_in_message(self) -> None:
        def boom() -> None:
            msg = "นาย Somchai admitted"  # noqa: F841
            raise RuntimeError("boom")

        try:
            boom()
        except RuntimeError as exc:
            scrubbed = scrub_traceback(type(exc), exc, exc.__traceback__)

        assert "นาย" not in scrubbed, scrubbed


# =============================================================================
# Boundary regression — what the CLI MUST NOT do
# =============================================================================


class TestNoForbiddenSubcommands:
    """The MUST-NOT subcommands stay out of the CLI surface."""

    @pytest.mark.parametrize("sub", ["migrate", "init-db", "serve-api"])
    def test_forbidden_subcommand_is_not_registered(
        self, runner: CliRunner, sub: str
    ) -> None:
        result = runner.invoke(cli, ["--help"])
        assert sub not in result.output, (
            f"forbidden subcommand '{sub}' leaked into bba CLI; "
            "see issue #29 'MUST NOT do' list"
        )

    def test_audit_has_no_dry_run_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["audit", "--help"])
        # If `audit` is not yet wired the result is a UsageError; we
        # still assert the help text never grows a --dry-run flag.
        assert "--dry-run" not in result.output


class TestNoRetryBackoffInCli:
    """CLI fails loud — no retry / backoff primitives belong here. They
    live in :mod:`bba.llm_client` (issue #22)."""

    def test_no_retry_or_backoff_imports(self) -> None:
        for source in CLI_PACKAGE_DIR.rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            for forbidden in ("import tenacity", "import backoff", "retry_call("):
                assert forbidden not in text, (
                    f"forbidden retry primitive '{forbidden}' found in {source}"
                )


class TestEnvVarSurfaceIsTight:
    """The CLI may only read ``BBA_DATA_DIR``, ``BBA_DB_URL``,
    ``ANTHROPIC_API_KEY`` — every other ``os.environ`` / ``os.getenv``
    reference is forbidden."""

    ALLOWED: frozenset[str] = frozenset(
        {"BBA_DATA_DIR", "BBA_DB_URL", "ANTHROPIC_API_KEY"}
    )

    def test_no_other_env_vars_referenced_in_cli_package(self) -> None:
        # Conservative pattern: any os.environ['X'] or os.getenv('X', ...) literal.
        env_lookup = re.compile(
            r"""(?:os\.environ\[\s*['"]|os\.getenv\(\s*['"])([A-Z_][A-Z0-9_]+)""",
        )
        leaked: dict[Path, set[str]] = {}
        for source in CLI_PACKAGE_DIR.rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            refs = set(env_lookup.findall(text)) - self.ALLOWED
            if refs:
                leaked[source.relative_to(REPO_ROOT)] = refs
        assert leaked == {}, f"unexpected env-var reads in bba.cli: {leaked}"


class TestMainModuleIsThin:
    """The cardinal rule: subcommands are ≤20-line wrappers, never the
    home of business logic. This regression guards the body length of
    each click command function."""

    def test_each_subcommand_body_is_thin(self) -> None:
        """Each click callback's *body* (post-``def``, post-docstring)
        stays under the 20-line thin-glue budget. Decorators and
        docstrings are deliberately excluded — they don't carry business
        logic; long ``@click.option(...)`` blocks are fine."""
        import ast
        import inspect

        import bba.cli.main as main_module

        max_body_statements = 20
        offenders: dict[str, int] = {}
        for fname in (
            "bba_ingest",
            "bba_audit",
            "bba_evaluate",
            "bba_report",
            "bba_serve_dashboard",
            "bba_sentinel",
        ):
            fn = getattr(main_module, fname).callback
            src = inspect.getsource(fn)
            tree = ast.parse(src.lstrip())
            func_def = tree.body[0]
            assert isinstance(func_def, ast.FunctionDef), (
                f"expected FunctionDef for {fname}, got {type(func_def)}"
            )
            body = func_def.body
            # Drop a leading docstring expression if present.
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            if len(body) > max_body_statements:
                offenders[fname] = len(body)
        assert offenders == {}, (
            f"subcommands exceeded {max_body_statements}-statement "
            f"thin-glue budget: {offenders}. Push the logic down to the "
            "underlying module."
        )


# =============================================================================
# Smoke / regression — re-import safety
# =============================================================================


class TestReimportSafety:
    """Re-importing the CLI module is idempotent and does not double-
    register any subcommand on the click group."""

    def test_reimport_does_not_duplicate_subcommands(self) -> None:
        before = dict(cli.commands)
        importlib.reload(importlib.import_module("bba.cli.main"))
        from bba.cli.main import cli as cli_after_reload

        after = dict(cli_after_reload.commands)
        assert set(after) == set(before)


# =============================================================================
# Subcommand behaviour — thin-glue wiring + loud-failure contracts
# =============================================================================


class TestBbaIngestWiring:
    """``bba ingest`` delegates to :func:`bba.ingest.ingest` with an
    :class:`~bba.ingest.IngestConfig` built from BBA_DATA_DIR + the
    package version."""

    def test_ingest_calls_underlying_facade_with_config(
        self,
        runner: CliRunner,
        fake_csv: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BBA_DATA_DIR", str(tmp_path))
        with patch("bba.cli.main.ingest") as mock_ingest:
            result = runner.invoke(cli, ["ingest", str(fake_csv)])
        assert result.exit_code == 0, result.output
        mock_ingest.assert_called_once()
        (config,) = mock_ingest.call_args.args
        assert config.input_dir == fake_csv.parent
        assert config.output_dir == tmp_path / "ingest" / "v1"
        assert isinstance(config.code_version, str)

    def test_ingest_without_data_dir_env_fails_loud(
        self,
        runner: CliRunner,
        fake_csv: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("BBA_DATA_DIR", raising=False)
        result = runner.invoke(cli, ["ingest", str(fake_csv)])
        assert result.exit_code != 0
        assert "BBA_DATA_DIR" in str(result.exception)


class TestUnwiredSubcommandsFailLoud:
    """The four subcommands whose underlying module does not yet expose
    a single-call facade fail loud with :class:`CliError` rather than
    fabricating defaults. Verifies the contract phrasing names the
    integration seam."""

    @pytest.mark.parametrize(
        ("argv", "expected_seam"),
        [
            (["evaluate", "--run-id", "abc"], "bba.cli.main.bba_evaluate"),
            (
                ["report", "--run-id", "abc", "--format", "pdf"],
                "bba.cli.main.bba_report",
            ),
            (
                ["serve-dashboard", "--port", "8765"],
                "bba.cli.main.bba_serve_dashboard",
            ),
            (["sentinel", "--weekly"], "bba.cli.main.bba_sentinel"),
        ],
    )
    def test_unwired_subcommand_raises_cli_error(
        self,
        runner: CliRunner,
        argv: list[str],
        expected_seam: str,
    ) -> None:
        result = runner.invoke(cli, argv)
        assert result.exit_code != 0
        assert isinstance(result.exception, CliError), result.exception
        assert expected_seam in str(result.exception), result.exception


class TestSentinelRequiresCadenceFlag:
    """``bba sentinel`` must reject invocations without exactly one of
    ``--weekly`` / ``--quarterly``. The spec says "exactly one"; click's
    ``flag_value`` shortcut silently last-wins, so the implementation
    uses two booleans + an explicit XOR check."""

    def test_no_cadence_is_a_usage_error(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["sentinel"])
        assert result.exit_code != 0
        assert (
            "weekly" in result.output.lower()
            or "quarterly" in result.output.lower()
            or "cadence" in result.output.lower()
        ), result.output

    def test_both_weekly_and_quarterly_is_a_usage_error(
        self, runner: CliRunner
    ) -> None:
        """The 'exactly one' contract rejects both being supplied —
        otherwise click's last-wins behaviour would silently pick one."""
        result = runner.invoke(
            cli, ["sentinel", "--weekly", "--quarterly"]
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower(), result.output


class TestGetAuditRunStoreEnvGate:
    """The production resolver ``_get_audit_run_store`` returns a
    :class:`FileBackedAuditRunStore` rooted at ``$BBA_DATA_DIR``."""

    def test_missing_data_dir_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import bba.cli.main as main_module

        monkeypatch.delenv("BBA_DATA_DIR", raising=False)
        with pytest.raises(CliError, match="BBA_DATA_DIR"):
            main_module._get_audit_run_store()

    def test_returns_file_backed_store(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        import bba.cli.main as main_module
        from bba.cli.audit_run_store import FileBackedAuditRunStore

        monkeypatch.setenv("BBA_DATA_DIR", str(tmp_path))
        store = main_module._get_audit_run_store()
        assert isinstance(store, FileBackedAuditRunStore)
        # Newly-resolved store reports the run as not complete and count zero.
        assert store.run_complete("nonexistent-run-id") is False
        assert store.run_count("nonexistent-run-id") == 0


class TestFileBackedAuditRunStore:
    """End-to-end behaviour of the file-backed :class:`AuditRunStore`:
    marker-creation, row-tracking, and the JSONL audit_log."""

    def test_marker_lifecycle(self, tmp_path: Path) -> None:
        from bba.cli.audit_run_store import FileBackedAuditRunStore

        store = FileBackedAuditRunStore(tmp_path)
        run_id = "abc123def4567890"
        assert store.run_complete(run_id) is False
        store.mark_run_complete(run_id)
        assert store.run_complete(run_id) is True

    def test_row_count_tracks_record_row_calls(self, tmp_path: Path) -> None:
        from bba.cli.audit_run_store import FileBackedAuditRunStore

        store = FileBackedAuditRunStore(tmp_path)
        run_id = "abc123def4567890"
        assert store.run_count(run_id) == 0
        store.record_row(run_id, "audit-0001")
        store.record_row(run_id, "audit-0002")
        assert store.run_count(run_id) == 2

    def test_idempotency_override_appends_jsonl_per_call(
        self, tmp_path: Path
    ) -> None:
        from bba.cli.audit_run_store import FileBackedAuditRunStore

        store = FileBackedAuditRunStore(tmp_path)
        run_id = "abc123def4567890"
        store.record_idempotency_override(run_id, reason="first --force")
        store.record_idempotency_override(run_id, reason="second --force")

        entries = store.audit_log_entries(run_id)
        assert len(entries) == 2
        for entry in entries:
            assert entry["idempotency_override"] is True
            assert entry["run_id"] == run_id
            assert isinstance(entry["ts"], str)
        assert entries[0]["reason"] == "first --force"
        assert entries[1]["reason"] == "second --force"

    def test_audit_log_filters_by_run_id(self, tmp_path: Path) -> None:
        from bba.cli.audit_run_store import FileBackedAuditRunStore

        store = FileBackedAuditRunStore(tmp_path)
        store.record_idempotency_override("run-A", reason="a")
        store.record_idempotency_override("run-B", reason="b")
        store.record_idempotency_override("run-A", reason="a2")
        assert tuple(e["reason"] for e in store.audit_log_entries("run-A")) == (
            "a",
            "a2",
        )
        assert tuple(e["reason"] for e in store.audit_log_entries("run-B")) == (
            "b",
        )

    def test_record_idempotency_override_handles_short_writes(
        self, tmp_path: Path
    ) -> None:
        """``os.write`` may return a short count under filesystem
        pressure; the implementation loops until the full payload is
        written. Simulate a short write that returns the buffer length
        in two slices and assert the final on-disk content matches."""
        from bba.cli.audit_run_store import FileBackedAuditRunStore

        store = FileBackedAuditRunStore(tmp_path)
        run_id = "short-write-run"
        real_os_write = os.write
        call_count = {"n": 0}

        def short_write(fd: int, data: bytes) -> int:
            # First call writes 10 bytes; subsequent calls write the rest.
            call_count["n"] += 1
            if call_count["n"] == 1 and len(data) > 10:
                return real_os_write(fd, data[:10])
            return real_os_write(fd, data)

        with patch("bba.cli.audit_run_store.os.write", side_effect=short_write):
            store.record_idempotency_override(
                run_id, reason="short-write test"
            )

        # Two writes total — the first short, the second covers the rest.
        assert call_count["n"] >= 2
        entries = store.audit_log_entries(run_id)
        assert len(entries) == 1
        assert entries[0]["reason"] == "short-write test"

    def test_record_idempotency_override_zero_byte_write_raises(
        self, tmp_path: Path
    ) -> None:
        """A 0-byte ``os.write`` is a write(2) contract violation —
        raise OSError rather than spinning forever."""
        from bba.cli.audit_run_store import FileBackedAuditRunStore

        store = FileBackedAuditRunStore(tmp_path)
        with patch("bba.cli.audit_run_store.os.write", return_value=0):
            with pytest.raises(OSError, match="zero-byte write"):
                store.record_idempotency_override(
                    "any-run", reason="zero-write test"
                )

    def test_empty_audit_log_returns_empty_tuple(self, tmp_path: Path) -> None:
        from bba.cli.audit_run_store import FileBackedAuditRunStore

        store = FileBackedAuditRunStore(tmp_path)
        assert store.audit_log_entries("any-id") == ()

    def test_concurrent_idempotency_overrides_do_not_corrupt_jsonl(
        self, tmp_path: Path
    ) -> None:
        """Under thread-level concurrency, every appended line stays a
        valid JSON object. The implementation uses
        ``os.write(fd_with_O_APPEND, ...)`` which is POSIX-atomic up to
        ``PIPE_BUF`` — each ~200-byte entry is well under that, so
        concurrent ``--force`` invocations cannot interleave bytes."""
        import json as _json
        import threading

        from bba.cli.audit_run_store import FileBackedAuditRunStore

        store = FileBackedAuditRunStore(tmp_path)
        run_id = "concurrent-run"
        n_threads = 8
        n_per_thread = 25

        def worker(thread_idx: int) -> None:
            for i in range(n_per_thread):
                store.record_idempotency_override(
                    run_id, reason=f"thread-{thread_idx}-call-{i}"
                )

        threads = [
            threading.Thread(target=worker, args=(t,))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every line must parse as JSON; total line count must equal
        # the total number of override calls.
        log_path = tmp_path / "audit_runs" / "audit_log.jsonl"
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == n_threads * n_per_thread
        for line in lines:
            entry = _json.loads(line)
            assert entry["idempotency_override"] is True
            assert entry["run_id"] == run_id

    def test_acquire_run_lock_creates_and_releases_lockfile(
        self, tmp_path: Path
    ) -> None:
        """The context manager creates the lockfile on entry and
        leaves it in place on exit (an empty lockfile is the normal
        steady state once a run has been locked at least once)."""
        from bba.cli.audit_run_store import FileBackedAuditRunStore

        store = FileBackedAuditRunStore(tmp_path)
        run_id = "lock-test"
        lock_path = tmp_path / "audit_runs" / f"run_{run_id}.lock"
        assert not lock_path.exists()
        with store.acquire_run_lock(run_id):
            assert lock_path.is_file()
        # File persists post-exit; the flock is released, not the file.
        assert lock_path.is_file()

    def test_acquire_run_lock_blocks_across_processes(
        self, tmp_path: Path
    ) -> None:
        """``fcntl.flock`` is advisory per-open-file-description; the
        guarantee that matters in production is *cross-process*
        exclusion. Spawn a subprocess that holds the lock, then assert
        that a second non-blocking attempt from this process fails."""
        import fcntl
        import os
        import subprocess
        import sys
        import textwrap

        from bba.cli.audit_run_store import FileBackedAuditRunStore

        # Make the lockfile via a first acquire in this process so the
        # subprocess does not race to create it.
        store = FileBackedAuditRunStore(tmp_path)
        run_id = "x-proc-lock"
        with store.acquire_run_lock(run_id):
            pass
        lock_path = tmp_path / "audit_runs" / f"run_{run_id}.lock"
        assert lock_path.is_file()

        # Subprocess that holds an exclusive flock for 2 s.
        holder_script = textwrap.dedent(
            f"""
            import fcntl, os, time
            fd = os.open({str(lock_path)!r}, os.O_WRONLY)
            fcntl.flock(fd, fcntl.LOCK_EX)
            print("locked", flush=True)
            time.sleep(2.0)
            """
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", holder_script],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            # Wait until the subprocess says it holds the lock.
            assert proc.stdout is not None
            line = proc.stdout.readline()
            assert line.strip() == "locked", line

            # Non-blocking acquire from THIS process must fail with
            # BlockingIOError because the subprocess holds the lock.
            fd = os.open(lock_path, os.O_WRONLY)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_audit_log_skips_blank_lines(self, tmp_path: Path) -> None:
        """A blank or whitespace-only line in audit_log.jsonl (e.g. a
        trailing newline after a manual edit) must be skipped without
        raising ``json.JSONDecodeError``."""
        from bba.cli.audit_run_store import FileBackedAuditRunStore

        store = FileBackedAuditRunStore(tmp_path)
        store.record_idempotency_override("run-A", reason="first")
        log_path = tmp_path / "audit_runs" / "audit_log.jsonl"
        # Append a blank line manually — mirrors the trailing-newline case.
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n   \n")
        store.record_idempotency_override("run-A", reason="second")
        entries = store.audit_log_entries("run-A")
        assert tuple(e["reason"] for e in entries) == ("first", "second")


class TestAuditWithoutInputCannotReachPipeline:
    """``bba audit --run-id X`` where the store says run is not complete
    cannot proceed: the pipeline needs the input CSV. The CLI raises
    :class:`click.UsageError` rather than silently no-op'ing."""

    def test_run_id_only_with_incomplete_store_is_usage_error(
        self,
        runner: CliRunner,
        stub_store: MagicMock,
    ) -> None:
        stub_store.run_complete.return_value = False
        with patch(
            "bba.cli.main._get_audit_run_store", return_value=stub_store
        ):
            result = runner.invoke(cli, ["audit", "--run-id", "abc123"])
        assert result.exit_code != 0
        # click.UsageError flows through; message names the constraint.
        assert (
            "without --input" in result.output
            or "without --input" in str(result.exception)
        ), (result.output, result.exception)


class TestPhiScrubberEdgeCases:
    """Defensive branches in :mod:`bba.cli.phi_scrubber` — non-Sized
    objects, repr() that raises, dunder skipping, and the
    faulthandler_sidecar argument path."""

    def test_redact_by_name_handles_objects_without_len(self) -> None:
        """A PHI-named local whose value has no ``__len__`` (e.g. an
        int) still redacts to a marker; the marker length is ``-1``
        rather than throwing :class:`TypeError`."""

        def boom() -> None:
            patient_age = 42  # noqa: F841 — name matches PHI_LOCAL_NAME_REGEX
            raise RuntimeError("boom")

        try:
            boom()
        except RuntimeError as exc:
            scrubbed = scrub_traceback(type(exc), exc, exc.__traceback__)

        assert "<REDACTED:type=int len=-1>" in scrubbed, scrubbed

    def test_repr_failed_branch_does_not_crash(self) -> None:
        """If a frame-local's ``repr`` itself raises, the scrubber emits
        ``<repr-failed>`` rather than letting the exception escape and
        clobbering the original traceback."""

        class _ReprBomb:
            def __repr__(self) -> str:
                raise ValueError("repr blew up")

        def boom() -> None:
            other = _ReprBomb()  # noqa: F841 — non-PHI name, hit regex sweep
            raise RuntimeError("boom")

        try:
            boom()
        except RuntimeError as exc:
            scrubbed = scrub_traceback(type(exc), exc, exc.__traceback__)

        assert "<repr-failed>" in scrubbed, scrubbed

    def test_dunder_locals_skipped(self) -> None:
        """``__doc__`` / ``__name__`` / ``__builtins__`` style bindings
        in a frame are filtered out of the scrubbed dump."""

        def boom() -> None:
            __secret__ = "should-not-appear"  # noqa: F841
            raise RuntimeError("boom")

        try:
            boom()
        except RuntimeError as exc:
            scrubbed = scrub_traceback(type(exc), exc, exc.__traceback__)

        assert "__secret__" not in scrubbed, scrubbed

    def test_install_excepthook_with_sidecar_creates_file(
        self,
        tmp_path: Path,
    ) -> None:
        """The optional ``faulthandler_sidecar`` argument creates the
        parent directory and opens the file for append."""
        sidecar = tmp_path / "sub" / "faulthandler.sidecar"
        install_excepthook(faulthandler_sidecar=sidecar)
        assert sidecar.exists()
        assert sidecar.parent.is_dir()


class TestRootCallbackInstallsExcepthook:
    """``bba <subcommand>`` must install the PHI-scrubbing excepthook
    before any subcommand body runs — otherwise the contract surface
    exists but no production CLI invocation benefits from it.

    The group callback only fires when a subcommand is dispatched, so
    these tests use ``bba ingest --help`` (which short-circuits inside
    the subcommand *after* the group callback runs) to assert the wire
    is in place."""

    def test_excepthook_installed_with_sidecar_when_data_dir_set(
        self,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BBA_DATA_DIR", str(tmp_path))
        with patch("bba.cli.main.install_excepthook") as mock_install:
            result = runner.invoke(cli, ["ingest", "--help"])
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once()
        assert mock_install.call_args.kwargs["faulthandler_sidecar"] == (
            tmp_path / "logs" / "faulthandler.sidecar"
        )

    def test_excepthook_installed_without_sidecar_when_data_dir_unset(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Excepthook installation must not depend on BBA_DATA_DIR; only
        the optional faulthandler sidecar does."""
        monkeypatch.delenv("BBA_DATA_DIR", raising=False)
        with patch("bba.cli.main.install_excepthook") as mock_install:
            result = runner.invoke(cli, ["ingest", "--help"])
        assert result.exit_code == 0, result.output
        mock_install.assert_called_once()
        assert mock_install.call_args.kwargs["faulthandler_sidecar"] is None


class TestEndToEndAuditIdempotency:
    """End-to-end integration: ``bba audit`` twice on the same input
    with the *real* :class:`FileBackedAuditRunStore` no-ops the second
    time. Only the audit-pipeline orchestrator (the LLM analysis leg)
    is mocked; the store, the marker writes, and the audit_log JSONL
    are exercised against the file system."""

    def test_second_real_invocation_is_real_noop(
        self,
        runner: CliRunner,
        fake_csv: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BBA_DATA_DIR", str(tmp_path))
        with patch(
            "bba.cli.main._run_audit_pipeline", return_value=None
        ) as fake_pipeline:
            r1 = runner.invoke(cli, ["audit", "--input", str(fake_csv)])
            r2 = runner.invoke(cli, ["audit", "--input", str(fake_csv)])
        assert r1.exit_code == 0, r1.output
        assert r2.exit_code == 0, r2.output
        assert fake_pipeline.call_count == 1, (
            "second real invocation must skip the pipeline, but it was "
            f"called {fake_pipeline.call_count} times"
        )
        # The marker must exist on disk for the third process to honor.
        run_marker_dir = tmp_path / "audit_runs"
        assert run_marker_dir.is_dir()
        markers = list(run_marker_dir.glob("run_*.complete"))
        assert len(markers) == 1, markers

    def test_force_invocations_persist_audit_log_jsonl(
        self,
        runner: CliRunner,
        fake_csv: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--force`` twice writes two real JSONL lines to
        ``audit_runs/audit_log.jsonl`` on disk — not just to a mock."""
        from bba.cli.audit_run_store import FileBackedAuditRunStore

        monkeypatch.setenv("BBA_DATA_DIR", str(tmp_path))
        with patch("bba.cli.main._run_audit_pipeline", return_value=None):
            r0 = runner.invoke(cli, ["audit", "--input", str(fake_csv)])
            r1 = runner.invoke(
                cli, ["audit", "--input", str(fake_csv), "--force"]
            )
            r2 = runner.invoke(
                cli, ["audit", "--input", str(fake_csv), "--force"]
            )
        assert r0.exit_code == 0, r0.output
        assert r1.exit_code == 0, r1.output
        assert r2.exit_code == 0, r2.output

        log_path = tmp_path / "audit_runs" / "audit_log.jsonl"
        assert log_path.is_file()
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2, lines
        # Re-open through the store to validate the schema invariants.
        post_store = FileBackedAuditRunStore(tmp_path)
        # We don't know the exact run_id; fetch via the audit_log file directly.
        import json as _json

        entries = [_json.loads(line) for line in lines]
        for entry in entries:
            assert entry["idempotency_override"] is True
            assert "ts" in entry
            assert "reason" in entry
        # Sanity: both lines share one run_id (same input twice).
        assert len({e["run_id"] for e in entries}) == 1
        # And the store reports both entries for that run_id.
        run_id = entries[0]["run_id"]
        assert len(post_store.audit_log_entries(run_id)) == 2


class TestRunAuditPipelineIngestLeg:
    """The Phase 1 ``_run_audit_pipeline`` deliverable is the ingest leg
    — calling :func:`bba.ingest.ingest` with a per-run output directory,
    then recording one row marker per ingested HOSxP table so
    :meth:`AuditRunStore.run_count` stays consistent with
    :meth:`AuditRunStore.run_complete`."""

    def test_calls_ingest_with_per_run_output_dir(
        self,
        tmp_path: Path,
        fake_csv: Path,
        stub_store: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import bba.cli.main as main_module
        from bba.ingest import IngestResult

        monkeypatch.setenv("BBA_DATA_DIR", str(tmp_path))
        run_id = "abc123def4567890"
        fake_result = IngestResult(
            run_id=run_id,
            rows_written=0,
            tables_written=("BDVST",),
            skipped_idempotent=False,
        )
        with patch(
            "bba.cli.main.ingest", return_value=fake_result
        ) as mock_ingest:
            main_module._run_audit_pipeline(
                run_id=run_id, input_csv=fake_csv, store=stub_store
            )
        mock_ingest.assert_called_once()
        (config,) = mock_ingest.call_args.args
        assert config.input_dir == fake_csv.parent
        assert config.output_dir == tmp_path / "audit" / run_id

    def test_missing_data_dir_raises_cli_error(
        self,
        tmp_path: Path,
        fake_csv: Path,
        stub_store: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import bba.cli.main as main_module

        monkeypatch.delenv("BBA_DATA_DIR", raising=False)
        with pytest.raises(CliError, match="BBA_DATA_DIR"):
            main_module._run_audit_pipeline(
                run_id="abc123def4567890",
                input_csv=fake_csv,
                store=stub_store,
            )

    def test_records_one_row_marker_per_ingested_table(
        self,
        tmp_path: Path,
        fake_csv: Path,
        stub_store: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The pipeline trusts :attr:`IngestResult.tables_written` —
        the typed return value from :func:`bba.ingest.ingest` — and
        records one ``store.record_row`` per validated table."""
        import bba.cli.main as main_module
        from bba.ingest import IngestResult

        monkeypatch.setenv("BBA_DATA_DIR", str(tmp_path))
        run_id = "abc123def4567890"
        fake_result = IngestResult(
            run_id=run_id,
            rows_written=0,
            tables_written=("BDVST", "BDTYPE", "Diagnosis"),
            skipped_idempotent=False,
        )
        with patch("bba.cli.main.ingest", return_value=fake_result):
            main_module._run_audit_pipeline(
                run_id=run_id, input_csv=fake_csv, store=stub_store
            )
        recorded_audit_ids = {
            call.args[1] for call in stub_store.record_row.call_args_list
        }
        assert recorded_audit_ids == {
            "phase1_ingest_BDVST",
            "phase1_ingest_BDTYPE",
            "phase1_ingest_Diagnosis",
        }

    def test_zero_tables_written_raises_cli_error(
        self,
        tmp_path: Path,
        fake_csv: Path,
        stub_store: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Defensive guard: if ingest's contract changes and it returns
        an empty ``tables_written`` tuple, the pipeline must fail loud
        rather than let bba_audit mark the run complete with zero rows."""
        import bba.cli.main as main_module
        from bba.ingest import IngestResult

        monkeypatch.setenv("BBA_DATA_DIR", str(tmp_path))
        empty_result = IngestResult(
            run_id="abc123def4567890",
            rows_written=0,
            tables_written=(),
            skipped_idempotent=False,
        )
        with patch("bba.cli.main.ingest", return_value=empty_result):
            with pytest.raises(CliError, match="zero ingested tables"):
                main_module._run_audit_pipeline(
                    run_id="abc123def4567890",
                    input_csv=fake_csv,
                    store=stub_store,
                )
        stub_store.record_row.assert_not_called()

    def test_real_store_run_count_matches_after_pipeline(
        self,
        runner: CliRunner,
        fake_csv: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-to-end integration against the real
        :class:`FileBackedAuditRunStore`: after a successful audit,
        ``run_complete`` is True and ``run_count`` equals the number of
        validated HOSxP tables returned by ingest. This proves the
        round-2 inconsistency (complete=True with count=0) is
        impossible — the pipeline writes the markers BEFORE bba_audit
        writes the completion marker, and the count source is the
        ingest function's typed return value (so a future ingest that
        adds parquet writes does not break this contract)."""
        from bba.cli.audit_run_store import FileBackedAuditRunStore
        from bba.ingest import IngestResult

        monkeypatch.setenv("BBA_DATA_DIR", str(tmp_path))
        fake_result = IngestResult(
            run_id="placeholder",  # overwritten by CLI; real value is computed
            rows_written=0,
            tables_written=("BDVST", "Diagnosis", "Lab"),
            skipped_idempotent=False,
        )
        with patch("bba.cli.main.ingest", return_value=fake_result):
            result = runner.invoke(cli, ["audit", "--input", str(fake_csv)])
        assert result.exit_code == 0, result.output

        markers = list((tmp_path / "audit_runs").glob("run_*.complete"))
        assert len(markers) == 1, markers
        run_id = markers[0].stem.removeprefix("run_").removesuffix(".complete")
        store = FileBackedAuditRunStore(tmp_path)
        assert store.run_complete(run_id) is True
        assert store.run_count(run_id) == 3
