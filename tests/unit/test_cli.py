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

import hashlib
import importlib
import logging
import re
import sys
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


@pytest.fixture
def fake_csv(tmp_path: Path) -> Path:
    """A small CSV that satisfies ``click.Path(exists=True)``.

    Content is irrelevant — the CLI's idempotency tests stub the audit
    pipeline. Only the *bytes* matter for run_id computation."""
    p = tmp_path / "input.csv"
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
        from bba.ingest.hashing import content_hash

        input_hash = content_hash(fake_csv)
        expected = hashlib.sha256(
            (input_hash + "schema-fp-v1" + "0.1.0").encode("utf-8")
        ).hexdigest()[:RUN_ID_LENGTH]
        actual = compute_run_id(
            input_csv=fake_csv,
            schema_fingerprint=SchemaFingerprint("schema-fp-v1"),
            code_version_str=CodeVersion("0.1.0"),
        )
        assert actual == expected


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

    def test_identical_bytes_at_different_paths_same_id(
        self, tmp_path: Path
    ) -> None:
        """The recipe is content-based, not path-based: same bytes at
        ``/tmp/a.csv`` and ``/tmp/b.csv`` produce the same run_id."""
        p1 = tmp_path / "a.csv"
        p2 = tmp_path / "b.csv"
        p1.write_bytes(b"HN,AN\n1,2\n")
        p2.write_bytes(b"HN,AN\n1,2\n")
        a = compute_run_id(
            input_csv=p1,
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        b = compute_run_id(
            input_csv=p2,
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        assert a == b


class TestRunIdSensitivity:
    """Changing any component of the triple must change the run_id."""

    def test_different_input_different_id(self, tmp_path: Path) -> None:
        a_path = tmp_path / "a.csv"
        b_path = tmp_path / "b.csv"
        a_path.write_bytes(b"a,b\n1,2\n")
        b_path.write_bytes(b"a,b\n1,3\n")
        a = compute_run_id(
            input_csv=a_path,
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        b = compute_run_id(
            input_csv=b_path,
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        assert a != b

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
    def test_property_bytewise_identical_inputs_same_id(
        self, tmp_path: Path, payload: bytes
    ) -> None:
        """Property: bytewise-identical inputs ⇒ identical run_id."""
        p1 = tmp_path / "p1.csv"
        p2 = tmp_path / "p2.csv"
        p1.write_bytes(payload)
        p2.write_bytes(payload)
        r1 = compute_run_id(
            input_csv=p1,
            schema_fingerprint=SchemaFingerprint("x"),
            code_version_str=CodeVersion("0.1.0"),
        )
        r2 = compute_run_id(
            input_csv=p2,
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
