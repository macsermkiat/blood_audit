"""RED-phase failing tests for issue #3 (bba.ingest).

Each ``class`` maps to one acceptance criterion in the issue body. Tests assert
contracts (the WHY), not implementation choices — see PRD §"Testing Decisions".

No implementation exists yet; every test MUST fail in this scaffold commit.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bba.ingest.hashing import compute_run_id, content_hash
from bba.ingest.models import (
    CSVTable,
    IngestConfig,
    IngestResult,
    ParseResult,
)
from bba.ingest.pipeline import ingest
from bba.ingest.schemas import (
    IncompleteInputError,
    SchemaDriftError,
    all_tables,
    get_schema,
    schema_fingerprint,
)
from bba.ingest.time_parser import parse_hosxp_time
from bba.ingest.tz import to_utc
from bba.ingest.writer import is_run_complete


REQUIRED_TABLES: tuple[CSVTable, ...] = (
    "BDVST",
    "BDVSTDT",
    "BDTYPE",
    "BDVSTST",
    "Diagnosis",
    "Lab",
    "MED",
    "IPDADMPROGRESS",
    "IPDNRFOCUSDT",
    "UnUSE_Patient_Background",
)


def _write_complete_hosxp(in_dir: Path) -> None:
    """Populate ``in_dir`` with a minimal valid CSV for every required HOSxP table.

    Header columns are derived from the live pandera schemas so the fixture
    follows future schema bumps without manual edits. Each file has the header
    plus one empty-string row, which is enough to exercise the header-drift
    and run-id paths without depending on per-row parsing (#4-#7).
    """
    in_dir.mkdir(parents=True, exist_ok=True)
    for table in REQUIRED_TABLES:
        schema = get_schema(table)
        cols = list(schema.columns)
        header = ",".join(cols)
        row = ",".join("" for _ in cols)
        (in_dir / f"{table}.csv").write_text(f"{header}\n{row}\n", encoding="utf-8")


@pytest.fixture
def complete_hosxp_dir(tmp_path: Path) -> Path:
    """A fully-populated minimal HOSxP input dir for tests that need ingest to succeed."""
    in_dir = tmp_path / "in"
    _write_complete_hosxp(in_dir)
    return in_dir


# =============================================================================
# AC: package with strict time parser, pandera schemas, DuckDB writer
# (Public API surface is implicitly verified by the imports at the top of this
# module — if any re-export is missing, collection fails. Per-suite tests below
# exercise the actual behavior.)
# =============================================================================


# =============================================================================
# AC: strict time parser — happy path + sentinel + unrecognized formats
# AC: unrecognized → parse_warning (NULL value); never silently shifts
# =============================================================================


class TestTimeParserHappyPath:
    """Allow-listed formats parse cleanly with no parse_warning."""

    def test_hhmmss_six_digit(self) -> None:
        r = parse_hosxp_time("083045")
        assert r.parse_warning is None
        assert r.value is not None
        assert (r.value.hour, r.value.minute, r.value.second) == (8, 30, 45)
        assert r.raw == "083045"

    def test_hh_mm_colon(self) -> None:
        r = parse_hosxp_time("08:30")
        assert r.parse_warning is None
        assert r.value is not None
        assert (r.value.hour, r.value.minute) == (8, 30)

    def test_midnight_hhmmss(self) -> None:
        r = parse_hosxp_time("000000")
        assert r.parse_warning is None
        assert r.value is not None
        assert (r.value.hour, r.value.minute, r.value.second) == (0, 0, 0)

    def test_end_of_day_hhmmss(self) -> None:
        r = parse_hosxp_time("235959")
        assert r.parse_warning is None
        assert r.value is not None
        assert (r.value.hour, r.value.minute, r.value.second) == (23, 59, 59)


class TestTimeParserRejectsUnrecognizedFormats:
    """Per Round 2 fix E35, every unrecognized format becomes a parse_warning,
    never a silently shifted value. The whole reason the ±24-h evidence
    window stays correct."""

    def test_decimal_hour(self) -> None:
        # "8.5" could mean 08:30, but the strict parser refuses to guess.
        r = parse_hosxp_time("8.5")
        assert r.value is None
        assert r.parse_warning is not None and r.parse_warning != ""

    def test_excel_serial_fraction(self) -> None:
        # Excel serial fraction (~0.354 ~= 08:30) — also refused.
        r = parse_hosxp_time("0.354166")
        assert r.value is None
        assert r.parse_warning is not None

    def test_buddhist_year_prefix(self) -> None:
        # Buddhist-year dates leak into time columns occasionally; refuse them.
        r = parse_hosxp_time("2568-01-01")
        assert r.value is None
        assert r.parse_warning is not None

    def test_sentinel_zero(self) -> None:
        r = parse_hosxp_time("0")
        assert r.value is None
        assert r.parse_warning is not None

    def test_sentinel_9999(self) -> None:
        r = parse_hosxp_time("9999")
        assert r.value is None
        assert r.parse_warning is not None

    def test_null_token(self) -> None:
        r = parse_hosxp_time("null")
        assert r.value is None
        assert r.parse_warning is not None

    def test_empty_string(self) -> None:
        r = parse_hosxp_time("")
        assert r.value is None
        assert r.parse_warning is not None

    def test_garbage(self) -> None:
        r = parse_hosxp_time("not-a-time-at-all")
        assert r.value is None
        assert r.parse_warning is not None

    def test_none_input(self) -> None:
        r = parse_hosxp_time(None)
        assert r.value is None
        assert r.parse_warning is not None


class TestTimeParserInvariants:
    """Property-based invariants (PRD §"Testing Decisions" — property where invariant)."""

    @given(
        h=st.integers(min_value=0, max_value=23),
        m=st.integers(min_value=0, max_value=59),
        s=st.integers(min_value=0, max_value=59),
    )
    def test_hhmmss_round_trip(self, h: int, m: int, s: int) -> None:
        raw = f"{h:02d}{m:02d}{s:02d}"
        r = parse_hosxp_time(raw)
        assert r.parse_warning is None, f"valid HHMMSS rejected: {raw!r} → {r.parse_warning!r}"
        assert r.value is not None
        assert (r.value.hour, r.value.minute, r.value.second) == (h, m, s)
        assert r.raw == raw

    @given(
        h=st.integers(min_value=0, max_value=23),
        m=st.integers(min_value=0, max_value=59),
    )
    def test_hh_mm_round_trip(self, h: int, m: int) -> None:
        raw = f"{h:02d}:{m:02d}"
        r = parse_hosxp_time(raw)
        assert r.parse_warning is None
        assert r.value is not None
        assert (r.value.hour, r.value.minute) == (h, m)

    @given(st.text(min_size=0, max_size=20))
    @settings(max_examples=300)
    def test_either_value_or_warning_never_both_or_neither(self, raw: str) -> None:
        r = parse_hosxp_time(raw)
        # Exactly one of value / parse_warning is non-None.
        has_value = r.value is not None
        has_warning = r.parse_warning is not None
        assert has_value ^ has_warning, (
            f"parser produced value={r.value!r}, warning={r.parse_warning!r} for raw={raw!r}; "
            "exactly one must be set"
        )

    @given(st.text(min_size=1, max_size=8).filter(
        lambda s: not (s.isdigit() and len(s) == 6)
        and not (len(s) == 5 and s[2] == ":" and s[:2].isdigit() and s[3:].isdigit())
    ))
    @settings(max_examples=300)
    def test_unknown_format_never_silently_yields_a_value(self, raw: str) -> None:
        # Inputs that are NOT on the allow-list must produce value=None.
        r = parse_hosxp_time(raw)
        if r.value is not None:
            pytest.fail(
                f"strict parser silently produced {r.value!r} for unrecognized input {raw!r}"
            )


# =============================================================================
# AC: All 10 CSVs ingestable — pandera schemas registered for every table
# =============================================================================


class TestSchemaCoverage:
    @pytest.mark.parametrize("table", REQUIRED_TABLES)
    def test_every_required_table_has_a_schema(self, table: CSVTable) -> None:
        schema = get_schema(table)
        assert schema is not None

    def test_all_tables_returns_canonical_ten(self) -> None:
        assert tuple(all_tables()) == REQUIRED_TABLES

    def test_no_extra_tables_silently_registered(self) -> None:
        # Adding a table without bumping the schema version would change the
        # fingerprint and produce a new run_id — but we still want a hard
        # tripwire so a contributor cannot register a tenth-plus-one silently.
        assert len(all_tables()) == 10


# =============================================================================
# AC: Schema-drift detection — unknown column → fail loud with helpful error
# =============================================================================


class TestSchemaDriftDetection:
    def test_unknown_column_raises_schema_drift_error(self, tmp_path: Path) -> None:
        # Craft a single-table input dir; the BDVST.csv has an unknown column.
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "BDVST.csv").write_text("HN,SOMETHING_NEW\n123,foo\n", encoding="utf-8")

        cfg = IngestConfig(
            input_dir=in_dir,
            output_dir=tmp_path / "out",
            code_version="test-0.0.1",
        )
        with pytest.raises(SchemaDriftError) as exc_info:
            ingest(cfg)
        msg = str(exc_info.value)
        # Helpful error must surface both the offending column and the table.
        assert "SOMETHING_NEW" in msg, f"drift error did not name unknown column: {msg!r}"
        assert "BDVST" in msg, f"drift error did not name the source table: {msg!r}"


# =============================================================================
# AC: All timestamps stored UTC; rendered Asia/Bangkok at boundaries
# =============================================================================


class TestTimezoneNormalization:
    def test_bangkok_local_to_utc_subtracts_seven_hours(self) -> None:
        # Asia/Bangkok is fixed UTC+07:00 (no DST since 1920); 15:00 local → 08:00 UTC.
        local = datetime(2026, 5, 14, 15, 0, 0)
        utc = to_utc(local, tz="Asia/Bangkok")
        assert utc.tzinfo is not None, "to_utc must return a tz-aware datetime"
        assert utc.utcoffset() == timedelta(0), "tzinfo must be UTC, not a local zone"
        assert (utc.year, utc.month, utc.day, utc.hour, utc.minute) == (2026, 5, 14, 8, 0)

    def test_midnight_local_crosses_day_boundary(self) -> None:
        # Month-boundary correctness (PRD §1, fix E31): 2026-05-15 03:00 +07
        # ↦ 2026-05-14 20:00 UTC — must NOT bucket as the 15th.
        local = datetime(2026, 5, 15, 3, 0, 0)
        utc = to_utc(local, tz="Asia/Bangkok")
        assert (utc.year, utc.month, utc.day, utc.hour) == (2026, 5, 14, 20)


# =============================================================================
# AC: Lint rule banning naive datetime.now()/utcnow() in this module
# =============================================================================


class TestNaiveDatetimeBan:
    """Structural check that complements the ruff rule.

    Why this is a test, not "just" a ruff rule: a contributor who silences ruff
    locally still hits this test in CI.
    """

    @pytest.mark.parametrize("forbidden", ["datetime.now()", "datetime.utcnow()"])
    def test_no_naive_datetime_calls_in_module(self, forbidden: str) -> None:
        import bba.ingest as pkg

        module_dir = Path(pkg.__file__).parent
        offenders: list[str] = []
        for py in module_dir.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            if forbidden in text:
                offenders.append(str(py.relative_to(module_dir)))
        assert not offenders, (
            f"forbidden naive datetime call {forbidden!r} found in: {offenders}; "
            "use tz-aware datetime.now(timezone.utc) instead"
        )


# =============================================================================
# AC: run_id idempotency — sha256(input_csv_hash + schema_fingerprint + code_version)
# =============================================================================


class TestRunIdContract:
    def test_run_id_is_sha256_of_concatenated_inputs(self) -> None:
        # The concatenation order is part of the public contract (PRD §1, fix E32).
        a = "a" * 64  # plausible sha256 hex of input
        b = "b" * 64  # plausible schema fingerprint
        v = "0.1.0"
        expected = hashlib.sha256((a + b + v).encode("utf-8")).hexdigest()
        assert compute_run_id(a, b, v) == expected

    def test_same_inputs_same_run_id(self) -> None:
        assert compute_run_id("x", "y", "z") == compute_run_id("x", "y", "z")

    def test_run_id_changes_when_any_component_changes(self) -> None:
        base = compute_run_id("x", "y", "z")
        assert compute_run_id("X", "y", "z") != base
        assert compute_run_id("x", "Y", "z") != base
        assert compute_run_id("x", "y", "Z") != base

    def test_run_id_is_64_char_hex(self) -> None:
        rid = compute_run_id("x", "y", "z")
        assert isinstance(rid, str) and len(rid) == 64
        int(rid, 16)  # parses as hex


class TestIsRunComplete:
    def test_missing_output_dir_is_not_complete(self, tmp_path: Path) -> None:
        # A run that has never written anywhere is, by definition, incomplete.
        assert is_run_complete(tmp_path, "feedface" * 8) is False

    def test_independent_run_ids_do_not_collide(self, tmp_path: Path) -> None:
        # The completion check must be scoped to the specific run_id.
        out = tmp_path / "out"
        out.mkdir()
        a = "a" * 64
        b = "b" * 64
        # Without any side-effects, both must be False — and the function must
        # at minimum be callable on a fresh directory.
        assert is_run_complete(out, a) is False
        assert is_run_complete(out, b) is False


class TestSchemaFingerprintStable:
    def test_fingerprint_is_deterministic(self) -> None:
        assert schema_fingerprint() == schema_fingerprint()

    def test_fingerprint_is_sha256_hex(self) -> None:
        fp = schema_fingerprint()
        assert isinstance(fp, str) and len(fp) == 64
        int(fp, 16)


# =============================================================================
# AC: content hashing (input_csv_hash feeds run_id)
# =============================================================================


class TestContentHash:
    def test_same_bytes_same_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "x.csv"
        f.write_bytes(b"a,b,c\n1,2,3\n")
        assert content_hash(f) == content_hash(f)

    def test_hash_is_sha256_hex(self, tmp_path: Path) -> None:
        f = tmp_path / "x.csv"
        payload = b"HN,DATE\n1,2026-05-14\n"
        f.write_bytes(payload)
        h = content_hash(f)
        assert h == hashlib.sha256(payload).hexdigest()

    def test_different_bytes_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        a.write_bytes(b"a,b\n1,2\n")
        b.write_bytes(b"a,b\n1,3\n")
        assert content_hash(a) != content_hash(b)


# =============================================================================
# AC: Pydantic v2 models — outputs returned by the public API are immutable
# (Tests are framed around the public function outputs rather than direct model
# construction, so they fail in RED when the impl is missing.)
# =============================================================================


class TestPublicOutputsAreImmutable:
    def test_parse_result_from_parser_is_frozen(self) -> None:
        r = parse_hosxp_time("083045")
        assert isinstance(r, ParseResult)
        with pytest.raises(ValidationError):
            r.value = datetime(2026, 1, 1)  # type: ignore[misc]

    def test_ingest_result_from_pipeline_is_frozen(
        self, tmp_path: Path, complete_hosxp_dir: Path
    ) -> None:
        cfg = IngestConfig(
            input_dir=complete_hosxp_dir,
            output_dir=tmp_path / "out",
            code_version="0.0.1",
        )
        result = ingest(cfg)
        assert isinstance(result, IngestResult)
        with pytest.raises(ValidationError):
            result.rows_written = 999  # type: ignore[misc]


class TestIngestResultShape:
    def test_pipeline_returns_run_id_and_idempotency_flag(
        self, tmp_path: Path, complete_hosxp_dir: Path
    ) -> None:
        cfg = IngestConfig(
            input_dir=complete_hosxp_dir,
            output_dir=tmp_path / "out",
            code_version="0.0.1",
        )
        first = ingest(cfg)
        # First run is not a no-op.
        assert first.skipped_idempotent is False
        assert isinstance(first.run_id, str) and len(first.run_id) == 64

    def test_pipeline_second_call_is_idempotent_noop(
        self, tmp_path: Path, complete_hosxp_dir: Path
    ) -> None:
        cfg = IngestConfig(
            input_dir=complete_hosxp_dir,
            output_dir=tmp_path / "out",
            code_version="0.0.1",
        )
        first = ingest(cfg)
        second = ingest(cfg)
        assert second.run_id == first.run_id
        assert second.skipped_idempotent is True


# =============================================================================
# Codex-review follow-ups for issue #3
# =============================================================================


class TestSchemaMissingRequiredColumn:
    """Codex P2 (pipeline.py:74-75): the drift check only rejected unknown
    columns; a CSV that omitted a required column slipped past, was hashed,
    and marked complete. Missing required columns must fail loud just like
    unknown ones — downstream joins rely on every declared column."""

    def test_missing_required_column_raises_schema_drift_error(
        self, tmp_path: Path
    ) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        # BDVST declares HN, AN, REQNO, BDVSTST, REQTYPE, CANCELDATE.
        # This CSV provides only HN + AN — REQNO and four others are missing.
        (in_dir / "BDVST.csv").write_text("HN,AN\n123,456\n", encoding="utf-8")
        cfg = IngestConfig(
            input_dir=in_dir,
            output_dir=tmp_path / "out",
            code_version="test-0.0.1",
        )
        with pytest.raises(SchemaDriftError) as exc_info:
            ingest(cfg)
        msg = str(exc_info.value)
        # Helpful error must surface a specific missing column + the table.
        assert "REQNO" in msg, f"drift error did not name missing column REQNO: {msg!r}"
        assert "BDVST" in msg, f"drift error did not name the source table: {msg!r}"

    def test_missing_column_check_does_not_write_complete_marker(
        self, tmp_path: Path
    ) -> None:
        # No partial state must be left behind when drift is detected.
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        out_dir = tmp_path / "out"
        (in_dir / "BDVST.csv").write_text("HN,AN\n123,456\n", encoding="utf-8")
        cfg = IngestConfig(input_dir=in_dir, output_dir=out_dir, code_version="test")
        with pytest.raises(SchemaDriftError):
            ingest(cfg)
        # The writer must NOT have created any completion marker.
        markers = list(out_dir.glob("_run_*.complete")) if out_dir.exists() else []
        assert markers == [], f"drift run leaked a complete marker: {markers}"


class TestIngestResultTablesWrittenIsImmutable:
    """Codex P2 (models.py:65-67): `frozen=True` blocks reassignment but does
    NOT freeze a nested mutable container, so a caller could `append` to
    `result.tables_written`. The public output contract promises immutability —
    use a tuple."""

    def test_tables_written_is_a_tuple(
        self, tmp_path: Path, complete_hosxp_dir: Path
    ) -> None:
        cfg = IngestConfig(
            input_dir=complete_hosxp_dir,
            output_dir=tmp_path / "out",
            code_version="0.0.1",
        )
        result = ingest(cfg)
        assert isinstance(result.tables_written, tuple), (
            f"tables_written must be a tuple for immutability, got "
            f"{type(result.tables_written).__name__}"
        )

    def test_tables_written_cannot_be_appended_to(
        self, tmp_path: Path, complete_hosxp_dir: Path
    ) -> None:
        cfg = IngestConfig(
            input_dir=complete_hosxp_dir,
            output_dir=tmp_path / "out",
            code_version="0.0.1",
        )
        result = ingest(cfg)
        # tuple has no .append; AttributeError is the immutability signal.
        with pytest.raises(AttributeError):
            result.tables_written.append("BDVST")  # type: ignore[attr-defined]


# =============================================================================
# Codex pass-2 P1: reject incomplete input directories
# =============================================================================


class TestIncompleteInputRejection:
    """Codex P1 (pipeline.py:91-92): an input_dir that is missing, empty, or
    lacks one of the 10 canonical HOSxP CSVs used to get hashed and marked
    complete — making a typo or partial export look successful, with the
    incomplete run no-op'd as idempotent on retry. Every "incomplete set"
    must fail loud."""

    def test_missing_input_dir_raises(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does-not-exist"
        cfg = IngestConfig(
            input_dir=nonexistent,
            output_dir=tmp_path / "out",
            code_version="test",
        )
        with pytest.raises(IncompleteInputError) as exc_info:
            ingest(cfg)
        assert "does-not-exist" in str(exc_info.value) or "missing" in str(exc_info.value).lower()

    def test_empty_input_dir_raises(self, tmp_path: Path) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        cfg = IngestConfig(
            input_dir=in_dir, output_dir=tmp_path / "out", code_version="test"
        )
        with pytest.raises(IncompleteInputError) as exc_info:
            ingest(cfg)
        # Error must name at least one missing canonical table.
        msg = str(exc_info.value)
        assert any(table in msg for table in REQUIRED_TABLES), (
            f"empty-dir error did not name any missing table: {msg!r}"
        )

    def test_missing_one_table_raises(
        self, tmp_path: Path, complete_hosxp_dir: Path
    ) -> None:
        # Remove a single canonical CSV; the rest are fine.
        (complete_hosxp_dir / "Lab.csv").unlink()
        cfg = IngestConfig(
            input_dir=complete_hosxp_dir,
            output_dir=tmp_path / "out",
            code_version="test",
        )
        with pytest.raises(IncompleteInputError) as exc_info:
            ingest(cfg)
        assert "Lab" in str(exc_info.value)

    def test_incomplete_run_does_not_write_complete_marker(
        self, tmp_path: Path
    ) -> None:
        # The raise must happen before any marker is written.
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        out_dir = tmp_path / "out"
        cfg = IngestConfig(input_dir=in_dir, output_dir=out_dir, code_version="test")
        with pytest.raises(IncompleteInputError):
            ingest(cfg)
        markers = list(out_dir.glob("_run_*.complete")) if out_dir.exists() else []
        assert markers == [], f"incomplete run leaked a complete marker: {markers}"
