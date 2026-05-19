"""RED-phase failing tests for issue #3 (bba.ingest).

Each ``class`` maps to one acceptance criterion in the issue body. Tests assert
contracts (the WHY), not implementation choices — see PRD §"Testing Decisions".

No implementation exists yet; every test MUST fail in this scaffold commit.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bba.ingest.hashing import content_hash
from bba.ingest.models import (
    CSVTable,
    IngestConfig,
    IngestResult,
    ParsedTimeOfDay,
    ParseResult,
)
from bba.ingest.pipeline import ingest
from bba.ingest.row_timestamp import RowTimestamp
from bba.ingest.run_identity import RunIdentity
from bba.ingest.schemas import (
    IncompleteInputError,
    SchemaDriftError,
    all_tables,
    get_schema,
    schema_fingerprint,
    validate_header,
)
from bba.ingest.date_parser import parse_iptsumoprt_date
from bba.ingest.normalize import (
    RAGGED_ROW_WARNING_KEY,
    _row_positions,
    normalize_header,
    normalize_row,
    normalize_rows,
)
from bba.ingest.time_parser import parse_hosxp_time


REQUIRED_TABLES: tuple[CSVTable, ...] = (
    "BDVST",
    "BDVSTDT",
    "BDVSTST",
    "BDTYPE",
    "Diagnosis",
    "Lab",
    "Med",
    "IPDADMPROGRESS",
    "IPDNRFOCUSDT",
    "IPTSUMOPRT",
    "ICD9CM",
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
        assert r.parse_warning is None, (
            f"valid HHMMSS rejected: {raw!r} → {r.parse_warning!r}"
        )
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

    @given(
        st.text(min_size=1, max_size=8).filter(
            lambda s: (
                not (s.isdigit() and len(s) == 6)
                and not (
                    len(s) == 5 and s[2] == ":" and s[:2].isdigit() and s[3:].isdigit()
                )
            )
        )
    )
    @settings(max_examples=300)
    def test_unknown_format_never_silently_yields_a_value(self, raw: str) -> None:
        # Inputs that are NOT on the allow-list must produce value=None.
        r = parse_hosxp_time(raw)
        if r.value is not None:
            pytest.fail(
                f"strict parser silently produced {r.value!r} for unrecognized input {raw!r}"
            )


# =============================================================================
# AC: KCMH-export date parser — IPTSUMOPRT.INDATE long-form, date-only output
# (companion to parse_hosxp_time's long-form branch which discards date; this
# parser discards the embedded time and extracts only the calendar date.)
# =============================================================================


class TestIPTSUMOPRTDateParserHappyPath:
    """Allow-listed English-locale long-form formats parse to a date with no warning."""

    def test_basic_morning(self) -> None:
        r = parse_iptsumoprt_date("June 7, 2025, 12:00 AM")
        assert r.parse_warning is None
        assert r.value is not None
        assert (r.value.year, r.value.month, r.value.day) == (2025, 6, 7)
        assert r.raw == "June 7, 2025, 12:00 AM"

    def test_january_first(self) -> None:
        r = parse_iptsumoprt_date("January 1, 2014, 12:00 AM")
        assert r.parse_warning is None
        assert r.value is not None
        assert (r.value.year, r.value.month, r.value.day) == (2014, 1, 1)

    def test_pm_time_ignored(self) -> None:
        # PM in the embedded time component must not affect the extracted date.
        r = parse_iptsumoprt_date("June 18, 2025, 6:13 PM")
        assert r.parse_warning is None
        assert r.value is not None
        assert (r.value.year, r.value.month, r.value.day) == (2025, 6, 18)

    def test_two_digit_day(self) -> None:
        r = parse_iptsumoprt_date("December 31, 2024, 11:59 PM")
        assert r.parse_warning is None
        assert r.value is not None
        assert (r.value.year, r.value.month, r.value.day) == (2024, 12, 31)

    def test_single_digit_day(self) -> None:
        r = parse_iptsumoprt_date("February 3, 2025, 12:00 AM")
        assert r.parse_warning is None
        assert r.value is not None
        assert (r.value.year, r.value.month, r.value.day) == (2025, 2, 3)


class TestIPTSUMOPRTDateParserRejectsUnrecognizedFormats:
    """Strict invariant (mirrors PRD §1 fix E35): unrecognized formats produce a
    parse_warning, never a silently mis-shifted date."""

    def test_none_input(self) -> None:
        r = parse_iptsumoprt_date(None)
        assert r.value is None
        assert r.parse_warning is not None

    def test_empty_string(self) -> None:
        r = parse_iptsumoprt_date("")
        assert r.value is None
        assert r.parse_warning is not None

    def test_hosxp_standard_format_is_not_accepted(self) -> None:
        # The HOSxP standard "2025-06-07 00:00:00.000" must NOT be accepted by
        # this parser — that branch is handled elsewhere and silent overlap
        # would mask schema-source confusion.
        r = parse_iptsumoprt_date("2025-06-07 00:00:00.000")
        assert r.value is None
        assert r.parse_warning is not None

    def test_iso_date_only(self) -> None:
        r = parse_iptsumoprt_date("2025-06-07")
        assert r.value is None
        assert r.parse_warning is not None

    def test_invalid_calendar_date(self) -> None:
        # "June 31, 2025" is not a real date. Must reject, never shift to July 1.
        r = parse_iptsumoprt_date("June 31, 2025, 12:00 AM")
        assert r.value is None
        assert r.parse_warning is not None
        assert "invalid calendar date" in r.parse_warning

    def test_unknown_month_name(self) -> None:
        r = parse_iptsumoprt_date("Junio 7, 2025, 12:00 AM")
        assert r.value is None
        assert r.parse_warning is not None

    def test_garbage(self) -> None:
        r = parse_iptsumoprt_date("not-a-date-at-all")
        assert r.value is None
        assert r.parse_warning is not None

    def test_value_xor_warning_invariant(self) -> None:
        # Exactly one of value / parse_warning is non-None — same invariant as
        # the time parser (covered by property test there).
        for raw in [
            "June 7, 2025, 12:00 AM",
            "garbage",
            "",
            None,
            "2025-06-07 00:00:00.000",
            "June 31, 2025, 12:00 AM",
        ]:
            r = parse_iptsumoprt_date(raw)
            has_value = r.value is not None
            has_warning = r.parse_warning is not None
            assert has_value ^ has_warning, (
                f"parser produced value={r.value!r}, warning={r.parse_warning!r} "
                f"for raw={raw!r}; exactly one must be set"
            )


# =============================================================================
# AC: Normalize layer — projection + case-normalize + dedupe before
# validate_header. Policy (a) per docs/ingest-mapping.md: drop undeclared
# extras and log them rather than failing loud, but keep "missing required
# column" as a fatal drift signal.
# =============================================================================


class TestNormalizeHeaderProjection:
    """Undeclared extras land in ``dropped``; declared columns land in ``header``."""

    def test_clean_header_passes_through_unchanged(self) -> None:
        # Every declared column present, no extras — projection is a no-op.
        cols = list(get_schema("BDVST").columns)
        r = normalize_header("BDVST", cols)
        assert r.header == cols
        assert r.dropped == []

    def test_undeclared_extras_are_dropped(self) -> None:
        cols = list(get_schema("BDVST").columns)
        r = normalize_header("BDVST", [*cols, "FOO", "BAR"])
        assert set(r.header) == set(cols)
        assert r.dropped == ["FOO", "BAR"]

    def test_only_undeclared_columns_yields_empty_kept(self) -> None:
        # Pathological input: nothing declared is present. Header is empty
        # (validate_header will then raise on every missing required column).
        r = normalize_header("BDVST", ["FOO", "BAR"])
        assert r.header == []
        assert r.dropped == ["FOO", "BAR"]

    def test_dropped_preserves_input_order(self) -> None:
        # Auditors reading the dropped list should see file order, not sort
        # order, so they can correlate with column positions.
        cols = list(get_schema("BDVST").columns)
        r = normalize_header("BDVST", [*cols, "Z_LAST", "A_FIRST"])
        assert r.dropped == ["Z_LAST", "A_FIRST"]


class TestNormalizeHeaderCaseNormalize:
    """IPTSUMOPRT and ICD9CM exports use Title-Case; normalize uppercases
    before projection so they line up with ALL-CAPS schema declarations."""

    def test_iptsumoprt_title_case_uppercased(self) -> None:
        # The real file header (Title-Case) projects to the declared
        # ALL-CAPS columns after normalize.
        title_case = ["An", "Diagorder", "Itemno", "Icd9cm", "Indate", "Intime"]
        r = normalize_header("IPTSUMOPRT", title_case)
        assert "AN" in r.header
        assert "ICD9CM" in r.header
        assert "INDATE" in r.header
        assert "INTIME" in r.header
        # Diagorder/Itemno are not in the locked schema → dropped (as upper-cased).
        assert "DIAGORDER" in r.dropped
        assert "ITEMNO" in r.dropped

    def test_icd9cm_title_case_uppercased(self) -> None:
        title_case = ["Icd9cm", "Name", "Thainame", "Orflag", "Firststf"]
        r = normalize_header("ICD9CM", title_case)
        assert set(r.header) == {"ICD9CM", "NAME", "ORFLAG"}
        # Thainame and Firststf are not declared → dropped (upper-cased).
        assert "THAINAME" in r.dropped
        assert "FIRSTSTF" in r.dropped

    def test_non_procedure_table_keeps_case_as_is(self) -> None:
        # Only IPTSUMOPRT and ICD9CM are case-normalized. Other tables that
        # somehow arrive with mixed-case columns would correctly fail the
        # projection (mixed-case names are not in the declared set).
        r = normalize_header("BDVST", ["hn", "REQNO"])
        # "hn" lowercase is NOT the same as declared "HN" → dropped.
        assert "hn" in r.dropped
        assert r.header == ["REQNO"]


class TestNormalizeHeaderDedupe:
    """IPDADMPROGRESS arrives with duplicate HN and AN columns (positions
    1 + 30 and 2 + 3). The normalize layer drops duplicates by first-
    occurrence-wins, which matches the locked spec's positional-read rule
    without needing column-index bookkeeping."""

    def test_ipdadmprogress_duplicate_hn_an_deduped(self) -> None:
        # Real header shape (abbreviated to the duplicates + a declared field).
        r = normalize_header(
            "IPDADMPROGRESS",
            ["HN", "AN", "AN", "PROGDATE", "HN", "OBJECTIVE"],
        )
        # First HN and first AN survive; second occurrences land in dropped.
        assert r.header.count("HN") == 1
        assert r.header.count("AN") == 1
        # Two duplicate names → two entries in dropped, in source order.
        assert r.dropped == ["AN", "HN"]

    def test_dedupe_applied_universally(self) -> None:
        # The dedupe rule is global so an unexpected duplicate in any future
        # export shows up in the dropped list rather than silently corrupting
        # validate_header's set-based drift check.
        cols = list(get_schema("BDVST").columns)
        r = normalize_header("BDVST", [*cols, "HN"])
        assert r.header == cols
        assert r.dropped == ["HN"]  # the second HN, dropped by first-wins rule


# =============================================================================
# AC: Per-row normalize layer (issue #61) — position map, projection,
# year-filter, INDATE date-parse, IPDADMPROGRESS positional dedupe
# alignment. Drains rows after validate_header passes and before any
# Parquet write; surfaces row-level parse warnings to the run audit.
# =============================================================================


class TestRowPositions:
    """The position map mirrors normalize_header so the row pass aligns
    1:1 with the projected header. First-occurrence wins on duplicates;
    case-normalize applies to IPTSUMOPRT and ICD9CM."""

    def test_plain_table_positions_match_input_order(self) -> None:
        raw = list(get_schema("BDVST").columns)
        kept = list(get_schema("BDVST").columns)
        assert _row_positions("BDVST", raw, kept) == tuple(range(len(raw)))

    def test_iptsumoprt_title_case_resolves_to_raw_positions(self) -> None:
        # The real bundle ships Title-Case names. _row_positions must
        # match by upper-cased name and return the raw position so the
        # row reader picks the correct cell.
        raw = ["An", "Diagorder", "Itemno", "Icd9cm", "Indate", "Intime"]
        kept = ["AN", "ICD9CM", "INDATE", "INTIME"]
        positions = _row_positions("IPTSUMOPRT", raw, kept)
        # AN→0, ICD9CM→3, INDATE→4, INTIME→5
        assert positions == (0, 3, 4, 5)

    def test_ipdadmprogress_duplicates_resolve_to_first_position(self) -> None:
        # Real file shape (abbreviated): HN at 0, AN at 1, duplicate AN
        # at 2, PROGDATE at 3, duplicate HN at 4, SOAP fields follow.
        raw = [
            "HN",
            "AN",
            "AN",  # dup → ignored
            "PROGDATE",
            "HN",  # dup → ignored
            "SUBJECTIVE",
            "OBJECTIVE",
            "ASSESSMENT",
            "PLAN",
        ]
        kept = ["HN", "AN", "PROGDATE", "SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"]
        positions = _row_positions("IPDADMPROGRESS", raw, kept)
        # First HN → 0, first AN → 1, PROGDATE → 3, SOAP → 5..8.
        # The duplicate AN at 2 and duplicate HN at 4 are skipped.
        assert positions == (0, 1, 3, 5, 6, 7, 8)


class TestNormalizeRowProjection:
    """Projection picks the right cells; the result aligns with kept_header."""

    def test_projection_picks_cells_by_position(self) -> None:
        raw = list(get_schema("BDVST").columns)
        kept = list(get_schema("BDVST").columns)
        positions = _row_positions("BDVST", raw, kept)
        # Construct a raw row with distinctive values per column.
        raw_row = [f"v{i}" for i in range(len(raw))]
        result = normalize_row("BDVST", raw_row, positions, kept)
        assert result is not None
        # Cells align with kept_header order.
        assert result.cells == tuple(raw_row)
        # No row-level rules fire for BDVST → no warnings.
        assert result.parse_warnings == ()

    def test_projection_drops_undeclared_cells(self) -> None:
        # Raw row has extras at the tail; positions only reference the
        # declared columns, so the extras are not read.
        raw = [*get_schema("BDVST").columns, "EXTRA_A", "EXTRA_B"]
        kept = list(get_schema("BDVST").columns)
        positions = _row_positions("BDVST", raw, kept)
        raw_row = [f"v{i}" for i in range(len(raw))]
        result = normalize_row("BDVST", raw_row, positions, kept)
        assert result is not None
        # Only the first len(kept) cells survive; trailing extras dropped.
        assert result.cells == tuple(raw_row[: len(kept)])


class TestNormalizeRowYearFilter:
    """IPDADMPROGRESS and IPDNRFOCUSDT drop rows whose date column is
    outside the cohort year. Single-point constant in normalize.py
    (COHORT_YEAR) controls this; year-filter columns are per-table
    declarative mappings."""

    def _ipdadm_setup(self) -> tuple[tuple[int, ...], list[str]]:
        kept = list(get_schema("IPDADMPROGRESS").columns)
        positions = _row_positions("IPDADMPROGRESS", kept, kept)
        return positions, kept

    def _ipdadm_row(self, kept: list[str], progdate: str) -> list[str]:
        idx = kept.index("PROGDATE")
        row = ["" for _ in kept]
        row[idx] = progdate
        return row

    def test_ipdadmprogress_in_cohort_year_kept(self) -> None:
        positions, kept = self._ipdadm_setup()
        row = self._ipdadm_row(kept, "2025-05-19 00:00:00.000")
        result = normalize_row("IPDADMPROGRESS", row, positions, kept)
        assert result is not None

    def test_ipdadmprogress_prior_year_dropped(self) -> None:
        positions, kept = self._ipdadm_setup()
        row = self._ipdadm_row(kept, "2024-12-31 23:59:59.999")
        assert normalize_row("IPDADMPROGRESS", row, positions, kept) is None

    def test_ipdadmprogress_following_year_dropped(self) -> None:
        # Spillover into the year after the cohort must also drop.
        positions, kept = self._ipdadm_setup()
        row = self._ipdadm_row(kept, "2026-01-01 00:00:00.000")
        assert normalize_row("IPDADMPROGRESS", row, positions, kept) is None

    def test_ipdadmprogress_empty_progdate_dropped(self) -> None:
        # Blank PROGDATE cannot satisfy any year filter; conservatism
        # keeps it out of the audit cohort.
        positions, kept = self._ipdadm_setup()
        row = self._ipdadm_row(kept, "")
        assert normalize_row("IPDADMPROGRESS", row, positions, kept) is None

    def test_ipdnrfocusdt_year_filter_on_progressdate(self) -> None:
        kept = list(get_schema("IPDNRFOCUSDT").columns)
        positions = _row_positions("IPDNRFOCUSDT", kept, kept)
        idx = kept.index("PROGRESSDATE")
        in_row = ["" for _ in kept]
        in_row[idx] = "2025-08-01 00:00:00.000"
        out_row = ["" for _ in kept]
        out_row[idx] = "2021-08-01 00:00:00.000"
        assert normalize_row("IPDNRFOCUSDT", in_row, positions, kept) is not None
        assert normalize_row("IPDNRFOCUSDT", out_row, positions, kept) is None

    def test_other_tables_have_no_year_filter(self) -> None:
        # BDVST has no year-filter rule even though it has date columns.
        # A row with a non-2025 anchor must still pass through (the
        # audit-orders layer applies its own cohort logic).
        kept = list(get_schema("BDVST").columns)
        positions = _row_positions("BDVST", kept, kept)
        row = ["" for _ in kept]
        # Put a 1999 anchor in REQDATE and verify the row is kept.
        row[kept.index("REQDATE")] = "1999-01-01 00:00:00.000"
        result = normalize_row("BDVST", row, positions, kept)
        assert result is not None

    def test_custom_cohort_year_parameter(self) -> None:
        # The cohort_year argument is the single point of change for
        # future audit windows. A 2026 cohort run drops 2025 rows.
        kept = list(get_schema("IPDADMPROGRESS").columns)
        positions = _row_positions("IPDADMPROGRESS", kept, kept)
        idx = kept.index("PROGDATE")
        row = ["" for _ in kept]
        row[idx] = "2025-05-19 00:00:00.000"
        assert (
            normalize_row("IPDADMPROGRESS", row, positions, kept, cohort_year=2026)
            is None
        )
        assert (
            normalize_row("IPDADMPROGRESS", row, positions, kept, cohort_year=2025)
            is not None
        )


class TestNormalizeRowDateParse:
    """IPTSUMOPRT.INDATE arrives in Excel-locale long form. The row pass
    parses it to ISO via parse_iptsumoprt_date(). Parse failures attach a
    warning to the row but DO NOT drop the row — strict-parser philosophy
    per PRD §1 fix E35."""

    def _iptsumoprt_row(
        self, indate: str
    ) -> tuple[list[str], tuple[int, ...], list[str]]:
        kept = list(get_schema("IPTSUMOPRT").columns)
        positions = _row_positions("IPTSUMOPRT", kept, kept)
        row = ["" for _ in kept]
        row[kept.index("INDATE")] = indate
        return row, positions, kept

    def test_indate_long_form_replaced_with_iso(self) -> None:
        row, positions, kept = self._iptsumoprt_row("June 7, 2025, 12:00 AM")
        result = normalize_row("IPTSUMOPRT", row, positions, kept)
        assert result is not None
        assert result.parse_warnings == ()
        assert result.cells[kept.index("INDATE")] == "2025-06-07"

    def test_indate_unparseable_yields_warning_but_keeps_row(self) -> None:
        row, positions, kept = self._iptsumoprt_row("not-a-date")
        result = normalize_row("IPTSUMOPRT", row, positions, kept)
        assert result is not None  # row is NOT dropped
        # Warning attached on the INDATE column.
        assert len(result.parse_warnings) == 1
        col, msg = result.parse_warnings[0]
        assert col == "INDATE"
        assert msg is not None and msg != ""
        # Original (unparseable) cell value preserved so an auditor can
        # inspect what the export sent.
        assert result.cells[kept.index("INDATE")] == "not-a-date"

    def test_indate_empty_yields_warning(self) -> None:
        row, positions, kept = self._iptsumoprt_row("")
        result = normalize_row("IPTSUMOPRT", row, positions, kept)
        assert result is not None
        assert len(result.parse_warnings) == 1
        assert result.parse_warnings[0][0] == "INDATE"

    def test_other_tables_do_not_run_date_parse(self) -> None:
        # BDVST has BDVSTDATE in HOSxP standard format. The row pass
        # must leave it alone (no English-locale parser invoked).
        kept = list(get_schema("BDVST").columns)
        positions = _row_positions("BDVST", kept, kept)
        row = ["" for _ in kept]
        row[kept.index("BDVSTDATE")] = "2025-05-19 00:00:00.000"
        row[kept.index("REQDATE")] = "2025-05-19 00:00:00.000"
        result = normalize_row("BDVST", row, positions, kept)
        assert result is not None
        assert result.parse_warnings == ()
        # BDVSTDATE cell unchanged — not pushed through date parser.
        assert result.cells[kept.index("BDVSTDATE")] == "2025-05-19 00:00:00.000"


class TestIPDADMPROGRESSPositionalDedupeAlignment:
    """The killer test for issue #61: when raw IPDADMPROGRESS rows arrive
    with duplicate HN/AN cells, the projected row must align with the
    projected header by INDEX, not by name. Without positional dedupe at
    the row level, every cell past position 2 shifts and the SOAP fields
    read from the wrong columns."""

    def test_row_cells_align_with_projected_header_after_dedupe(self) -> None:
        # Real file shape (abbreviated): HN, AN, AN(dup), PROGDATE,
        # HN(dup), SUBJECTIVE, OBJECTIVE, ASSESSMENT, PLAN. Each cell
        # carries its column name so the test asserts which cell ends
        # up where after dedupe.
        raw_header = [
            "HN",
            "AN",
            "AN",
            "PROGDATE",
            "HN",
            "SUBJECTIVE",
            "OBJECTIVE",
            "ASSESSMENT",
            "PLAN",
        ]
        raw_row = [
            "hn-real",
            "an-real",
            "an-dup",
            "2025-05-19 00:00:00.000",
            "hn-dup",
            "subj text",
            "obj text",
            "asmnt text",
            "plan text",
        ]
        kept = list(get_schema("IPDADMPROGRESS").columns)
        # Strict containment: every kept col is in raw_header.
        for col in kept:
            assert col in raw_header
        positions = _row_positions("IPDADMPROGRESS", raw_header, kept)
        result = normalize_row("IPDADMPROGRESS", raw_row, positions, kept)
        assert result is not None
        d = dict(zip(kept, result.cells, strict=True))
        # The real HN/AN survive; the duplicates are skipped.
        assert d["HN"] == "hn-real"
        assert d["AN"] == "an-real"
        # SOAP fields read from THEIR positions, not shifted by the
        # duplicate-cell columns.
        assert d["SUBJECTIVE"] == "subj text"
        assert d["OBJECTIVE"] == "obj text"
        assert d["ASSESSMENT"] == "asmnt text"
        assert d["PLAN"] == "plan text"


class TestNormalizeRowRaggedRow:
    """A truncated CSV row (fewer cells than the header declares) must not
    crash the ingest. Missing cells are filled with empty string and a
    row-level parse warning is attached under :data:`RAGGED_ROW_WARNING_KEY`
    so the strict-parser philosophy (warn rather than drop the run) holds
    even on malformed exports."""

    def test_ragged_row_fills_missing_with_empty(self) -> None:
        kept = list(get_schema("BDVST").columns)
        positions = _row_positions("BDVST", kept, kept)
        # Build a row that's missing the last 3 cells.
        raw_row = ["v0", "v1", "v2"] + ["" for _ in range(len(kept) - 6)]
        # Length is len(kept) - 3 → last 3 columns are missing.
        assert len(raw_row) == len(kept) - 3
        result = normalize_row("BDVST", raw_row, positions, kept)
        assert result is not None
        # Output is full-width: the missing trailing cells filled with "".
        assert len(result.cells) == len(kept)
        # Tail three are empty.
        assert result.cells[-3:] == ("", "", "")
        # A row-level warning fires under the sentinel key.
        assert any(col == RAGGED_ROW_WARNING_KEY for col, _ in result.parse_warnings), (
            f"no ragged-row warning in {result.parse_warnings!r}"
        )

    def test_ragged_row_warning_names_missing_columns(self) -> None:
        kept = list(get_schema("BDVST").columns)
        positions = _row_positions("BDVST", kept, kept)
        raw_row = ["v0", "v1", "v2"]  # only 3 cells; missing everything else
        result = normalize_row("BDVST", raw_row, positions, kept)
        assert result is not None
        warnings = dict(result.parse_warnings)
        assert RAGGED_ROW_WARNING_KEY in warnings
        msg = warnings[RAGGED_ROW_WARNING_KEY]
        # Message names at least one missing column so an operator can
        # locate the export defect.
        missing_col = kept[-1]  # last column is definitely missing
        assert missing_col in msg, f"expected {missing_col!r} in warning: {msg!r}"

    def test_ragged_row_does_not_raise_index_error(self) -> None:
        # The blocker Codex flagged: raw_row[i] for i out of range used to
        # raise IndexError. This test pins the controlled-error behavior.
        kept = list(get_schema("BDVST").columns)
        positions = _row_positions("BDVST", kept, kept)
        # Pathological: empty row, positions reference cells 0..N-1.
        result = normalize_row("BDVST", [], positions, kept)
        assert result is not None  # no crash; row preserved with all "" cells

    def test_ragged_row_year_filter_still_drops_when_date_missing(self) -> None:
        # If the year-filter column happens to be one of the missing
        # cells, the empty fill ("") fails the startswith check and the
        # row is dropped — the ragged warning is lost in that case
        # (acceptable trade-off documented in normalize_row).
        kept = list(get_schema("IPDADMPROGRESS").columns)
        positions = _row_positions("IPDADMPROGRESS", kept, kept)
        # Cells stop short of the PROGDATE column.
        progdate_idx = kept.index("PROGDATE")
        raw_row = ["v"] * progdate_idx  # cell 0..progdate_idx-1 present
        assert normalize_row("IPDADMPROGRESS", raw_row, positions, kept) is None


class TestNormalizeRowsStream:
    """The streaming generator yields kept rows in input order and lets
    the caller count rows-in vs rows-yielded externally."""

    def test_yields_kept_rows_in_input_order(self) -> None:
        kept = list(get_schema("IPDADMPROGRESS").columns)
        idx = kept.index("PROGDATE")

        def make_row(progdate: str, marker: str) -> list[str]:
            row = ["" for _ in kept]
            row[idx] = progdate
            row[kept.index("OBJECTIVE")] = marker
            return row

        rows_in = [
            make_row("2025-01-01 00:00:00.000", "a"),
            make_row("2024-06-01 00:00:00.000", "b"),  # filtered
            make_row("2025-07-01 00:00:00.000", "c"),
            make_row("2023-12-31 00:00:00.000", "d"),  # filtered
            make_row("2025-12-31 00:00:00.000", "e"),
        ]
        kept_yielded = list(normalize_rows("IPDADMPROGRESS", kept, kept, iter(rows_in)))
        assert [r.cells[kept.index("OBJECTIVE")] for r in kept_yielded] == [
            "a",
            "c",
            "e",
        ]

    def test_parse_warnings_attach_per_row(self) -> None:
        kept = list(get_schema("IPTSUMOPRT").columns)
        idx = kept.index("INDATE")

        def make_row(indate: str) -> list[str]:
            row = ["" for _ in kept]
            row[idx] = indate
            return row

        rows = [
            make_row("June 7, 2025, 12:00 AM"),  # clean
            make_row("garbage"),  # warning
            make_row("January 1, 2014, 12:00 AM"),  # clean
        ]
        out = list(normalize_rows("IPTSUMOPRT", kept, kept, iter(rows)))
        assert len(out) == 3  # parse failure does NOT drop the row
        assert out[0].parse_warnings == ()
        assert len(out[1].parse_warnings) == 1
        assert out[2].parse_warnings == ()


class TestYearFilterRobustness:
    """Issue #63 (Codex advisory P2.A.3): year-filter must tolerate leading
    whitespace and UTF-8 BOM, and must reject clobbered-prefix false
    positives like ``"20259-..."`` that ``str.startswith`` would silently
    accept as the cohort year."""

    def _row_with_progdate(
        self, progdate: str
    ) -> tuple[list[str], tuple[int, ...], list[str]]:
        kept = list(get_schema("IPDADMPROGRESS").columns)
        positions = _row_positions("IPDADMPROGRESS", kept, kept)
        row = ["" for _ in kept]
        row[kept.index("PROGDATE")] = progdate
        return row, positions, kept

    def test_leading_whitespace_tolerated(self) -> None:
        # A real export occasionally pads cells with whitespace; that
        # should not silently drop the row when the year is otherwise
        # correct.
        row, positions, kept = self._row_with_progdate("  2025-05-19 00:00:00.000")
        assert normalize_row("IPDADMPROGRESS", row, positions, kept) is not None, (
            "leading whitespace must not falsely drop a cohort-year row"
        )

    def test_utf8_bom_tolerated(self) -> None:
        # A file saved with BOM keeps the marker on the first cell of the
        # first row; the year-filter must strip it.
        row, positions, kept = self._row_with_progdate("﻿2025-05-19 00:00:00.000")
        assert normalize_row("IPDADMPROGRESS", row, positions, kept) is not None, (
            "leading BOM must not falsely drop a cohort-year row"
        )

    def test_clobbered_prefix_rejected(self) -> None:
        # "20259-..." used to pass the old startswith("2025-") check
        # silently. The regex now requires exactly 4 leading digits
        # followed by a non-digit, so 5-digit prefixes are correctly
        # rejected.
        row, positions, kept = self._row_with_progdate("20259-05-19 00:00:00.000")
        assert normalize_row("IPDADMPROGRESS", row, positions, kept) is None, (
            "5-digit year prefix must NOT pass as 2025"
        )

    def test_non_year_prefix_rejected(self) -> None:
        # Non-digit prefix should reject cleanly.
        row, positions, kept = self._row_with_progdate("not-a-date")
        assert normalize_row("IPDADMPROGRESS", row, positions, kept) is None


class TestRuleOrderGuard:
    """Issue #63 (Codex advisory P2.A.4): the year-filter and date-parse
    rules currently run in fixed order (year first, parse second) on the
    *raw* cell value. No table appears in both rule dicts. If a future
    table needs both rules, the precedence must be revisited explicitly.
    Asserted at module import; this test pins the invariant."""

    def test_year_filter_and_date_parse_dicts_are_disjoint(self) -> None:
        from bba.ingest.normalize import _DATE_PARSE_COLUMN, _YEAR_FILTER_COLUMN

        overlap = set(_YEAR_FILTER_COLUMN) & set(_DATE_PARSE_COLUMN)
        assert not overlap, (
            f"_YEAR_FILTER_COLUMN and _DATE_PARSE_COLUMN must be disjoint; "
            f"overlap: {sorted(overlap)}. If a real table needs both rules, "
            f"decide the per-table precedence in normalize_row() first."
        )


class TestEmptyCSVNoLog:
    """Issue #63: a CSV with only the header (no data rows) MUST NOT emit
    a misleading per-table row-stats log line. The ``if rows_in > 0``
    guard in ``_drain_normalize_rows`` enforces this."""

    def test_header_only_csv_emits_no_row_stats_line(
        self,
        tmp_path: Path,
        complete_hosxp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Overwrite BDVST with a header-only file (no data rows). The drain
        # should observe zero rows_in and suppress the summary line.
        bdvst_path = complete_hosxp_dir / "BDVST.csv"
        kept_cols = list(get_schema("BDVST").columns)
        bdvst_path.write_text(",".join(kept_cols) + "\n", encoding="utf-8")

        cfg = IngestConfig(
            input_dir=complete_hosxp_dir,
            output_dir=tmp_path / "out",
            code_version="test-0.0.1",
        )
        with caplog.at_level("INFO", logger="bba.ingest.pipeline"):
            ingest(cfg)

        # No "table=BDVST rows_in=" line for the empty BDVST.
        # Bounded by "table=BDVST " (trailing space) so BDVSTDT/BDVSTST
        # don't false-positive against the substring search.
        misleading = [
            r
            for r in caplog.records
            if "table=BDVST " in r.message and "rows_in" in r.message
        ]
        assert misleading == [], (
            f"header-only BDVST.csv must not emit a row-stats line; got: "
            f"{[r.message for r in misleading]}"
        )


class TestFixtureDrainSmoke:
    """Issue #63: explicit smoke test that the 11 single-empty-row fixture
    tables drain cleanly through ``ingest()``. The existing partition /
    idempotency tests don't assert this, so a fixture-vs-drain regression
    would pass silently. This test pins the contract."""

    def test_all_11_tables_drain_without_crashing(
        self,
        tmp_path: Path,
        complete_hosxp_dir: Path,
    ) -> None:
        cfg = IngestConfig(
            input_dir=complete_hosxp_dir,
            output_dir=tmp_path / "out",
            code_version="test-0.0.1",
        )
        # Must not raise. The fixture's single empty row exercises every
        # per-table rule branch with degenerate inputs (empty PROGDATE
        # year-filter-drops, empty INDATE parse-warns, all other tables
        # pass through), and the row-pipeline must handle each cleanly.
        result = ingest(cfg)
        assert len(result.tables_written) == 11
        assert result.skipped_idempotent is False


class TestPipelineLogsWarningDetail:
    """Codex P2 from PR #62: the per-file drain must surface the actual
    ``(column, message)`` warning content, not just an aggregate count.
    Aggregated by column with bounded example messages so the log stays
    bounded even when every row warns."""

    def test_indate_parse_warnings_logged_per_column(
        self,
        tmp_path: Path,
        complete_hosxp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Replace IPTSUMOPRT with rows whose INDATE values are deliberately
        # broken in distinct ways. The drain must:
        #   - aggregate the count by column (INDATE)
        #   - log up to 3 example messages so the operator can diagnose
        #   - emit at WARNING level (operators filtering by severity see it)
        import csv as _csv  # local import to avoid shadowing module-scope name

        kept_cols = list(get_schema("IPTSUMOPRT").columns)
        indate_idx = kept_cols.index("INDATE")

        def cells_for(indate: str) -> list[str]:
            row = ["" for _ in kept_cols]
            row[indate_idx] = indate
            return row

        # Use csv.writer so values containing commas (like the valid
        # English-locale date "June 7, 2025, 12:00 AM") are properly
        # quoted; otherwise the commas would be re-interpreted as field
        # separators and the test would measure a different row shape.
        indates = [
            "garbage-1",
            "garbage-2",
            "not-a-date",
            "June 7, 2025, 12:00 AM",  # clean, must NOT warn
        ]
        with (complete_hosxp_dir / "IPTSUMOPRT.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = _csv.writer(fh)
            writer.writerow(kept_cols)
            for indate in indates:
                writer.writerow(cells_for(indate))

        cfg = IngestConfig(
            input_dir=complete_hosxp_dir,
            output_dir=tmp_path / "out",
            code_version="test-0.0.1",
        )
        with caplog.at_level("WARNING", logger="bba.ingest.pipeline"):
            ingest(cfg)

        # Find the IPTSUMOPRT INDATE warning log line.
        indate_logs = [
            r
            for r in caplog.records
            if r.levelname == "WARNING"
            and "IPTSUMOPRT" in r.message
            and "column=INDATE" in r.message
        ]
        assert indate_logs, (
            f"per-column parse_warning log not emitted; records: "
            f"{[(r.levelname, r.message) for r in caplog.records]}"
        )
        msg = indate_logs[0].message
        assert "count=3" in msg, f"expected count=3 in {msg!r}"
        # The 'examples=' detail must surface actual reasons so an
        # operator can diagnose which parser regression hit.
        assert "examples=" in msg


class TestPipelineDrainsRows:
    """Integration: ingest() reads rows after validate_header succeeds
    and logs per-table stats (rows_in / rows_kept / rows_filtered /
    rows_with_warnings). Phase 1 discards the drained rows; the next
    ticket replaces "discard" with Parquet write."""

    def test_year_filter_drop_count_logged(
        self,
        tmp_path: Path,
        complete_hosxp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Replace IPDADMPROGRESS with a multi-row CSV: 2 in-cohort,
        # 3 out-of-cohort. The row drain must log rows_kept=2 and
        # rows_filtered=3.
        kept_cols = list(get_schema("IPDADMPROGRESS").columns)
        idx = kept_cols.index("PROGDATE")

        def csv_row(progdate: str) -> str:
            cells = ["" for _ in kept_cols]
            cells[idx] = progdate
            return ",".join(cells)

        lines = [
            ",".join(kept_cols),
            csv_row("2025-01-01 00:00:00.000"),
            csv_row("2024-06-01 00:00:00.000"),
            csv_row("2025-07-01 00:00:00.000"),
            csv_row("2023-12-31 00:00:00.000"),
            csv_row("2026-01-01 00:00:00.000"),
        ]
        (complete_hosxp_dir / "IPDADMPROGRESS.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

        cfg = IngestConfig(
            input_dir=complete_hosxp_dir,
            output_dir=tmp_path / "out",
            code_version="test-0.0.1",
        )
        with caplog.at_level("INFO", logger="bba.ingest.pipeline"):
            ingest(cfg)

        # Find the IPDADMPROGRESS stats log.
        ipdadm_logs = [
            r
            for r in caplog.records
            if "IPDADMPROGRESS" in r.message and "rows_in" in r.message
        ]
        assert ipdadm_logs, (
            f"per-table row-stats log not emitted; records: "
            f"{[r.message for r in caplog.records]}"
        )
        msg = ipdadm_logs[0].message
        assert "rows_in=5" in msg
        assert "rows_kept=2" in msg
        assert "rows_filtered=3" in msg


# =============================================================================
# AC: All 11 CSVs ingestable — pandera schemas registered for every table
# =============================================================================


class TestSchemaCoverage:
    @pytest.mark.parametrize("table", REQUIRED_TABLES)
    def test_every_required_table_has_a_schema(self, table: CSVTable) -> None:
        schema = get_schema(table)
        assert schema is not None

    def test_all_tables_returns_canonical_set(self) -> None:
        assert tuple(all_tables()) == REQUIRED_TABLES

    def test_no_extra_tables_silently_registered(self) -> None:
        # Adding a table without bumping the schema version would change the
        # fingerprint and produce a new run_id — but we still want a hard
        # tripwire so a contributor cannot register an extra one silently.
        # Canonical set (2026-05-19 schema lock): 11 tables — see
        # docs/ingest-mapping.md. UnUSE_Patient_Background dropped because
        # the bundle's file with that name is obstetric records, not patient
        # demographics; audit no longer needs per-row age/sex.
        assert len(all_tables()) == 11


# =============================================================================
# AC: Schema-drift detection — unknown column → fail loud with helpful error
# =============================================================================


class TestValidateHeader:
    """Drift policy lives in schemas.validate_header; the interface IS the test
    surface (DEEPENING.md). Pipeline-level drift tests below confirm propagation."""

    def test_clean_header_returns_none(self) -> None:
        cols = list(get_schema("BDVST").columns)
        # Returns None implicitly when no drift; no exception raised.
        validate_header("BDVST", cols)

    def test_clean_header_order_insensitive(self) -> None:
        cols = list(reversed(list(get_schema("BDVST").columns)))
        validate_header("BDVST", cols)

    def test_unknown_column_raises(self) -> None:
        with pytest.raises(SchemaDriftError) as exc_info:
            validate_header(
                "BDVST",
                [
                    "HN",
                    "AN",
                    "REQNO",
                    "BDVSTST",
                    "REQTYPE",
                    "CANCELDATE",
                    "MYSTERY_COL",
                ],
            )
        msg = str(exc_info.value)
        assert "MYSTERY_COL" in msg
        assert "BDVST" in msg

    def test_missing_required_column_raises(self) -> None:
        # BDVST declares 6 columns; this CSV is missing REQNO.
        with pytest.raises(SchemaDriftError) as exc_info:
            validate_header("BDVST", ["HN", "AN", "BDVSTST", "REQTYPE", "CANCELDATE"])
        msg = str(exc_info.value)
        assert "REQNO" in msg
        assert "BDVST" in msg

    def test_unknown_and_missing_both_named(self) -> None:
        # BDVST: missing REQNO + has extra MYSTERY_COL.
        with pytest.raises(SchemaDriftError) as exc_info:
            validate_header(
                "BDVST", ["HN", "AN", "BDVSTST", "REQTYPE", "CANCELDATE", "MYSTERY_COL"]
            )
        msg = str(exc_info.value)
        assert "REQNO" in msg
        assert "MYSTERY_COL" in msg

    def test_empty_header_raises_with_full_missing_list(self) -> None:
        with pytest.raises(SchemaDriftError) as exc_info:
            validate_header("BDVST", [])
        # Empty header is the limit case of "missing everything".
        msg = str(exc_info.value)
        for col in get_schema("BDVST").columns:
            assert col in msg, f"column {col!r} not named in error: {msg!r}"


class TestSchemaDriftDetection:
    """Integration: drift detection (for *missing required* columns) still
    propagates through ingest() with no side-effects. Per the 2026-05-19
    schema lock (policy a; see docs/ingest-mapping.md), *unknown* extra
    columns are no longer fatal — they are projected away by the normalize
    layer and logged. The fatal direction is now only "missing required"."""

    def test_undeclared_extra_column_is_projected_not_raised(
        self, tmp_path: Path, complete_hosxp_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Start from a complete fixture and add a wholly-undeclared column
        # to BDVST. With the normalize layer wired in, ingest must NOT raise:
        # the extra is projected away and logged.
        bdvst_path = complete_hosxp_dir / "BDVST.csv"
        from bba.ingest.schemas import get_schema

        declared = list(get_schema("BDVST").columns)
        header = ",".join([*declared, "SOMETHING_NEW"])
        row = ",".join("" for _ in declared) + ",extra-value"
        bdvst_path.write_text(f"{header}\n{row}\n", encoding="utf-8")

        cfg = IngestConfig(
            input_dir=complete_hosxp_dir,
            output_dir=tmp_path / "out",
            code_version="test-0.0.1",
        )
        with caplog.at_level("INFO", logger="bba.ingest.pipeline"):
            result = ingest(cfg)
        # Ingest completes without drift.
        assert result.skipped_idempotent is False
        # The dropped column shows up in the run log so reviewers can diff.
        assert any(
            "BDVST" in rec.message and "SOMETHING_NEW" in rec.message
            for rec in caplog.records
        ), (
            f"normalize did not log the dropped column; caplog records: "
            f"{[r.message for r in caplog.records]}"
        )


# =============================================================================
# AC: All timestamps stored UTC; rendered Asia/Bangkok at boundaries
# =============================================================================


class TestRowTimestamp:
    """The deepened module: combine (date, ParsedTimeOfDay, tz) → tz-aware UTC.

    The old standalone ``to_utc`` helper had no real call site; the conversion
    only ever matters when a row's date column meets the parsed time, so the
    rule lives with that combination and tests target it there."""

    def test_bangkok_local_to_utc_subtracts_seven_hours(self) -> None:
        # Asia/Bangkok is fixed UTC+07:00 (no DST since 1920); 15:00 local → 08:00 UTC.
        ts = RowTimestamp.from_parts(
            date(2026, 5, 14), ParsedTimeOfDay(15, 0, 0), tz="Asia/Bangkok"
        )
        assert ts.utc.tzinfo is not None, "from_parts must return a tz-aware datetime"
        assert ts.utc.utcoffset() == timedelta(0), "tzinfo must be UTC"
        assert (ts.utc.year, ts.utc.month, ts.utc.day, ts.utc.hour, ts.utc.minute) == (
            2026,
            5,
            14,
            8,
            0,
        )

    def test_midnight_local_crosses_day_boundary(self) -> None:
        # Month-boundary correctness (PRD §1, fix E31): 2026-05-15 03:00 +07
        # ↦ 2026-05-14 20:00 UTC — must NOT bucket as the 15th.
        ts = RowTimestamp.from_parts(date(2026, 5, 15), ParsedTimeOfDay(3, 0, 0))
        assert (ts.utc.year, ts.utc.month, ts.utc.day, ts.utc.hour) == (
            2026,
            5,
            14,
            20,
        )

    def test_default_tz_is_bangkok(self) -> None:
        # The default arg encodes the PRD's source-zone choice; an off-default
        # caller has to be explicit.
        ts_default = RowTimestamp.from_parts(
            date(2026, 5, 14), ParsedTimeOfDay(15, 0, 0)
        )
        ts_bangkok = RowTimestamp.from_parts(
            date(2026, 5, 14), ParsedTimeOfDay(15, 0, 0), tz="Asia/Bangkok"
        )
        assert ts_default == ts_bangkok

    def test_parsed_time_of_day_round_trips_into_row_timestamp(self) -> None:
        # End-to-end: a raw HOSxP time string flows through the parser into a
        # ParsedTimeOfDay, then combines with a date column into a UTC moment.
        parsed = parse_hosxp_time("083045")
        assert isinstance(parsed.value, ParsedTimeOfDay)
        ts = RowTimestamp.from_parts(date(2026, 5, 14), parsed.value)
        # 08:30:45 +07 → 01:30:45 UTC.
        assert (ts.utc.hour, ts.utc.minute, ts.utc.second) == (1, 30, 45)

    def test_row_timestamp_is_frozen(self) -> None:
        ts = RowTimestamp.from_parts(date(2026, 5, 14), ParsedTimeOfDay(15, 0, 0))
        with pytest.raises(Exception):
            ts.utc = datetime(1970, 1, 1)  # type: ignore[misc]


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


class TestRunIdentity:
    """RunIdentity concentrates the run_id formula and the on-disk completion-marker
    convention. The interface IS the test surface (DEEPENING.md)."""

    def test_run_id_is_sha256_of_concatenated_inputs(self) -> None:
        # The concatenation order is part of the public contract (PRD §1, fix E32).
        a = "a" * 64
        b = "b" * 64
        v = "0.1.0"
        expected = hashlib.sha256((a + b + v).encode("utf-8")).hexdigest()
        assert RunIdentity.from_components(a, b, v).run_id == expected

    def test_same_inputs_same_run_id(self) -> None:
        assert (
            RunIdentity.from_components("x", "y", "z").run_id
            == RunIdentity.from_components("x", "y", "z").run_id
        )

    def test_run_id_changes_when_any_component_changes(self) -> None:
        base = RunIdentity.from_components("x", "y", "z").run_id
        assert RunIdentity.from_components("X", "y", "z").run_id != base
        assert RunIdentity.from_components("x", "Y", "z").run_id != base
        assert RunIdentity.from_components("x", "y", "Z").run_id != base

    def test_run_id_is_64_char_hex(self) -> None:
        rid = RunIdentity.from_components("x", "y", "z").run_id
        assert isinstance(rid, str) and len(rid) == 64
        int(rid, 16)  # parses as hex

    def test_from_inputs_invariant_under_dict_order(self) -> None:
        # The aggregate-hash step must sort by table; the same dict in different
        # iteration orders must yield the same run_id (cross-platform reproducibility).
        a: dict[CSVTable, str] = {"BDVST": "h1", "Lab": "h2", "Med": "h3"}
        b: dict[CSVTable, str] = {"Med": "h3", "BDVST": "h1", "Lab": "h2"}
        assert (
            RunIdentity.from_inputs(a, "schema-fp", "v1").run_id
            == RunIdentity.from_inputs(b, "schema-fp", "v1").run_id
        )

    def test_identity_is_frozen(self) -> None:
        identity = RunIdentity.from_components("x", "y", "z")
        # @dataclass(frozen=True) raises FrozenInstanceError on attribute set.
        with pytest.raises(Exception):
            identity.run_id = "other"  # type: ignore[misc]


class TestRunIdentityCompletion:
    """is_complete / mark_complete describe one on-disk protocol."""

    def test_fresh_identity_is_not_complete(self, tmp_path: Path) -> None:
        identity = RunIdentity(run_id="feedface" * 8)
        # tmp_path exists but no marker has been written.
        assert identity.is_complete(tmp_path) is False

    def test_missing_output_dir_is_not_complete(self, tmp_path: Path) -> None:
        identity = RunIdentity(run_id="feedface" * 8)
        nonexistent = tmp_path / "does-not-exist"
        assert identity.is_complete(nonexistent) is False

    def test_mark_then_is_complete(self, tmp_path: Path) -> None:
        identity = RunIdentity(run_id="cafebabe" * 8)
        out = tmp_path / "out"
        identity.mark_complete(out)
        assert identity.is_complete(out) is True

    def test_independent_identities_do_not_collide(self, tmp_path: Path) -> None:
        # Marking A complete must not mark B complete.
        a = RunIdentity(run_id="a" * 64)
        b = RunIdentity(run_id="b" * 64)
        out = tmp_path / "out"
        a.mark_complete(out)
        assert a.is_complete(out) is True
        assert b.is_complete(out) is False


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
            r.value = datetime(2026, 1, 1)  # type: ignore[misc, assignment]

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
        assert (
            "does-not-exist" in str(exc_info.value)
            or "missing" in str(exc_info.value).lower()
        )

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
