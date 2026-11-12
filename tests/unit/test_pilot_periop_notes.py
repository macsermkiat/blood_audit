"""Doctor progress notes carry the time they were written, not midnight.

Platelet order 68000711 (count 37,000 /uL; acute-on-chronic SDH on CT; NeuroSx
"keep Plt > 100,000") was judged INAPPROPRIATE because "no clinical note
describes an SDH". Both doctor notes from the day before said so, at 14:09 and
19:19. The loader stamped every IPDADMPROGRESS note at 00:00 of PROGDATE, the
evidence bundle keeps progress notes within 24 h of the order, and the order
was placed at 10:11, so a note stamped at the previous midnight looked 34 h
old and was dropped. On the hematology run 78 of 364 LLM-judged orders lost a
doctor note that way, and no previous-day doctor note ever reached the model.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
if str(PILOT_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_DIR))

from _periop_notes import vitals_notes_for  # noqa: E402

from bba.evidence_bundle_builder.builder import _filter_progress  # noqa: E402
from bba.evidence_bundle_builder.models import ProgressNote  # noqa: E402

_ORDER_UTC = datetime(2025, 1, 4, 3, 11, 28, tzinfo=UTC)  # 10:11 Bangkok


def _progress(progdate: str, firstdate: str | None) -> dict[str, str]:
    row = {
        "AN": "AN1",
        "PROGDATE": f"{progdate} 00:00:00.000",
        "ASSESSMENT": "acute on top chronic SDH; NeuroSx: keep Plt > 100,000",
    }
    if firstdate is not None:
        row["FIRSTDATE"] = firstdate
    return row


def _timestamp(progdate: str, firstdate: str | None) -> datetime:
    (note,) = vitals_notes_for([_progress(progdate, firstdate)], [], "AN1", _ORDER_UTC)
    return note.timestamp


def test_note_is_stamped_with_the_time_the_doctor_wrote_it() -> None:
    # 14:09:50 Bangkok = 07:09:50 UTC.
    assert _timestamp("2025-01-03", "2025-01-03 14:09:50.000") == datetime(
        2025, 1, 3, 7, 9, 50, tzinfo=UTC
    )


def test_previous_day_note_reaches_the_evidence_window() -> None:
    # The regression itself: written 20 h before the order, so inside the
    # bundle's 24 h progress window. Stamped at midnight it was 34 h old.
    written = _timestamp("2025-01-03", "2025-01-03 14:09:50.000")
    note = ProgressNote(timestamp=written, text="acute on top chronic SDH")

    assert _filter_progress((note,), _ORDER_UTC) == (note,)


def test_note_written_the_morning_after_keeps_its_entry_time() -> None:
    # Charted for 3 January, typed at 08:00 on the 4th: that is when the
    # information existed. 2% of real notes look like this.
    assert _timestamp("2025-01-03", "2025-01-04 08:00:00.000") == datetime(
        2025, 1, 4, 1, 0, 0, tzinfo=UTC
    )


def test_backfilled_note_is_placed_at_the_end_of_its_clinical_day() -> None:
    # Entered 5 days late (3.5% of real notes are 2+ days off): the entry time
    # says nothing about the clinical day. End of PROGDATE keeps the note
    # visible to a next-day order, which midnight did not.
    assert _timestamp("2025-01-03", "2025-01-08 09:00:00.000") == datetime(
        2025, 1, 3, 16, 59, 59, tzinfo=UTC
    )


def test_entry_time_before_the_clinical_date_is_not_trusted() -> None:
    assert _timestamp("2025-01-03", "2025-01-02 09:00:00.000") == datetime(
        2025, 1, 3, 16, 59, 59, tzinfo=UTC
    )


def test_missing_or_garbled_entry_time_falls_back_to_the_clinical_day() -> None:
    end_of_day = datetime(2025, 1, 3, 16, 59, 59, tzinfo=UTC)
    assert _timestamp("2025-01-03", None) == end_of_day
    assert _timestamp("2025-01-03", "not a date") == end_of_day


def test_nursing_focus_notes_keep_their_own_timestamp() -> None:
    focus = {
        "AN": "AN1",
        "PROGRESSDATE": "2025-01-03 00:00:00.000",
        "PROGRESSTIME": "141500",
        "ACTION": "observe neuro signs",
    }
    (note,) = vitals_notes_for([], [focus], "AN1", _ORDER_UTC)

    assert note.timestamp == datetime(2025, 1, 3, 7, 15, 0, tzinfo=UTC)
