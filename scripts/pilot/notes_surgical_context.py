"""Prototype: deterministic notes-based surgical-context fallback.

When the operative tables (IPTSUMOPRT / IPDDCHSUMOPRT / INCPT) miss an
order's *index* surgery, the deterministic pre-op-crossmatch bypass cannot
fire and a legitimate peri-operative transfusion is mislabeled. This
prototype recovers the surgical context from nursing focus notes
(IPDNRFOCUSDT) using regexes only -- NO LLM calls -- and computes a
notes-derived hours-to-surgery that could feed the same <=72 h bypass.

Run:
    python scripts/pilot/notes_surgical_context.py            # summary + 7 target cases
    python scripts/pilot/notes_surgical_context.py --all      # every order

Reads /tmp/bba_mini (or $BBA_PILOT_WORK_DIR): report.csv + bundle/IPDNRFOCUSDT.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

csv.field_size_limit(sys.maxsize)

WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
TZ_LOCAL = timezone(timedelta(hours=7))  # Asia/Bangkok

# Notes are scanned a little wider than the report's +/-1d display window so a
# pre-op admission note that names a surgery up to 3 days out is still seen.
NOTE_LO_DAYS = 2
NOTE_HI_DAYS = 3
PREOP_CROSSMATCH_HOURS = 72.0

# STRONG surgical signals: any one is enough to assert peri-operative context.
# Curated to favour precision over recall on Thai cardiac/ortho/general notes.
_STRONG = re.compile(
    r"(ผ่าตัด|นัดมาทำ|under\s+(?:SAB|GA|GGA|spinal)|optime|post[\s-]?op"
    r"|\bEBL\b|\bRedo\b|\bORIF\b|\bTKA\b|\bTHA\b|\bPVR\b|craniotomy|laparotomy"
    r"|ห้องผ่าตัด|ไป\s*OR\b)",
    re.IGNORECASE,
)
# A Thai/AD date sitting next to a surgery cue: "...ผ่าตัด ... วันที่ 5/9/68".
_SURG_DATE = re.compile(
    r"(?:ผ่าตัด|นัดมาทำ|วันที่|date)[^0-9]{0,40}?(\d{1,2})/(\d{1,2})/(\d{2,4})",
    re.IGNORECASE,
)


def _be_to_ce(year: int) -> int:
    """Thai Buddhist-era year -> Gregorian. 68 -> 2568 BE -> 2025 CE."""
    if year < 100:
        year = 2500 + year
    if year > 2400:  # looks Buddhist-era
        return year - 543
    return year


def _parse_order_dt(raw: str) -> datetime | None:
    s = (raw or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ_LOCAL)


def _parse_note_dt(date_raw: str, time_raw: str) -> datetime | None:
    d = (date_raw or "").strip().split(" ")[0]
    try:
        y, m, day = (int(x) for x in d.split("-"))
    except ValueError:
        return None
    t = re.sub(r"[^0-9]", "", time_raw or "").ljust(6, "0")[:6]
    try:
        return datetime(
            y, m, day, int(t[0:2]), int(t[2:4]), int(t[4:6]), tzinfo=TZ_LOCAL
        )
    except ValueError:
        try:
            return datetime(y, m, day, tzinfo=TZ_LOCAL)
        except ValueError:
            return None


@dataclass(frozen=True)
class Recovery:
    reqno: str
    surgical_context: bool
    surgery_dt: datetime | None
    hours_to_surgery: float | None
    snippet: str


def _scan_notes(notes: list[dict[str, str]], order_dt: datetime) -> Recovery:
    lo = order_dt - timedelta(days=NOTE_LO_DAYS)
    hi = order_dt + timedelta(days=NOTE_HI_DAYS)
    has_ctx = False
    best_dt: datetime | None = None
    snippet = ""
    for r in notes:
        ndt = _parse_note_dt(r.get("PROGRESSDATE", ""), r.get("PROGRESSTIME", ""))
        if ndt is None or not (lo <= ndt <= hi):
            continue
        text = " ".join((r.get(c, "") or "") for c in ("FOCUS", "ACTION", "RESPONSE"))
        m = _STRONG.search(text)
        if not m:
            continue
        has_ctx = True
        if not snippet:
            i = max(0, m.start() - 20)
            snippet = re.sub(r"\s+", " ", text[i : i + 90]).strip()
        for dm in _SURG_DATE.finditer(text):
            day, mon, yr = (
                int(dm.group(1)),
                int(dm.group(2)),
                _be_to_ce(int(dm.group(3))),
            )
            try:
                cand = datetime(yr, mon, day, tzinfo=TZ_LOCAL)
            except ValueError:
                continue
            # accept a surgery date within a sane window around the order
            if abs((cand.date() - order_dt.date()).days) <= 14:
                # prefer the date nearest to (and not long before) the order
                if best_dt is None or abs(cand - order_dt) < abs(best_dt - order_dt):
                    best_dt = cand
    hrs = (best_dt - order_dt).total_seconds() / 3600.0 if best_dt else None
    return Recovery(notes[0].get("__reqno", ""), has_ctx, best_dt, hrs, snippet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="print every order")
    args = ap.parse_args()

    rows = list(csv.DictReader((WORK / "report.csv").open(encoding="utf-8")))
    notes_by_an: dict[str, list[dict[str, str]]] = {}
    with (WORK / "bundle" / "IPDNRFOCUSDT.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            an = (r.get("AN") or "").strip()
            if an:
                notes_by_an.setdefault(an, []).append(r)

    targets = {  # the 7 cases whose AN procedure exists but is out of bypass window
        "68058479",
        "68054834",
        "68018770",
        "68009966",
        "68055282",
        "68068034",
        "68035246",
    }

    def num(x: str) -> float | None:
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    recovered = ctx_total = within_proc = 0
    target_hits = 0
    print(
        f"{'reqno':<10}{'cls':<26}{'proc_in_win':<12}{'ctx':<5}{'surg_hrs':<10}snippet"
    )
    print("-" * 110)
    for r in rows:
        reqno = r.get("reqno", "")
        an = (r.get("an") or "").strip()
        order_dt = _parse_order_dt(r.get("order_datetime_utc", ""))
        prox, upc = (
            num(r.get("procedure_proximity_hours", "")),
            num(r.get("upcoming_procedure_hours", "")),
        )
        proc_in_win = (prox is not None and prox <= 6.0) or (
            upc is not None and upc <= PREOP_CROSSMATCH_HOURS
        )
        if proc_in_win:
            within_proc += 1
        notes = notes_by_an.get(an, [])
        if order_dt is None or not notes:
            continue
        rec = _scan_notes(notes, order_dt)
        if rec.surgical_context:
            ctx_total += 1
        # "recovery" = no in-window procedure, but notes give a <=72h surgery
        is_recovery = (
            not proc_in_win
            and rec.hours_to_surgery is not None
            and -6.0 <= rec.hours_to_surgery <= PREOP_CROSSMATCH_HOURS
        )
        if is_recovery:
            recovered += 1
        if reqno in targets and is_recovery:
            target_hits += 1
        if args.all or reqno in targets:
            hrs = (
                f"{rec.hours_to_surgery:.1f}"
                if rec.hours_to_surgery is not None
                else "-"
            )
            print(
                f"{reqno:<10}{r.get('classification', ''):<26}"
                f"{'Y' if proc_in_win else 'n':<12}{'Y' if rec.surgical_context else 'n':<5}"
                f"{hrs:<10}{rec.snippet[:50]}"
            )

    print("-" * 110)
    print(f"orders with an in-window procedure (bypass already works): {within_proc}")
    print(f"orders with notes-derived surgical context:                {ctx_total}")
    print(
        f"orders RECOVERED (no proc in window, notes give <=72h surgery): {recovered}"
    )
    print(
        f"of the 7 known out-of-window-procedure cases, recovered:    {target_hits}/7"
    )


if __name__ == "__main__":
    main()
