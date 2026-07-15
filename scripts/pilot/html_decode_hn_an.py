"""
Extract (CaseNumber/REQNO, HN, AN) from a review HTML and resolve PHI tokens to originals.

HMAC-SHA256 is one-way, so resolution works by forward-scanning the raw CSVs:
hash every HN/AN value with the same key, then match against the tokens in the HTML.

Usage:
    export PHI_HMAC_KEY="$(cat /Users/admin/Project_Chatbot_research/Bloodbank/data/.phi_hmac_key)"
    python scripts/pilot/html_decode_hn_an.py \\
        --html /tmp/bba_mini/review.html \\
        --raw-dir /Users/admin/Project_Chatbot_research/Bloodbank/data/raw \\
        --out /tmp/bba_mini/hn_an_table.csv

Output columns: CaseNumber (= REQNO), HN, AN
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import os
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

TOKEN_PREFIX = "PHI_"
TOKEN_LEN_HEX = 16
_ID_COLS = {"hn", "an"}

# Matches the case detail block written by build_review.py:
#   id='case-N'>
#   <h3>Case N — REQNO 12345678</h3>
#   <div><b>HN:</b> <code>PHI_...</code></div>
#   <div><b>AN:</b> <code>PHI_...</code></div>
_CASE_RE = re.compile(
    r"id='case-\d+'>"
    r".*?<h3>[^<]*REQNO\s+(\d+)</h3>"
    r".*?<b>HN:</b>\s*<code>(PHI_[0-9a-f]+)</code>"
    r".*?<b>AN:</b>\s*<code>(PHI_[0-9a-f]+)</code>",
    re.DOTALL,
)


def _load_key(raw_dir: Path) -> bytes:
    key = os.environ.get("PHI_HMAC_KEY")
    if not key:
        candidate = raw_dir.parent / ".phi_hmac_key"
        if candidate.is_file():
            key = candidate.read_text(encoding="utf-8").strip()
    if not key:
        sys.stderr.write(
            "ERROR: no HMAC key. Set PHI_HMAC_KEY or place .phi_hmac_key in the data/ folder.\n"
        )
        sys.exit(2)
    return key.encode("utf-8")


def _token_for(value: str, key: bytes) -> str:
    norm = value.strip()
    if not norm:
        return ""
    digest = hmac.new(key, norm.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{TOKEN_PREFIX}{digest[:TOKEN_LEN_HEX]}"


def _extract_cases(html: str) -> list[dict[str, str]]:
    return [
        {"reqno": m.group(1), "hn_tok": m.group(2), "an_tok": m.group(3)}
        for m in _CASE_RE.finditer(html)
    ]


def _resolve_tokens(tokens: set[str], raw_dir: Path, key: bytes) -> dict[str, str]:
    resolved: dict[str, str] = {}
    seen_raws: set[str] = set()

    # Smallest files first; stop as soon as every token is resolved.
    files = sorted(
        (p for p in raw_dir.glob("*.csv") if p.is_file()),
        key=lambda p: p.stat().st_size,
    )

    for src in files:
        if tokens <= resolved.keys():
            break
        try:
            with src.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.reader(fh)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                idxs = [
                    i
                    for i, col in enumerate(header)
                    if col is not None and col.strip().lower() in _ID_COLS
                ]
                if not idxs:
                    continue
                for row in reader:
                    for i in idxs:
                        if i >= len(row):
                            continue
                        raw = row[i].strip()
                        if not raw or raw in seen_raws:
                            continue
                        seen_raws.add(raw)
                        tok = _token_for(raw, key)
                        if tok in tokens:
                            resolved[tok] = raw
                    if tokens <= resolved.keys():
                        break
        except csv.Error as exc:
            sys.stderr.write(f"  (skipped {src.name}: {exc})\n")

    return resolved


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--html", type=Path, required=True, help="Path to review HTML file")
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/Users/admin/Project_Chatbot_research/Bloodbank/data/raw"),
        help="Folder of unencrypted raw CSVs (default: Bloodbank/data/raw)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: stdout)",
    )
    args = ap.parse_args()

    if not args.html.is_file():
        sys.stderr.write(f"ERROR: {args.html} not found\n")
        sys.exit(1)

    html = args.html.read_text(encoding="utf-8")
    cases = _extract_cases(html)
    if not cases:
        sys.stderr.write("ERROR: no cases found. Is the HTML from build_review.py?\n")
        sys.exit(1)
    sys.stderr.write(f"Cases found: {len(cases)}\n")

    all_tokens: set[str] = {c["hn_tok"] for c in cases} | {c["an_tok"] for c in cases}
    key = _load_key(args.raw_dir)
    sys.stderr.write(f"Resolving {len(all_tokens)} unique tokens...\n")

    resolved = _resolve_tokens(all_tokens, args.raw_dir, key)

    missing = all_tokens - resolved.keys()
    if missing:
        sys.stderr.write(
            f"WARNING: {len(missing)} token(s) not found in raw CSVs "
            f"(kept as-is): {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}\n"
        )
    sys.stderr.write(f"Resolved: {len(resolved)}/{len(all_tokens)}\n")

    out_fh = (
        open(args.out, "w", newline="", encoding="utf-8")  # noqa: WPS515
        if args.out
        else sys.stdout
    )
    try:
        writer = csv.writer(out_fh)
        writer.writerow(["CaseNumber", "HN", "AN"])
        for c in cases:
            writer.writerow(
                [
                    c["reqno"],
                    resolved.get(c["hn_tok"], c["hn_tok"]),
                    resolved.get(c["an_tok"], c["an_tok"]),
                ]
            )
    finally:
        if args.out:
            out_fh.close()

    if args.out:
        sys.stderr.write(f"Written: {args.out}\n")


if __name__ == "__main__":
    main()
