"""Resolve specific PHI_* tokens back to raw HN/AN (LOCAL USE ONLY).

The pipeline pseudonymizes HN/AN with HMAC-SHA256 (encrypt_phi.py): the token
is PHI_<first 16 hex of HMAC-SHA256(key, value.strip())>. That truncated hash
is one-way, so the only way back is a forward scan: hash every raw HN/AN and
match against the token(s) you ask for.

This prints REAL PHI. Run it in your own terminal, never through a tool that
logs output to a shared transcript. It only emits the tokens you pass on the
command line -- no bulk dump.

Usage:
    export PHI_HMAC_KEY="$(cat /Users/admin/Project_Chatbot_research/Bloodbank/data/.phi_hmac_key)"
    python scripts/pilot/reverse_lookup_phi.py \
        --raw-dir /Users/admin/Project_Chatbot_research/Bloodbank/data/raw \
        PHI_f9780522f2fab7a5 PHI_6f9fd68c6651f692

If PHI_HMAC_KEY is unset it falls back to reading the .phi_hmac_key file next
to the raw dir's parent (data/.phi_hmac_key).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import os
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

TOKEN_PREFIX = "PHI_"
TOKEN_LEN_HEX = 16
ID_COLUMNS = {"hn", "an"}  # case-insensitive, matches encrypt_phi.py defaults


def load_key(raw_dir: Path) -> bytes:
    key = os.environ.get("PHI_HMAC_KEY")
    if not key:
        candidate = raw_dir.parent / ".phi_hmac_key"
        if candidate.is_file():
            key = candidate.read_text(encoding="utf-8").strip()
    if not key:
        sys.stderr.write(
            "ERROR: no key. Set PHI_HMAC_KEY or place .phi_hmac_key in the data/ folder.\n"
        )
        sys.exit(2)
    return key.encode("utf-8")


def token_for(value: str, key: bytes) -> str:
    norm = value.strip()
    if not norm:
        return ""
    digest = hmac.new(key, norm.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{TOKEN_PREFIX}{digest[:TOKEN_LEN_HEX]}"


def id_column_indices(header: list[str]) -> list[int]:
    return [
        i
        for i, name in enumerate(header)
        if name is not None and name.strip().lower() in ID_COLUMNS
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tokens", nargs="+", help="PHI_* token(s) to resolve")
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/Users/admin/Project_Chatbot_research/Bloodbank/data/raw"),
        help="Folder of UNENCRYPTED raw CSVs.",
    )
    args = ap.parse_args()

    targets = {t.strip() for t in args.tokens if t.strip()}
    if not targets:
        sys.stderr.write("ERROR: no tokens given.\n")
        sys.exit(1)

    key = load_key(args.raw_dir)

    # Smallest files first; stop once every requested token is resolved.
    files = sorted(
        (p for p in args.raw_dir.glob("*.csv") if p.is_file()),
        key=lambda p: p.stat().st_size,
    )

    resolved: dict[str, str] = {}
    seen: set[str] = set()  # raw values already hashed (dedupe across files/cols)

    for src in files:
        if targets <= resolved.keys():
            break
        try:
            with src.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.reader(fh)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                idxs = id_column_indices(header)
                if not idxs:
                    continue
                for row in reader:
                    for i in idxs:
                        if i >= len(row):
                            continue
                        raw = row[i].strip()
                        if not raw or raw in seen:
                            continue
                        seen.add(raw)
                        tok = token_for(raw, key)
                        if tok in targets and tok not in resolved:
                            resolved[tok] = raw
                    if targets <= resolved.keys():
                        break
        except csv.Error as exc:
            sys.stderr.write(f"  (skipped {src.name}: CSV error {exc})\n")
            continue

    print("-" * 50)
    for tok in args.tokens:
        print(f"{tok}  ->  {resolved.get(tok.strip(), '(NOT FOUND in raw CSVs)')}")
    print("-" * 50)


if __name__ == "__main__":
    main()
