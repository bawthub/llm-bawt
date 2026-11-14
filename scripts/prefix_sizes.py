#!/usr/bin/env python3
"""Turn a PROXY_CACHE_DIAG turn dump into a per-part size table.

The proxy's cache_diag writes one file per turn (``<key>_turn001.txt``) whose
lines are ``<label>\t<canonical_json>``, in cache order:

    instructions
    tools[0]:Bash
    tools[1]:Read
    ...
    input[0]:user
    ...

For a FRESH session (no message history) the only input item is the user's
first turn, so everything else in the file is the *static prefix* — exactly the
thing that costs ~32k before the conversation starts.

Usage:
    # 1. point the proxy at a dump dir and restart it
    PROXY_CACHE_DIAG=/tmp/cachediag  <start the proxy / claude-code backend>
    # 2. send ONE message to a fresh session
    # 3. analyze the baseline turn
    python scripts/prefix_sizes.py /tmp/cachediag/<key>_turn001.txt

Token estimate is chars/CHARS_PER_TOK (rough; JSON+English). For exact Anthropic
counts, feed the text to the Anthropic messages count_tokens endpoint.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

CHARS_PER_TOK = 3.7  # rough heuristic for mixed English + JSON schema


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if "-h" in argv or "--help" in argv else 2

    path = Path(argv[1])
    if not path.is_file():
        print(f"no such dump file: {path}", file=sys.stderr)
        return 1

    rows: list[tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        label, text = line.split("\t", 1)
        rows.append((label, len(text)))

    # Group: instructions / tools / input
    groups: dict[str, int] = defaultdict(int)
    for label, chars in rows:
        bucket = label.split("[", 1)[0].split(":", 1)[0]  # instructions|tools|input
        groups[bucket] += chars

    total_chars = sum(c for _, c in rows)

    def est(chars: int) -> int:
        return round(chars / CHARS_PER_TOK)

    print(f"\n{path.name}\n")
    print(f"{'part':<34}{'chars':>10}{'~tokens':>10}{'%':>7}")
    print("-" * 61)
    for label, chars in sorted(rows, key=lambda r: -r[1]):
        pct = 100 * chars / total_chars if total_chars else 0
        print(f"{label:<34}{chars:>10}{est(chars):>10}{pct:>6.1f}%")

    print("-" * 61)
    print("GROUP TOTALS")
    for bucket in ("instructions", "tools", "input"):
        if bucket in groups:
            chars = groups[bucket]
            pct = 100 * chars / total_chars if total_chars else 0
            print(f"{bucket:<34}{chars:>10}{est(chars):>10}{pct:>6.1f}%")
    print("-" * 61)
    print(f"{'TOTAL':<34}{total_chars:>10}{est(total_chars):>10}{100.0:>6.1f}%")
    print(f"\n(estimate at {CHARS_PER_TOK} chars/token; exact = count_tokens API)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
