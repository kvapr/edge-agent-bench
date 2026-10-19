#!/usr/bin/env python3
"""Scan a directory tree for strings that must not be published.

The patterns are read from a private file that is deliberately not part of the
repository (default ~/edge-agent-bench-private/leak-patterns.txt, or the path in
EAB_LEAK_PATTERNS): one case-insensitive regular expression per line, '#'
comments allowed. Exits 1 if anything matches, so it can gate a release.

Usage: python3 scripts/leak_scan.py <dir> [--include-git]
"""
import os
import re
import sys

SKIP_DIRS = {".git", "__pycache__"}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = sys.argv[1]
    skip = set() if "--include-git" in sys.argv else SKIP_DIRS
    pfile = os.environ.get("EAB_LEAK_PATTERNS",
                           os.path.expanduser("~/edge-agent-bench-private/leak-patterns.txt"))
    pats = [re.compile(line.strip(), re.I) for line in open(pfile)
            if line.strip() and not line.lstrip().startswith("#")]
    hits = 0
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in files:
            path = os.path.join(dirpath, fn)
            try:
                with open(path, encoding="utf-8") as f:
                    for n, line in enumerate(f, 1):
                        for p in pats:
                            if p.search(line):
                                hits += 1
                                print(f"{os.path.relpath(path, root)}:{n}: matches {p.pattern!r}")
                                break
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable file
    print(f"{hits} match(es) across {len(pats)} patterns" if hits else f"clean ({len(pats)} patterns)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
