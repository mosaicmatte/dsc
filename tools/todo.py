#!/usr/bin/env python3
"""List every outstanding TODO in the repo, grouped by severity and phase.

THE TAXONOMY
------------
  TODO(BLOCKER/<phase>-<task>)  Work that only YOU can do, because it depends on
                                BTC's data, evaluation code, or model cards.
                                Until it is resolved, the affected output is
                                either wrong or unverified. These are hard gates.

  TODO(TEAM/<phase>-<task>)     Analysis and writing the team must produce:
                                worksheets, interpretations, the paper. Nothing
                                crashes without them; the paper cannot be written
                                with them missing.

  TODO(YOU/<phase>)             A place to WRITE YOUR OWN CODE. Each one says what
                                to write, gives ideas, and gives the exact command
                                to test it. Start here if you are new.

  TODO(OPTIONAL/...)            Improvements worth doing if time allows.

Anything left as a bare `TODO` without a category is listed as UNCATEGORISED so
it cannot hide.

USAGE
  python tools/todo.py                 # everything, grouped
  python tools/todo.py --yours         # places to write your own code
  python tools/todo.py --blockers      # only the hard gates
  python tools/todo.py --phase 0       # only phase 0
  python tools/todo.py --count         # one line, for a status check
"""
from __future__ import annotations

import argparse
import os
import re
import sys

PATTERN = re.compile(r"TODO\(([A-Z]+)(?:/([a-zA-Z0-9\-]+))?\)\s*:?\s*(.*)")
# `(?!\.md)` so links to TODO.md are not reported as stray markers
BARE = re.compile(r"TODO(?!\()(?!\.md)")
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "dist", "data"}
SKIP_FILES = {"todo.py", "TODO.md"}
EXTS = {".py", ".md", ".yaml", ".yml", ".sh"}
ORDER = ["YOU", "BLOCKER", "TEAM", "OPTIONAL", "UNCATEGORISED"]


def scan(root="."):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1] not in EXTS or fn in SKIP_FILES:
                continue
            path = os.path.relpath(os.path.join(dirpath, fn), root)
            try:
                lines = open(os.path.join(dirpath, fn), encoding="utf-8").read().split("\n")
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(lines, 1):
                m = PATTERN.search(line)
                if m:
                    kind, phase, text = m.group(1), m.group(2) or "-", m.group(3)
                    if kind not in ORDER:
                        kind = "UNCATEGORISED"
                    found.append((kind, phase, path, i, text.strip(" *-:")))
                elif BARE.search(line) and "TODO" in line:
                    found.append(("UNCATEGORISED", "-", path, i,
                                  line.strip().lstrip("#/ *")[:90]))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blockers", action="store_true")
    ap.add_argument("--yours", action="store_true",
                    help="only the places where you write your own code")
    ap.add_argument("--phase", default=None)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = ap.parse_args()

    items = scan(a.root)
    if a.blockers:
        items = [i for i in items if i[0] == "BLOCKER"]
    if a.yours:
        items = [i for i in items if i[0] == "YOU"]
    if a.phase:
        items = [i for i in items if a.phase in i[1]]

    if a.count:
        n = {k: sum(1 for i in items if i[0] == k) for k in ORDER}
        print(" · ".join(f"{k}: {v}" for k, v in n.items() if v))
        sys.exit(1 if n.get("BLOCKER") else 0)

    for kind in ORDER:
        group = [i for i in items if i[0] == kind]
        if not group:
            continue
        print(f"\n{'='*78}\n{kind}  ({len(group)})")
        if kind == "YOU":
            print("write your own code here — each block says what to write, "
                  "gives ideas, and gives the command to test it")
        elif kind == "BLOCKER":
            print("only you can do these — they depend on BTC data, code or model cards")
        elif kind == "TEAM":
            print("analysis and writing — the paper cannot be assembled without them")
        print("=" * 78)
        for _, phase, path, line, text in sorted(group, key=lambda x: (x[1], x[2])):
            print(f"  [{phase:<12}] {path}:{line}")
            if text:
                print(f"                 {text[:88]}")

    nb = sum(1 for i in items if i[0] == "BLOCKER")
    print(f"\n{len(items)} open items · {nb} blockers")
    if nb:
        print("\nA BLOCKER left open means the affected numbers are unverified.\n"
              "Do not put an unverified number in the paper or on the leaderboard.")


if __name__ == "__main__":
    main()
