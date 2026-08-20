#!/usr/bin/env python3
"""Task B4 — execute the frozen pipeline. Private Test, 19-23/09.

THE RULE
--------
No modelling changes. Not a re-tuned cutoff, not a different checkpoint, not
"just one more alpha". The Public Test score is your estimate of the Private Test
score only if the pipeline is identical.

The ONLY permitted edits are ones that make the frozen pipeline RUN on the
private data — a schema difference, an unexpected field name, an encoding issue.
Log every one of them in `phases/5_freeze/freeze_checklist.md` §7; they belong in
the paper's limitations section.

WHAT IT DOES
------------
Reads `work/configs/FINAL.yaml`, replays the recorded pipeline stages in order against
a new query file, and refuses to run if the environment has drifted from what was
frozen.

USAGE
  python phases/5_freeze/run_frozen.py --config work/configs/FINAL.yaml \
      --queries data/processed/queries_private_test.jsonl
  python phases/5_freeze/run_frozen.py --config work/configs/FINAL.yaml --check-only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import config  # noqa: E402

from freeze_pipeline import env_fingerprint  # noqa: E402


def check_env(frozen_env):
    """Compare the current environment against the frozen fingerprint."""
    now = env_fingerprint()
    drift = []
    for pkg, ver in (frozen_env.get("packages") or {}).items():
        cur = (now.get("packages") or {}).get(pkg)
        if ver and cur and ver != cur:
            drift.append(f"{pkg}: frozen {ver} -> now {cur}")
    if frozen_env.get("python") != now.get("python"):
        drift.append(f"python: frozen {frozen_env.get('python')} -> "
                     f"now {now.get('python')}")
    if frozen_env.get("git_commit") and \
            frozen_env["git_commit"] != now.get("git_commit"):
        drift.append(f"git: frozen {frozen_env['git_commit'][:8]} -> "
                     f"now {(now.get('git_commit') or '?')[:8]}")
    return drift


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="work/configs/FINAL.yaml")
    ap.add_argument("--queries", default="data/processed/queries_private_test.jsonl")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="proceed despite environment drift (log it in the checklist)")
    a = ap.parse_args()

    cfg = config.load(a.config)
    run_id = cfg.get("run_id", "final")
    drift = check_env(cfg.get("environment", {}))

    print(f"frozen run : {run_id}")
    print(f"queries    : {a.queries}")
    if drift:
        print("\nENVIRONMENT DRIFT DETECTED:")
        for d in drift:
            print(f"  - {d}")
        print("\nThe frozen pipeline was validated against the OTHER versions.\n"
              "Reinstall from requirements-frozen.txt, or re-run with --force and\n"
              "record the drift in phases/5_freeze/freeze_checklist.md §7.")
        if not a.force and not a.check_only:
            sys.exit(1)
    else:
        print("environment matches the frozen fingerprint")

    if a.check_only:
        return

    stages = (cfg.get("pipeline") or {}).get("stages")
    if not stages:
        print("\nNo `pipeline.stages` recorded in the config.\n"
              "Add the exact commands, in order, as a list of strings under\n"
              "`pipeline.stages` in work/configs/FINAL.yaml — the same commands that\n"
              "produced the Public Test submission, with the query path\n"
              "parameterised as {queries}. Example:\n\n"
              "pipeline:\n"
              "  stages:\n"
              "    - python phases/1_bm25/bm25_baseline.py --queries {queries} ...\n"
              "    - python phases/3_rerank/rerank.py --run ... \n"
              "    - python phases/1_bm25/make_submission.py --run ... --queries {queries}\n")
        sys.exit(1)

    for i, stage in enumerate(stages, 1):
        cmd = stage.format(queries=a.queries, run_id=run_id)
        print(f"\n=== stage {i}/{len(stages)} ===\n$ {cmd}")
        r = subprocess.run(cmd, shell=True)
        if r.returncode != 0:
            print(f"\nSTAGE {i} FAILED (exit {r.returncode}).\n"
                  "Permitted fixes: whatever makes the FROZEN pipeline run on the\n"
                  "private data (schema, field names, encoding). NOT model changes.\n"
                  "Record what you changed in freeze_checklist.md §7.")
            sys.exit(r.returncode)

    print(f"\nfrozen pipeline complete for {run_id}")
    print("Submit through the registered Organization and verify it is recorded valid.")


if __name__ == "__main__":
    main()
