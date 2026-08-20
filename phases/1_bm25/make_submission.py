#!/usr/bin/env python3
"""Task B5 — turn a run into a Codabench submission, and log it.

WHAT YOU NEED TO DO
-------------------
1. Fill in ``format_submission()`` below from `phases/0_harness/01_schema_summary.md`
   section 4. It is marked as a blocker. Guessing the format wastes a submission
   from a budget of ten per day.
2. Run it against the PUBLIC TEST queries, not dev:
     --queries data/processed/queries_public_test.jsonl
3. Submit through the **registered Organization**. A submission from a personal
   account does not count.
4. Record the leaderboard score the moment it appears:
     python -c "from src.exp_log import update_leaderboard as u; u('<run_id>', 0.xxxx)"
5. After >=3 submissions, check the gate:
     python -c "from src.exp_log import correlation as c; print(c())"

THE GATE
--------
If dev and leaderboard do not move together, STOP MODELLING and fix the harness.
Usual causes, in order of frequency:
  * doc_ids in the submission do not match BTC's corpus ids (chunk ids leaked out
    without ``aggregate_to_parent``)
  * queries missing from the submission (scored as zero)
  * wrong averaging in ``src/metrics.OFFICIAL_AVERAGING``
  * dev leakage — dev queries also present in the training data

USAGE
  python phases/1_bm25/make_submission.py --run work/experiments/runs/<id>.jsonl \
      --queries data/processed/queries_public_test.jsonl --cutoff ratio --alpha 0.85
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import cutoff as cutoff_mod  # noqa: E402
from src import exp_log, io_utils, metrics  # noqa: E402


def format_submission(preds, queries):
    """TODO(BLOCKER/phase0-B1): match BTC's exact required format.

    HOW: copy the format block from `phases/0_harness/01_schema_summary.md` §4.
    The placeholder below is the most common shape in Vietnamese legal-IR shared
    tasks, but DO NOT trust it — verify against their docs.

    Things that are usually load-bearing and easy to get wrong:
      * the key name for the id list (``relevant_id`` vs ``predicted`` vs ``labels``)
      * whether every query must appear, even with an empty list
      * whether ids must be strings or integers
      * whether the file must be named exactly ``predict.json`` inside the zip
    """
    return [{"question_id": q["qid"], "relevant_id": preds.get(q["qid"], [])}
            for q in queries]


# TODO(BLOCKER/phase0-B1): confirm the exact filename BTC expects inside the zip.
SUBMISSION_FILENAME = "predict.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--queries", default="data/processed/queries_public_test.jsonl")
    ap.add_argument("--cutoff", default="ratio",
                    choices=["top_k", "ratio", "threshold", "gap"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--max-k", type=int, default=50)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-dir", default="work/submissions")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    run = io_utils.load_run(a.run)
    queries = io_utils.load_queries(a.queries)
    preds = cutoff_mod.apply_to_run(run, rule=a.cutoff, k=a.k, alpha=a.alpha,
                                    min_k=a.min_k, max_k=a.max_k)

    # --- pre-flight checks. Each of these has cost somebody a submission. ---
    problems = []
    missing = [q["qid"] for q in queries if q["qid"] not in preds]
    if missing:
        problems.append(f"{len(missing)} queries have no prediction "
                        f"(e.g. {missing[:5]}) — they will score zero recall")
    empty = [q["qid"] for q in queries if not preds.get(q["qid"])]
    if empty:
        problems.append(f"{len(empty)} queries have an EMPTY prediction "
                        f"(e.g. {empty[:5]}) — raise --min-k")
    chunky = [d for v in preds.values() for d in v[:1] if "#" in d or "::" in d]
    if chunky:
        problems.append(f"predictions contain chunk-style ids (e.g. {chunky[:3]}) — "
                        f"did you forget --aggregate max when retrieving?")
    for p in problems:
        print(f"PRE-FLIGHT WARNING: {p}", file=sys.stderr)

    run_id = a.run_id or os.path.basename(a.run).replace(".jsonl", "")
    os.makedirs(a.out_dir, exist_ok=True)
    json_path = os.path.join(a.out_dir, f"{run_id}.json")
    zip_path = os.path.join(a.out_dir, f"{run_id}.zip")

    payload = format_submission(preds, queries)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, SUBMISSION_FILENAME)

    # size stats must be over the queries BEING SUBMITTED, not over everything in
    # the run file — otherwise a run built on a different split reports healthy
    # numbers while most of the submission is empty.
    sizes = [len(preds.get(q["qid"], [])) for q in queries]
    print(f"\nrun_id       : {run_id}")
    print(f"queries      : {len(queries)}")
    print(f"cutoff       : {a.cutoff} "
          f"({'alpha='+str(a.alpha) if a.cutoff=='ratio' else 'k='+str(a.k)})")
    print(f"answer sets  : mean {sum(sizes)/max(len(sizes),1):.2f}, "
          f"min {min(sizes, default=0)}, max {max(sizes, default=0)}")
    print(f"wrote        : {json_path}\n               {zip_path}")

    # if the queries happen to be labelled (dev), report the expected score
    if any(q["relevant"] for q in queries):
        s = metrics.official(preds, io_utils.qrels(queries))
        print(f"\n(labelled input — expected R={s['primary_recall']:.4f} "
              f"P={s['tiebreak_precision']:.4f})")

    exp_log.log_run({
        "run_id": run_id, "phase": "1", "task": "1",
        "cutoff_rule": f"{a.cutoff}:{a.alpha if a.cutoff=='ratio' else a.k}",
        "leaderboard": "PENDING",
        "notes": f"SUBMITTED {os.path.basename(zip_path)}; {a.notes}",
    })
    print(f"\nlogged as PENDING. After the leaderboard updates, run:\n"
          f'  python -c "from src.exp_log import update_leaderboard as u; '
          f"u('{run_id}', 0.xxxx)\"")
    print("Submit through the REGISTERED ORGANIZATION, and verify it is "
          "recorded as valid.")


if __name__ == "__main__":
    main()
