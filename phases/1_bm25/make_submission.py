#!/usr/bin/env python3
"""Task B5 — turn a run into a Codabench submission, and log it.

WHAT YOU NEED TO DO
-------------------
1. The format is confirmed against BTC's task document and scoring code — see
   ``format_submission()`` below. Nothing to fill in.
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

Produces submissions/<run_id>.zip containing submission.json:
    {"147194": {"answer": ["177504", "740"]}, ...}
At most 5 ids per question — more scores ZERO for that question.
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
    """BTC's required format, confirmed against their Task 1 overview document
    and their scoring program (`btc_eval/scoring_legalir.py`, which does
    ``y_pred = {k: v['answer'] for k, v in y_pred.items()}``).

        {
          "147194": {"answer": ["177504", "740"]},
          "147195": {"answer": ["12"]}
        }

    A JSON OBJECT keyed by question id — not a list of records. Every question
    in the reference must be present: their scorer raises if the key counts
    differ, so an incomplete file fails outright rather than scoring poorly.
    """
    return {q["qid"]: {"answer": list(preds.get(q["qid"], []))} for q in queries}


# Confirmed: submission.zip containing exactly submission.json.
SUBMISSION_FILENAME = "submission.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--queries", default="data/processed/queries_public_test.jsonl")
    # Default top_k/5, not ratio/10. 92% of questions have exactly ONE gold
    # document, recall is the primary metric, and recall never decreases with
    # more ids up to BTC's cap of 5 — so 5 is optimal and k>5 zeroes the
    # question. See docs/reference/10_data_facts.md §2.
    ap.add_argument("--cutoff", default="top_k",
                    choices=["top_k", "ratio", "threshold", "gap"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--max-k", type=int, default=5,
                    help="BTC hard cap: >5 documents scores ZERO for that question")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-dir", default="work/submissions")
    ap.add_argument("--notes", default="")
    ap.add_argument("--force", action="store_true",
                    help="write the file even if pre-flight found problems")
    a = ap.parse_args()

    run = io_utils.load_run(a.run)
    queries = io_utils.load_queries(a.queries)
    preds = cutoff_mod.apply_to_run(run, rule=a.cutoff, k=a.k, alpha=a.alpha,
                                    min_k=a.min_k, max_k=a.max_k)

    # --- pre-flight. These are ERRORS, not warnings: BTC's scorer raises on a
    # malformed submission, so an invalid file wastes one of ten daily attempts.
    qids = [q["qid"] for q in queries]
    payload_preds = {q: list(preds.get(q, [])) for q in qids}
    problems = metrics.check_submittable(payload_preds, qids)
    chunky = [d for v in payload_preds.values() for d in v[:1]
              if "#" in str(d) or "::" in str(d)]
    if chunky:
        problems.append(f"predictions contain chunk-style ids (e.g. {chunky[:3]}) — "
                        f"you forgot --aggregate max when retrieving; BTC has "
                        f"never seen these ids and every one is a miss")
    for p in problems:
        print(f"PRE-FLIGHT ERROR: {p}", file=sys.stderr)
    if problems and not a.force:
        print("\nRefusing to write an invalid submission. Fix the above, or pass "
              "--force if you know better.", file=sys.stderr)
        sys.exit(1)

    run_id = a.run_id or os.path.basename(a.run).replace(".jsonl", "")
    os.makedirs(a.out_dir, exist_ok=True)
    json_path = os.path.join(a.out_dir, f"{run_id}.json")
    zip_path = os.path.join(a.out_dir, f"{run_id}.zip")

    payload = format_submission(payload_preds, queries)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, SUBMISSION_FILENAME)

    # size stats must be over the queries BEING SUBMITTED, not over everything in
    # the run file — otherwise a run built on a different split reports healthy
    # numbers while most of the submission is empty.
    sizes = [len(payload_preds[q]) for q in qids]
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
