#!/usr/bin/env python3
"""Task B4 — the single scoring entrypoint. Everything is scored through here.

WHAT YOU NEED TO DO
-------------------
1. Drop BTC's published scoring script into ``phases/0_harness/btc_eval/``.
   Do NOT modify it. Add an ``__init__.py`` next to it.
2. Fill in ``btc_official_score()`` below (it is marked as a blocker) so it
   calls their function with their expected argument shapes.
3. Run with ``--cross-check``. Our reimplementation in ``src/metrics.py`` must
   agree with theirs to 1e-9. If it does not, OUR code is wrong — fix
   ``src/metrics.py``, never theirs.

WHY BOTH IMPLEMENTATIONS
------------------------
Theirs is authoritative but usually slow, awkward to import, and only takes
files. Ours is fast, takes in-memory dicts, and is what the sweeps call
thousands of times. Cross-checking once buys the right to use ours everywhere.

USAGE
-----
  # score a prediction file
  python phases/0_harness/evaluate.py --pred work/experiments/predictions/run-x.jsonl

  # score a full run by applying a cutoff rule on the fly
  python phases/0_harness/evaluate.py --run work/experiments/runs/run-x.jsonl \
      --cutoff ratio --alpha 0.85

  # verify our metrics == BTC's metrics
  python phases/0_harness/evaluate.py --pred <f> --cross-check

  # score and append a row to work/experiments/runs.csv
  python phases/0_harness/evaluate.py --pred <f> --log --run-id run-x --notes "bm25 b=0.5"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import cutoff as cutoff_mod  # noqa: E402
from src import exp_log, io_utils, metrics  # noqa: E402

DEFAULT_GOLD = "data/processed/queries_dev.jsonl"


def btc_official_score(pred_path: str, gold_path: str) -> float:
    """TODO(BLOCKER/phase0-B4): call BTC's published scorer.

    HOW: read their script, find the top-level function (often ``evaluate``,
    ``score``, or a ``__main__`` block). If it is only runnable as a CLI, shell
    out to it here and parse stdout — that is fine, this is called once.

    Example shape once you know theirs:

        from phases/0_harness.btc_eval import scoring
        return scoring.evaluate(pred_path, gold_path)["recall"]
    """
    raise NotImplementedError(
        "Fill in btc_official_score() with BTC's scorer before using --cross-check.\n"
        "Until then, dev numbers are unverified and must not be trusted.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred", help="prediction file {qid, predicted:[...]}")
    src.add_argument("--run", help="run file {qid, ranked:[[doc,score],...]}")
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--cutoff", default="top_k",
                    choices=["top_k", "ratio", "threshold", "gap"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--max-k", type=int, default=50)
    ap.add_argument("--diagnostics", action="store_true",
                    help="also print rank-based metrics (needs --run)")
    ap.add_argument("--cross-check", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--log", action="store_true", help="append to work/experiments/runs.csv")
    ap.add_argument("--run-id"); ap.add_argument("--notes", default="")
    a = ap.parse_args()

    gold = io_utils.load_queries(a.gold)
    qrels = io_utils.qrels(gold)

    run = None
    if a.run:
        run = io_utils.load_run(a.run)
        preds = cutoff_mod.apply_to_run(
            run, rule=a.cutoff, k=a.k, alpha=a.alpha, tau=a.tau,
            min_k=a.min_k, max_k=a.max_k)
        pred_path = a.run
    else:
        preds = io_utils.load_predictions(a.pred)
        pred_path = a.pred

    missing = set(qrels) - set(preds)
    if missing:
        # Loud, because a silently-dropped query is scored as zero and is the
        # #1 cause of "my dev score and the leaderboard disagree".
        print(f"WARNING: {len(missing)} gold queries absent from predictions "
              f"(scored as empty), e.g. {sorted(missing)[:5]}", file=sys.stderr)

    s = metrics.official(preds, qrels)
    if a.diagnostics and run is not None:
        s.update(metrics.diagnostics(run, qrels))

    if a.json:
        print(json.dumps({k: v for k, v in s.items() if k != "sort_key"},
                         indent=2, ensure_ascii=False))
    else:
        print(f"gold        : {a.gold}  ({int(s['n_queries'])} queries)")
        print(f"predictions : {pred_path}")
        if a.run:
            print(f"cutoff      : {a.cutoff} "
                  f"(k={a.k}, alpha={a.alpha}, min_k={a.min_k}, max_k={a.max_k})")
        print(f"avg set size: {s['avg_pred_size']:.2f}  "
              f"(avg relevant: {s['avg_rel_size']:.2f})")
        print("-" * 52)
        print(f"  RECALL    (primary)  : {s['primary_recall']:.4f}   [{s['averaging']}]")
        print(f"  PRECISION (tiebreak) : {s['tiebreak_precision']:.4f}")
        print("-" * 52)
        for k in ("micro_precision", "micro_recall", "micro_f1",
                  "macro_precision", "macro_recall", "macro_f1"):
            print(f"  {k:<22}: {s[k]:.4f}")
        for k, v in s.items():
            if "@" in k:
                print(f"  {k:<22}: {v:.4f}")

    if a.cross_check:
        theirs = btc_official_score(pred_path, a.gold)
        ours = s["primary_recall"]
        delta = abs(theirs - ours)
        print(f"\ncross-check: BTC={theirs:.9f}  ours={ours:.9f}  delta={delta:.2e}")
        print("AGREE" if delta < 1e-9 else
              "*** MISMATCH — fix src/metrics.py before doing any modelling ***")
        if delta >= 1e-9:
            sys.exit(1)

    if a.log:
        exp_log.log_run({
            "run_id": a.run_id or os.path.basename(pred_path).split(".")[0],
            "phase": "0", "task": "1",
            "cutoff_rule": f"{a.cutoff}:{a.alpha if a.cutoff=='ratio' else a.k}",
            "dev_P": s["tiebreak_precision"], "dev_R": s["primary_recall"],
            "dev_official": s["primary_recall"], "notes": a.notes,
        })
        print(f"\nlogged to {exp_log.LOG_PATH}")


if __name__ == "__main__":
    main()
