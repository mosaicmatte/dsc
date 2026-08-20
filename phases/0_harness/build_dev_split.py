#!/usr/bin/env python3
"""Task B5 — the stratified dev split. Your only real feedback loop.

WHY STRATIFY
------------
A plain random 10% will not preserve the distribution of *number of relevant
documents per query*. That matters more here than anywhere else, because
Precision/Recall behave completely differently on a 1-answer query than on a
6-answer one, and the optimal cutoff is a direct function of that distribution.
An unstratified dev split will hand you a cutoff tuned for the wrong mixture.

WHY SOME QUESTIONS ARE DROPPED FIRST
------------------------------------
A handful of training questions have gold documents whose ``passage`` is empty
(BTC confirmed this is real on train, and absent from public/private test — see
``docs/reference/10_data_facts.md`` §4.2). They cannot be answered by any
retriever, so leaving them in dev parks a fixed block of guaranteed zeros in
every measurement you will ever take, and makes dev disagree with the
leaderboard by a constant. ``ingest.py --validate`` writes the list; this script
reads it and drops those questions from both sides. ``--keep-quarantined`` opts
out.

WHY A FIXED SEED
----------------
The reproduction package (Phase 5) must regenerate this exact split. Changing
the split mid-competition silently invalidates every number in the log, because
runs from before and after are no longer comparable.

USAGE
  python phases/0_harness/build_dev_split.py                 # 85/15, seed 42
  python phases/0_harness/build_dev_split.py --dev-frac 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import io_utils  # noqa: E402


def stratum(q, cap=5):
    """Bucket by |relevant|. Everything above `cap` shares one bucket, because
    those queries are rare and would otherwise form singleton strata that cannot
    be split at all."""
    n = len(q.get("relevant", []))
    return min(n, cap)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default="data/processed/queries_train.jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    # 0.15 of ~7k training questions is ~1050 dev questions, deliberately close
    # to the 999-question public test: a dev point is then worth about the same
    # as a leaderboard point, and a difference that is noise on dev is noise on
    # the leaderboard too. Smaller splits make you chase sampling error.
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--prefix", default="queries",
                    help="output stem: 'queries' for Task 1, 'task2' for Task 2")
    ap.add_argument("--quarantine", default=None,
                    help="qids to exclude, one per line; defaults to the "
                         "quarantine_*.jsonl that ingest.py --validate writes "
                         "next to --queries")
    ap.add_argument("--keep-quarantined", action="store_true",
                    help="keep questions whose every gold passage is empty "
                         "(they can never be answered; you will regret it)")
    a = ap.parse_args()

    qs = io_utils.load_queries(a.queries)

    quar = a.quarantine or os.path.join(
        os.path.dirname(a.queries),
        "quarantine_" + os.path.basename(a.queries))
    dropped = set()
    if not a.keep_quarantined and os.path.exists(quar):
        with open(quar, encoding="utf-8") as f:
            dropped = {ln.strip() for ln in f if ln.strip()}
        before = len(qs)
        qs = [q for q in qs if q["qid"] not in dropped]
        if before != len(qs):
            print(f"quarantine: dropped {before - len(qs)} unanswerable questions "
                  f"(empty gold passage) listed in {quar}\n")
    rng = random.Random(a.seed)

    buckets = defaultdict(list)
    for q in qs:
        buckets[stratum(q, a.cap)].append(q)

    train, dev = [], []
    for s in sorted(buckets):
        group = sorted(buckets[s], key=lambda q: q["qid"])  # deterministic order
        rng.shuffle(group)
        n_dev = max(1, round(len(group) * a.dev_frac)) if len(group) > 1 else 0
        dev.extend(group[:n_dev])
        train.extend(group[n_dev:])

    rng.shuffle(train)
    rng.shuffle(dev)

    tr_path = os.path.join(a.out_dir, f"{a.prefix}_train_split.jsonl")
    dv_path = os.path.join(a.out_dir, f"{a.prefix}_dev.jsonl")
    io_utils.write_jsonl(tr_path, train)
    io_utils.write_jsonl(dv_path, dev)

    # The proof that stratification worked. Eyeball these columns: they should
    # match to within a percent or two. If they do not, the split is not usable.
    print(f"seed={a.seed}  dev_frac={a.dev_frac}\n")
    print(f"{'|rel|':>6} {'all':>7} {'train':>7} {'dev':>6} "
          f"{'all%':>7} {'train%':>7} {'dev%':>7}")
    print("-" * 54)
    ca, ct, cd = Counter(map(lambda q: stratum(q, a.cap), qs)), \
        Counter(map(lambda q: stratum(q, a.cap), train)), \
        Counter(map(lambda q: stratum(q, a.cap), dev))
    for s in sorted(ca):
        lab = f"{s}+" if s == a.cap else str(s)
        print(f"{lab:>6} {ca[s]:>7} {ct[s]:>7} {cd[s]:>6} "
              f"{100*ca[s]/len(qs):>6.1f}% {100*ct[s]/max(len(train),1):>6.1f}% "
              f"{100*cd[s]/max(len(dev),1):>6.1f}%")
    print("-" * 54)
    print(f"{'TOTAL':>6} {len(qs):>7} {len(train):>7} {len(dev):>6}")
    print(f"\nwrote {tr_path}\nwrote {dv_path}")
    print(f"\nFrom here on: train on {a.prefix}_train_split.jsonl, "
          f"report on {a.prefix}_dev.jsonl. Never tune on dev's labels by hand.")


if __name__ == "__main__":
    main()
