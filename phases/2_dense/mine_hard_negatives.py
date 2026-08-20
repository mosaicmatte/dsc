#!/usr/bin/env python3
"""Tasks B3/B4 — mine hard negatives from a run file.

WHAT A HARD NEGATIVE IS
-----------------------
A document your current retriever ranks highly for a query and that is NOT
labelled relevant. Training against random negatives stops teaching the model
anything after roughly one epoch; training against these is what raises
precision.

THE FALSE-NEGATIVE TRAP — READ BEFORE RUNNING
---------------------------------------------
Some of those top-ranked "negatives" are actually correct answers that nobody
labelled. Training on them teaches the model that a right answer is wrong, which
is worse than not training at all. Legal corpora are full of near-duplicate
provisions, so this is a real and common failure.

Two defences, both implemented here:
  --skip-top N     discard the N highest-ranked candidates outright. Ranks 1-2 of
                   a good retriever are the most likely unlabelled positives.
  --margin EPS     discard any candidate scoring within EPS of the gold document's
                   score. If the model cannot tell them apart, neither can you,
                   and it is probably a duplicate provision.

WHAT YOU MUST DO AFTER RUNNING
------------------------------
Open the file and read ten mined negatives by hand. Confirm none of them answers
its query. This takes ten minutes and is the highest value-per-minute action in
the whole phase.

    python phases/2_dense/mine_hard_negatives.py --run <run> --out <pairs> --inspect 10

USAGE
  # round 2: mine from BM25
  python phases/2_dense/mine_hard_negatives.py --run work/experiments/runs/bm25-best.jsonl \
      --out data/processed/train_pairs_bm25neg.jsonl --skip-top 2 --n-neg 8
  # round 3: mine from your own round-2 retriever
  python phases/2_dense/mine_hard_negatives.py --run work/experiments/runs/dense-r2.jsonl \
      --out data/processed/train_pairs_selfneg.jsonl --skip-top 3 --n-neg 8
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import io_utils  # noqa: E402


# =============================================================================
# TODO(YOU/phase2): choose which candidates become training negatives.
# -----------------------------------------------------------------------------
# WHY HERE: which negatives you train on matters more than almost any
#   hyperparameter (see docs/reference/03_dense_retrieval.md §3-4). The default
#   below takes the first `n_neg` candidates, or samples randomly. You can
#   probably do better.
#
# WHAT TO WRITE: given `candidates` (doc ids the retriever ranked highly but
#   which are NOT labelled relevant, already filtered by --skip-top and
#   --margin), return the ones to train against, as a list of at most `n_neg`.
#
# IDEAS:
#   * spread them across ranks (one from 3-10, one from 10-25, one from 25-50)
#     instead of taking the hardest ones only -- "hardest" often means
#     "unlabelled positive"
#   * prefer negatives from the same legal area as the positive
#   * drop candidates whose text is nearly identical to the positive
#
# HOW TO TEST IT: mine, then READ them, then retrain and compare on dev:
#   python phases/2_dense/mine_hard_negatives.py --run <run> --out <pairs> --inspect 10
#
# PYTHON NOTE (from C++): `candidates[a:b]` is a slice (a sub-vector).
#   `rng.sample(xs, n)` picks n distinct elements at random.
# =============================================================================
def select_negatives(candidates, n_neg, rng, positive_id=None, dtext=None):
    """Pick which of `candidates` to train against. Default: the top n_neg."""
    # TODO(YOU/phase2): replace this with your own strategy and measure it.
    return candidates[:n_neg]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run file to mine from")
    ap.add_argument("--queries", default="data/processed/queries_train_split.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus_article.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-neg", type=int, default=8, help="negatives per positive")
    ap.add_argument("--skip-top", type=int, default=2,
                    help="discard the N highest-ranked candidates (false-negative guard)")
    ap.add_argument("--depth", type=int, default=50, help="mine within this rank range")
    ap.add_argument("--margin", type=float, default=None,
                    help="discard candidates scoring within EPS of the gold doc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inspect", type=int, default=0,
                    help="print N mined (query, negative) pairs and exit")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    run = io_utils.load_run(a.run)
    queries = io_utils.load_queries(a.queries)
    doc_ids, texts, _ = io_utils.load_corpus(a.corpus)
    dtext = dict(zip(doc_ids, texts))

    rows, n_skipped_margin, n_no_cands = [], 0, 0
    for q in queries:
        rel = set(q["relevant"])
        if not rel:
            continue
        ranked = run.get(q["qid"], [])[:a.depth]
        gold_scores = [s for d, s in ranked if d in rel]
        gold_top = max(gold_scores) if gold_scores else None

        cands = []
        for rank, (d, s) in enumerate(ranked):
            if rank < a.skip_top:
                continue                      # false-negative guard #1
            if d in rel:
                continue
            if a.margin is not None and gold_top is not None and s > gold_top - a.margin:
                n_skipped_margin += 1         # false-negative guard #2
                continue
            cands.append(d)

        if not cands:
            n_no_cands += 1
            continue
        negs = select_negatives(cands, a.n_neg, rng, dtext=dtext)
        for pos in rel:
            if pos not in dtext:
                continue
            rows.append({"qid": q["qid"], "query": q["text"],
                         "positive_id": pos, "positive": dtext[pos],
                         "negative_ids": negs,
                         "negatives": [dtext.get(n, "") for n in negs]})

    if a.inspect:
        print(f"=== {a.inspect} mined pairs — READ THESE, confirm none is a real answer ===\n")
        for r in rows[:a.inspect]:
            print(f"QUERY   : {r['query']}")
            print(f"POSITIVE({r['positive_id']}): {r['positive'][:220]}")
            for nid, nt in list(zip(r["negative_ids"], r["negatives"]))[:3]:
                print(f"  NEG({nid}): {nt[:200]}")
            print("-" * 76)
        return

    n = io_utils.write_jsonl(a.out, rows)
    print(f"mined from   : {a.run}")
    print(f"training rows: {n}  ({a.n_neg} negatives each)")
    print(f"skip-top     : {a.skip_top}   depth: {a.depth}")
    if a.margin is not None:
        print(f"margin guard : discarded {n_skipped_margin} candidates within "
              f"{a.margin} of gold")
    if n_no_cands:
        print(f"WARNING      : {n_no_cands} queries yielded no negatives "
              f"(retriever returned only gold — lower --skip-top or raise --depth)")
    print(f"wrote {a.out}")
    print(f"\nNOW DO THIS:  python {sys.argv[0]} --run {a.run} --out {a.out} --inspect 10")


if __name__ == "__main__":
    main()
