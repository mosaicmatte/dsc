#!/usr/bin/env python3
"""Measure the dataset and print the tables in docs/reference/10_data_facts.md.

WHY THIS EXISTS
---------------
Every claim on that page is a measurement, and measurements go stale. When BTC
ships Private Test, run this instead of trusting the page:

    python tools/data_facts.py

It reads only data/processed/, so run ingest.py first. Nothing is written; this
is a reporting tool.

WHAT TO LOOK AT FIRST
---------------------
1. The gold-set size table. If most questions still have exactly one gold
   document, recall is binary per question and you should return exactly 5 ids.
2. The length percentiles. If the median document still dwarfs 512 tokens,
   chunking is mandatory for anything dense.
3. The landmine section. Non-string ids and empty gold documents both cost
   recall silently.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import io_utils  # noqa: E402

PROC = "data/processed"
WS = re.compile(r"\s+")


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p / 100))]


def rule(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=f"{PROC}/corpus_document.jsonl")
    ap.add_argument("--queries", default=f"{PROC}/queries_train.jsonl")
    ap.add_argument("--test", default=f"{PROC}/queries_public_test.jsonl")
    a = ap.parse_args()

    doc_ids, texts, _ = io_utils.load_corpus(a.corpus)
    queries = io_utils.load_queries(a.queries)
    test = io_utils.load_queries(a.test) if os.path.exists(a.test) else []

    # ---------------------------------------------------------------- shape
    gold_sets = [set(q["relevant"]) for q in queries if q["relevant"]]
    all_gold = set().union(*gold_sets) if gold_sets else set()
    ids = set(doc_ids)
    tokens = [len(t.split()) for t in texts]

    rule("1. Shape")
    print(f"  corpus documents            {len(doc_ids):>8,}  ({len(ids):,} unique ids)")
    print(f"  train questions             {len(queries):>8,}")
    print(f"  public-test questions       {len(test):>8,}")
    print(f"  qid overlap train/test      "
          f"{len({q['qid'] for q in queries} & {q['qid'] for q in test}):>8,}")
    print(f"  documents used as gold      {len(all_gold):>8,}")
    print(f"  documents never gold        {len(ids - all_gold):>8,}"
          f"  ({100*len(ids-all_gold)/max(len(ids),1):.0f}%)")
    print(f"  total whitespace tokens     {sum(tokens):>8,}")
    dangling = all_gold - ids
    if dangling:
        print(f"  !! {len(dangling)} gold ids NOT in corpus, e.g. {sorted(dangling)[:5]}")

    # ------------------------------------------------------- gold-set sizes
    rule("2. Gold documents per question  (drives the whole submission strategy)")
    sizes = collections.Counter(len(g) for g in gold_sets)
    for n in sorted(sizes):
        print(f"  {n} gold{'s' if n != 1 else ' ':<2}  {sizes[n]:>6,}"
              f"  {100*sizes[n]/len(gold_sets):>5.1f}%")
    mean = sum(len(g) for g in gold_sets) / max(len(gold_sets), 1)
    single = 100 * sizes.get(1, 0) / max(len(gold_sets), 1)
    print(f"  mean {mean:.2f}, max {max(sizes) if sizes else 0}")
    if single > 80:
        print(f"\n  => {single:.0f}% of questions have exactly ONE gold document, so per-question")
        print("     recall is binary and macro-recall == 'is the gold doc in my top 5'.")
        print("     Return exactly 5 ids on every question: recall never decreases with")
        print("     more ids up to the cap, and precision only breaks exact recall ties.")

    # ------------------------------------------------------------- lengths
    rule("3. Document length in whitespace tokens")
    ts = sorted(tokens)
    print(f"  min {ts[0]:,}   p25 {pct(ts,25):,}   median {pct(ts,50):,}"
          f"   p75 {pct(ts,75):,}   p95 {pct(ts,95):,}   max {ts[-1]:,}")
    for t in (256, 512, 1024, 4096, 8192):
        n = sum(1 for x in ts if x > t)
        print(f"  longer than {t:>5,} tokens: {n:>6,}  ({100*n/max(len(ts),1):>5.1f}%)")
    qt = sorted(len(q["text"].split()) for q in queries)
    print(f"  questions: min {qt[0]}  median {pct(qt,50)}  p95 {pct(qt,95)}  max {qt[-1]}")
    over512 = 100 * sum(1 for x in ts if x > 512) / max(len(ts), 1)
    if over512 > 50:
        print(f"\n  => {over512:.0f}% of documents exceed 512 tokens. A dense encoder that reads")
        print("     only the first 512 reads the letterhead, which every document shares.")
        print("     Chunk, retrieve chunks, aggregate back to the parent document id.")

    # ----------------------------------------------------------- landmines
    rule("4. Landmines")
    nonstr = [d for q in queries for d in q["relevant"] if not isinstance(d, str)]
    print(f"  non-string gold ids after ingest       {len(nonstr):>6}"
          f"   (must be 0 — int ids score a silent zero)")

    empty = {d for d, t in zip(doc_ids, texts) if not t.strip()}
    empty_gold = empty & all_gold
    unreachable = [q["qid"] for q in queries
                   if q["relevant"] and set(q["relevant"]) <= empty]
    print(f"  documents with an empty passage        {len(empty):>6}")
    print(f"  of those, gold for some question       {len(empty_gold):>6}")
    print(f"  questions whose every gold is empty    {len(unreachable):>6}")
    print(f"  => local recall ceiling                "
          f"{1 - len(unreachable)/max(len(queries),1):>6.4f}")

    groups = collections.defaultdict(list)
    for d, t in zip(doc_ids, texts):
        t = WS.sub(" ", t).strip().lower()
        if t:
            groups[hashlib.sha1(t.encode()).hexdigest()].append(d)
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    dup_docs = {d for v in dups.values() for d in v}
    hit = [q["qid"] for q in queries if set(q["relevant"]) & dup_docs]
    print(f"  duplicate-passage groups               {len(dups):>6}"
          f"   ({len(dup_docs)} documents)")
    print(f"  questions whose gold has a twin        {len(hit):>6}")
    for v in list(dups.values())[:5]:
        print(f"      {sorted(v)}")

    print("\nSee docs/reference/10_data_facts.md for what each of these means.")


if __name__ == "__main__":
    main()
