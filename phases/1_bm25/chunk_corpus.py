#!/usr/bin/env python3
"""Task B1 — build the corpus at a chosen granularity. Ablation #1.

WHAT YOU NEED TO DO
-------------------
Run it once per granularity, then compare retrieval scores between them in
Phase 1 Task B2. Do not pick one on intuition — this ablation is worth more
than most model swaps, and the answer is not the same for every dataset.

HOW TO READ THE OUTPUT
----------------------
Watch two numbers:
  * ``n_chunks``  — the search space. More chunks = slower, but sharper units.
  * ``len_p90``   — if this exceeds your encoder's max sequence length
                    (PhoBERT ~180 words, BGE-M3 ~6000), that model is silently
                    truncating 10% of your corpus.

If ``n_chunks`` barely grows when you go from ``document`` to ``article``, the
article regex is not matching — inspect a raw record and check that newlines
survived ingest (``src.normalize.normalize(..., keep_newlines=True)``).

USAGE
  python phases/1_bm25/chunk_corpus.py --granularity article
  python phases/1_bm25/chunk_corpus.py --granularity article --window 256
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import chunking, io_utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/processed/corpus_document.jsonl")
    ap.add_argument("--granularity", default="article",
                    choices=["document", "article", "clause"])
    ap.add_argument("--window", type=int, default=None,
                    help="also split into overlapping N-word windows")
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    recs = list(io_utils.read_jsonl(a.corpus))
    out = a.out or f"data/processed/corpus_{a.granularity}.jsonl"
    chunks = chunking.build_corpus(recs, a.granularity, window=a.window,
                                   stride=a.stride, min_chars=a.min_chars)
    io_utils.write_jsonl(out, chunks)

    s = chunking.stats(chunks)
    print(f"granularity : {a.granularity}"
          + (f" (+{a.window}-word windows, stride {a.stride})" if a.window else ""))
    print(f"input docs  : {len(recs)}")
    print(f"chunks      : {s['n_chunks']}  "
          f"({s['n_chunks']/max(len(recs),1):.1f} per input doc)")
    print(f"word length : mean {s['len_mean']:.0f}  p50 {s['len_p50']}  "
          f"p90 {s['len_p90']}  p99 {s['len_p99']}  max {s['len_max']}")
    if s["len_p90"] > 180:
        print("  NOTE: p90 exceeds PhoBERT's ~180-word window — a PhoBERT-backbone\n"
              "        encoder will truncate. Either chunk finer or use BGE-M3.")
    if s["n_chunks"] <= len(recs) and a.granularity != "document":
        print("  WARNING: chunking produced no split. Check that corpus text still\n"
              "           contains newlines (see the module docstring).")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
