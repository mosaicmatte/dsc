#!/usr/bin/env python3
"""Task B4 — the retrieval ceiling table. Decides where your remaining days go.

WHAT IT ANSWERS
---------------
"Should I invest in the retriever or the reranker?" — and it is exactly the
"why was this method insufficient / what did the next method fix" analysis BTC
asked for in the paper.

HOW TO READ IT
--------------
The table shows the retriever's recall@k next to the final score after reranking
that top-k.

  * final score CLOSE to the ceiling  -> the reranker is doing its job; the only
    way up is a better RETRIEVER (or a deeper rerank).
  * final score FAR BELOW the ceiling -> the relevant documents are in the
    candidate list but ranked badly; invest in the RERANKER.
  * ceiling itself LOW at depth 100   -> stop reranking entirely. You have a
    retrieval problem and no amount of reordering can fix it.

The ``gap`` column is what you act on. The ``Δceiling`` column tells you whether
reranking deeper would even help: if recall@100 is barely above recall@50, depth
100 costs you 2x compute for nothing.

USAGE
  python phases/3_rerank/ceiling_table.py \
      --retriever work/experiments/runs/hybrid-best.jsonl \
      --reranked work/experiments/runs/rerank-best.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import cutoff, io_utils, metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retriever", required=True)
    ap.add_argument("--reranked", nargs="+", required=True,
                    help="one or more reranked runs (label them by depth)")
    ap.add_argument("--queries", default="data/processed/queries_dev.jsonl")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100])
    ap.add_argument("--cutoff", default="ratio")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--out", default="work/analysis/ceiling_table.md")
    a = ap.parse_args()

    qrels = io_utils.qrels(io_utils.load_queries(a.queries))
    ret = io_utils.load_run(a.retriever)

    L = ["# Retrieval ceiling table", "",
         f"Retriever: `{os.path.basename(a.retriever)}`  ·  "
         f"cutoff: {a.cutoff} α={a.alpha}", "",
         "## 1. Retriever ceiling by depth", "",
         "| depth k | retriever recall@k | Δ vs previous |", "|---|---|---|"]
    prev = None
    ceilings = {}
    for k in a.ks:
        r = metrics.recall_at_k(ret, qrels, k)
        ceilings[k] = r
        d = "—" if prev is None else f"+{r-prev:.4f}"
        L.append(f"| {k} | {r:.4f} | {d} |")
        prev = r

    L += ["", "> A small Δ between depth 50 and 100 means reranking deeper costs "
              "2x compute for almost no ceiling gain.", "",
          "## 2. Final score after reranking", "",
          "| reranked run | final recall | final precision | ceiling used | "
          "gap to ceiling | % of ceiling |", "|---|---|---|---|---|---|"]

    verdicts = []
    for rr_path in a.reranked:
        rr = io_utils.load_run(rr_path)
        preds = cutoff.apply_to_run(rr, rule=a.cutoff, alpha=a.alpha)
        s = metrics.official(preds, qrels)
        # infer the rerank depth from the filename if it is encoded as -d<N>
        base = os.path.basename(rr_path)
        depth = next((k for k in a.ks if f"-d{k}" in base), max(a.ks))
        ceil = ceilings.get(depth, metrics.recall_at_k(ret, qrels, depth))
        gap = ceil - s["primary_recall"]
        pct = 100 * s["primary_recall"] / max(ceil, 1e-9)
        L.append(f"| `{base}` | {s['primary_recall']:.4f} | "
                 f"{s['tiebreak_precision']:.4f} | {ceil:.4f} (@{depth}) | "
                 f"{gap:.4f} | {pct:.0f}% |")
        verdicts.append((base, gap, ceil, pct))

    L += ["", "## 3. Verdict — retriever or reranker?", ""]
    for base, gap, ceil, pct in verdicts:
        if ceil < 0.80:
            v = ("**RETRIEVER.** The ceiling itself is low — relevant documents are "
                 "never retrieved, so reranking cannot reach them. Improve recall "
                 "first (deeper retrieval, better fusion weight, finer chunking).")
        elif gap > 0.10:
            v = ("**RERANKER.** The documents are in the candidate list but ranked "
                 "badly. Fine-tune the cross-encoder on your retriever's errors, or "
                 "re-sweep the cutoff — a large gap is often a cutoff problem, not a "
                 "model problem.")
        else:
            v = ("**RETRIEVER (or stop).** The reranker is already extracting most of "
                 f"what the candidate list contains ({pct:.0f}% of ceiling). Further "
                 "reranker work has little headroom; raise the ceiling instead.")
        L.append(f"- `{base}`: {v}")

    L += ["", "## 4. Write one paragraph here (BTC asked for this)", "",
          "_Which method was insufficient, why, and what did the next method fix?_",
          "", "> "]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
