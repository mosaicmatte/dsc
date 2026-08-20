#!/usr/bin/env python3
"""Task B5 — fuse the lexical and dense runs.

WHY THIS WINS
-------------
BM25 and a dense encoder fail on disjoint query types. BM25 nails
"Nghị định 100/2019/NĐ-CP" (maximum IDF, exact match) and misses "nghỉ phép năm"
against a corpus that says "nghỉ hằng năm" (no shared term). The dense model does
the opposite. Fusing recovers both.

TWO METHODS
-----------
--method rrf        rank-based: sum of w/(K+rank). Scale-free, needs no
                    calibration, very hard to break. Start here.
--method weighted   score-based: per-query normalisation then weighted sum.
                    Keeps score MARGINS, which is what the `ratio` cutoff rule
                    consumes downstream — so this often wins end-to-end even when
                    its recall@100 merely ties RRF.

--sweep tunes the dense weight on dev. Do not assume 0.5.

WHAT TO CHECK IN THE OUTPUT
---------------------------
If the best weight is 0.0 or 1.0, fusion is not helping and one system dominates
— report that honestly rather than shipping a hybrid that is really one model.
If recall@100 barely moves but recall@10 jumps, fusion is reordering rather than
finding new documents: good for the final score, no help to the reranker ceiling.

USAGE
  python phases/2_dense/hybrid.py --dense work/experiments/runs/dense-r3.jsonl \
      --lexical work/experiments/runs/bm25-best.jsonl --sweep
  python phases/2_dense/hybrid.py --dense <d> --lexical <l> --method weighted \
      --w-dense 0.6 --run-id hybrid-best
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import cutoff, exp_log, fusion, io_utils, metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense", required=True)
    ap.add_argument("--lexical", required=True)
    ap.add_argument("--queries", default="data/processed/queries_dev.jsonl")
    ap.add_argument("--method", default="rrf", choices=["rrf", "weighted"])
    ap.add_argument("--w-dense", type=float, default=0.5)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--norm", default="minmax", choices=["minmax", "zscore", "sum"])
    ap.add_argument("--depth", type=int, default=100)
    # Default top_k/5, not ratio/10. 92% of questions have exactly ONE gold
    # document, recall is the primary metric, and recall never decreases with
    # more ids up to BTC's cap of 5 — so 5 is optimal and k>5 zeroes the
    # question. See docs/reference/10_data_facts.md §2.
    ap.add_argument("--cutoff", default="top_k")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--sweep", action="store_true", help="tune the dense weight on dev")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", default="work/analysis/hybrid_weight_sweep.md")
    a = ap.parse_args()

    d_run = io_utils.load_run(a.dense)
    l_run = io_utils.load_run(a.lexical)
    queries = io_utils.load_queries(a.queries)
    qrels = io_utils.qrels(queries)

    print("component runs on dev:")
    for name, r in (("dense  ", d_run), ("lexical", l_run)):
        dg = metrics.diagnostics(r, qrels, ks=(10, a.depth))
        print(f"  {name}: recall@10={dg['recall@10']:.4f}  "
              f"recall@{a.depth}={dg[f'recall@{a.depth}']:.4f}  "
              f"mrr@10={dg['mrr@10']:.4f}")

    if a.sweep:
        lines = ["# Hybrid weight sweep", "",
                 f"dense: `{os.path.basename(a.dense)}`  ·  "
                 f"lexical: `{os.path.basename(a.lexical)}`", "",
                 "## Weighted score fusion", "",
                 f"| w_dense | recall@{a.depth} | recall@10 | mrr@10 |",
                 "|---|---|---|---|"]
        rows = fusion.sweep_weight(d_run, l_run, qrels, depth=a.depth, method=a.norm)
        for r in sorted(rows, key=lambda x: x["w_dense"]):
            lines.append(f"| {r['w_dense']:.1f} | {r[f'recall@{a.depth}']:.4f} | "
                         f"{r['recall@10']:.4f} | {r['mrr@10']:.4f} |")
        best_w = rows[0]["w_dense"]
        lines += ["", f"Best dense weight: **{best_w:.1f}** "
                      f"(recall@{a.depth}={rows[0][f'recall@{a.depth}']:.4f})", ""]

        lines += ["## RRF (rank-based, for comparison)", "",
                  f"| w_dense | recall@{a.depth} | recall@10 | mrr@10 |",
                  "|---|---|---|---|"]
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            f_run = fusion.rrf([d_run, l_run], [w, 1 - w], K=a.rrf_k, top_k=a.depth)
            dg = metrics.diagnostics(f_run, qrels, ks=(10, a.depth))
            lines.append(f"| {w:.2f} | {dg[f'recall@{a.depth}']:.4f} | "
                         f"{dg['recall@10']:.4f} | {dg['mrr@10']:.4f} |")

        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))
        print(f"\nwrote {a.out}")
        if best_w in (0.0, 1.0):
            print("\nNOTE: the best weight is at an endpoint — fusion is not helping "
                  "and one system dominates. Say so in the log rather than shipping a "
                  "'hybrid' that is really a single model.")
        a.w_dense = best_w

    if a.method == "rrf":
        fused = fusion.rrf([d_run, l_run], [a.w_dense, 1 - a.w_dense],
                           K=a.rrf_k, top_k=a.depth)
        desc = f"rrf(K={a.rrf_k},w_dense={a.w_dense})"
    else:
        fused = fusion.weighted([d_run, l_run], [a.w_dense, 1 - a.w_dense],
                                method=a.norm, top_k=a.depth)
        desc = f"weighted({a.norm},w_dense={a.w_dense})"

    run_id = a.run_id or f"hybrid-{a.method}-w{a.w_dense}"
    io_utils.write_run(f"work/experiments/runs/{run_id}.jsonl", fused)

    preds = cutoff.apply_to_run(fused, rule=a.cutoff, alpha=a.alpha)
    s = metrics.official(preds, qrels)
    dg = metrics.diagnostics(fused, qrels)
    print(f"\nFUSED [{desc}]  run_id={run_id}")
    print(f"  RECALL (primary)   : {s['primary_recall']:.4f}")
    print(f"  PRECISION(tiebreak): {s['tiebreak_precision']:.4f}")
    for k in (10, 50, 100):
        print(f"  recall@{k:<4}: {dg[f'recall@{k}']:.4f}")

    exp_log.log_run({
        "run_id": run_id, "phase": "2", "task": "1",
        "retriever": f"hybrid:{desc}", "negatives": "-", "reranker": "-",
        "cutoff_rule": f"{a.cutoff}:{a.alpha}",
        "dev_P": s["tiebreak_precision"], "dev_R": s["primary_recall"],
        "dev_official": s["primary_recall"],
        "notes": f"dense={os.path.basename(a.dense)} lex={os.path.basename(a.lexical)} "
                 f"recall@100={dg['recall@100']:.4f}",
    })
    print(f"\nlogged. NOW RE-SWEEP THE CUTOFF — the score distribution changed:\n"
          f"  python phases/1_bm25/cutoff_sweep.py --run work/experiments/runs/{run_id}.jsonl --plot")


if __name__ == "__main__":
    main()
