#!/usr/bin/env python3
"""Task B3 — grid search k1 x b on dev.

WHY THIS IS CHEAP
-----------------
``src.bm25.BM25Index`` takes k1 and b at *scoring* time, so all 16 configs share
one tokenisation + one index build. Tokenising is the expensive part; do it once.

WHAT TO LOOK FOR IN THE HEATMAP
-------------------------------
* A **flat** grid means BM25 is not the bottleneck — stop tuning, move to Phase 2.
* Best at **b = 1.0** means long documents were winning on length alone; strong
  normalisation fixed it. Expect this at document granularity.
* Best at **b = 0.3** means length is genuinely informative — a longer article
  really is more likely to be the answer. Expect this at article granularity,
  where units are already length-comparable.
* Best at **low k1** means repetition carries little signal (typical for legal
  text, where the key term appears once in a definition).

Report the metric you are actually scored on. This script optimises Recall with
Precision as tiebreak, exactly like the leaderboard.

USAGE
  python phases/1_bm25/grid_search.py --corpus data/processed/corpus_article.jsonl
  python phases/1_bm25/grid_search.py --metric recall@100   # tune for the reranker's ceiling
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import chunking, cutoff, exp_log, io_utils, metrics, normalize  # noqa: E402

from bm25_baseline import build_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/processed/corpus_article.jsonl")
    ap.add_argument("--queries", default="data/processed/queries_dev.jsonl")
    ap.add_argument("--segmenter", default="none")
    ap.add_argument("--k1", type=float, nargs="+", default=[0.9, 1.2, 1.5, 2.0])
    ap.add_argument("--b", type=float, nargs="+", default=[0.3, 0.5, 0.75, 1.0])
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--aggregate", default=None, choices=[None, "max", "sum", "mean"])
    ap.add_argument("--cutoff", default="ratio")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--metric", default="official",
                    help="'official' (recall, precision tiebreak) or e.g. recall@100")
    ap.add_argument("--out", default="work/analysis/bm25_grid.md")
    ap.add_argument("--save-best", action="store_true",
                    help="write the winning run file and log it")
    a = ap.parse_args()

    idx, _ = build_index(a.corpus, a.segmenter, False)
    queries = io_utils.load_queries(a.queries)
    qrels = io_utils.qrels(queries)
    qtok = {q["qid"]: normalize.tokenize(q["text"], segmenter=a.segmenter)
            for q in queries}
    pmap = None
    if a.aggregate:
        pmap = chunking.parent_map(list(io_utils.read_jsonl(a.corpus)))

    grid, best = {}, None
    for k1 in a.k1:
        for b in a.b:
            run = idx.batch_search(qtok, top_k=a.depth, k1=k1, b=b, progress=False)
            if pmap:
                run = chunking.aggregate_to_parent(run, pmap, a.aggregate, a.depth)
            preds = cutoff.apply_to_run(run, rule=a.cutoff, alpha=a.alpha, k=a.k)
            s = metrics.official(preds, qrels)
            val = (s["primary_recall"] if a.metric == "official"
                   else metrics.diagnostics(run, qrels, ks=(int(a.metric.split("@")[1]),))[a.metric])
            grid[(k1, b)] = (val, s["tiebreak_precision"])
            key = (val, s["tiebreak_precision"])
            if best is None or key > best[0]:
                best = (key, k1, b, run, s)
            print(f"k1={k1:<4} b={b:<5} -> {a.metric}={val:.4f} "
                  f"P={s['tiebreak_precision']:.4f}")

    lines = [f"# BM25 grid — {os.path.basename(a.corpus)} (segmenter={a.segmenter})", "",
             f"Metric: **{a.metric}** (Precision shown as tiebreak). "
             f"Cutoff: {a.cutoff} alpha={a.alpha}.", "",
             "| k1 \\ b | " + " | ".join(str(b) for b in a.b) + " |",
             "|---" * (len(a.b) + 1) + "|"]
    for k1 in a.k1:
        cells = []
        for b in a.b:
            v, p = grid[(k1, b)]
            mark = "**" if (k1, b) == (best[1], best[2]) else ""
            cells.append(f"{mark}{v:.4f}{mark}")
        lines.append(f"| **{k1}** | " + " | ".join(cells) + " |")
    lines += ["", f"Best: **k1={best[1]}, b={best[2]}** "
                  f"-> {a.metric}={best[0][0]:.4f}, P={best[0][1]:.4f}", "",
              "## Interpretation (fill in)", "",
              "- Is the grid flat? If so BM25 is not the bottleneck — move on.",
              "- Where did b land, and does that match the granularity argument?",
              "- Where did k1 land, and what does that say about repetition in this corpus?"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[3:]))
    print(f"\nwrote {a.out}")

    if a.save_best:
        gran = os.path.basename(a.corpus).replace("corpus_", "").replace(".jsonl", "")
        run_id = f"bm25-{gran}-{a.segmenter}-grid-best"
        io_utils.write_run(f"work/experiments/runs/{run_id}.jsonl", best[3])
        exp_log.log_run({
            "run_id": run_id, "phase": "1", "task": "1", "chunking": gran,
            "retriever": f"bm25(k1={best[1]},b={best[2]},seg={a.segmenter})",
            "cutoff_rule": f"{a.cutoff}:{a.alpha}",
            "dev_P": best[4]["tiebreak_precision"], "dev_R": best[4]["primary_recall"],
            "dev_official": best[4]["primary_recall"],
            "notes": f"grid winner of {len(grid)} configs",
        })
        print(f"wrote work/experiments/runs/{run_id}.jsonl and logged it")


if __name__ == "__main__":
    main()
