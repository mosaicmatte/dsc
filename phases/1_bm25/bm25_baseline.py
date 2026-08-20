#!/usr/bin/env python3
"""Task B2 — the BM25 baseline. Produces a run file + a row in the experiment log.

WHAT YOU NEED TO DO
-------------------
Run it four times (2 granularities x segmented/unsegmented) and log all four.
Zero-effort baselines are ablation rows in the paper, not throwaways.

    python phases/1_bm25/bm25_baseline.py --corpus data/processed/corpus_document.jsonl --segmenter none
    python phases/1_bm25/bm25_baseline.py --corpus data/processed/corpus_document.jsonl --segmenter pyvi
    python phases/1_bm25/bm25_baseline.py --corpus data/processed/corpus_article.jsonl  --segmenter none
    python phases/1_bm25/bm25_baseline.py --corpus data/processed/corpus_article.jsonl  --segmenter pyvi

HOW IT WORKS
------------
1. Tokenise corpus and queries with the SAME function (this is not optional —
   any asymmetry silently destroys recall).
2. Build one inverted index; k1/b are applied at scoring time.
3. Retrieve ``--depth`` candidates per query and write a *run* file
   (full ranking). The cutoff into an answer set happens later, in
   ``cutoff_sweep.py`` — keep the two separate so one retrieval pass can be
   re-cut a hundred different ways.
4. If the corpus is chunked but gold labels are at document level, pass
   ``--aggregate max`` to collapse chunk scores back to parent documents.

WHY DEPTH MATTERS
-----------------
``--depth`` is your recall ceiling for everything downstream. Reranking a top-50
whose Recall@50 is 0.85 can never exceed 0.85. Retrieve deep (100), cut shallow.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import chunking, cutoff, exp_log, io_utils, metrics, normalize  # noqa: E402
from src.bm25 import BM25Index  # noqa: E402


def build_index(corpus_path, segmenter, stopwords, cache=True):
    doc_ids, texts, _ = io_utils.load_corpus(corpus_path)
    tag = f"{os.path.basename(corpus_path)}.{segmenter}.{int(stopwords)}"
    cache_path = os.path.join("data/processed", f".bm25_{tag}.pkl")
    if cache and os.path.exists(cache_path):
        print(f"loading cached index {cache_path}")
        return BM25Index.load(cache_path), doc_ids
    t0 = time.time()
    toks = [normalize.tokenize(t, segmenter=segmenter, remove_stopwords=stopwords)
            for t in texts]
    idx = BM25Index(toks, doc_ids)
    print(f"indexed {idx} in {time.time()-t0:.1f}s")
    if cache:
        idx.save(cache_path)
    return idx, doc_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/processed/corpus_article.jsonl")
    ap.add_argument("--queries", default="data/processed/queries_dev.jsonl")
    ap.add_argument("--segmenter", default="none",
                    choices=["none", "pyvi", "underthesea"])
    ap.add_argument("--stopwords", action="store_true")
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--aggregate", default=None, choices=[None, "max", "sum", "mean"],
                    help="collapse chunk scores to parent doc_id before scoring")
    # Default top_k/5, not ratio/10. 92% of questions have exactly ONE gold
    # document, recall is the primary metric, and recall never decreases with
    # more ids up to BTC's cap of 5 — so 5 is optimal and k>5 zeroes the
    # question. See docs/reference/10_data_facts.md §2.
    ap.add_argument("--cutoff", default="top_k",
                    choices=["top_k", "ratio", "threshold", "gap"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    idx, _ = build_index(a.corpus, a.segmenter, a.stopwords, cache=not a.no_cache)
    queries = io_utils.load_queries(a.queries)
    qtok = {q["qid"]: normalize.tokenize(q["text"], segmenter=a.segmenter,
                                         remove_stopwords=a.stopwords)
            for q in queries}

    run = idx.batch_search(qtok, top_k=a.depth, k1=a.k1, b=a.b)

    if a.aggregate:
        corpus = list(io_utils.read_jsonl(a.corpus))
        run = chunking.aggregate_to_parent(run, chunking.parent_map(corpus),
                                           how=a.aggregate, top_k=a.depth)

    gran = os.path.basename(a.corpus).replace("corpus_", "").replace(".jsonl", "")
    run_id = a.run_id or f"bm25-{gran}-{a.segmenter}-k{a.k1}-b{a.b}"
    run_path = f"work/experiments/runs/{run_id}.jsonl"
    io_utils.write_run(run_path, run)

    qrels = io_utils.qrels(queries)
    if not any(qrels.values()):
        # An unlabelled split (public/private test) has nothing to score. Printing
        # 0.0000 here reads as "the model failed" and has sent people debugging a
        # working pipeline.
        print(f"\nrun_id : {run_id}")
        print(f"wrote {run_path}")
        print("\nqueries are UNLABELLED (test split) — no scores to report.\n"
              "This is expected. Build the submission with:\n"
              f"  python phases/1_bm25/make_submission.py --run {run_path} "
              f"--queries {a.queries}")
        return

    preds = cutoff.apply_to_run(run, rule=a.cutoff, k=a.k, alpha=a.alpha)
    s = metrics.official(preds, qrels)
    d = metrics.diagnostics(run, qrels)

    print(f"\nrun_id   : {run_id}")
    print(f"corpus   : {a.corpus}  segmenter={a.segmenter}  k1={a.k1} b={a.b}")
    print(f"cutoff   : {a.cutoff}  avg set size {s['avg_pred_size']:.2f}")
    print(f"  RECALL (primary)   : {s['primary_recall']:.4f}")
    print(f"  PRECISION(tiebreak): {s['tiebreak_precision']:.4f}")
    print("  --- retriever ceiling (what reranking could reach) ---")
    for k in (10, 50, 100):
        print(f"  recall@{k:<4}: {d[f'recall@{k}']:.4f}")
    print(f"  mrr@10     : {d['mrr@10']:.4f}")
    print(f"\nwrote {run_path}")

    if not a.no_log:
        exp_log.log_run({
            "run_id": run_id, "phase": "1", "task": "1",
            "chunking": gran,
            "retriever": f"bm25(k1={a.k1},b={a.b},seg={a.segmenter},agg={a.aggregate})",
            "negatives": "-", "reranker": "-",
            "cutoff_rule": f"{a.cutoff}:{a.alpha if a.cutoff=='ratio' else a.k}",
            "dev_P": s["tiebreak_precision"], "dev_R": s["primary_recall"],
            "dev_official": s["primary_recall"], "n_params": 0,
            "notes": f"recall@100={d['recall@100']:.4f}",
        })
        print(f"logged to {exp_log.LOG_PATH}")


if __name__ == "__main__":
    main()
