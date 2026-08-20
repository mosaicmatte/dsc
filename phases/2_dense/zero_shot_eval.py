#!/usr/bin/env python3
"""Task B1 — evaluate a bi-encoder (zero-shot or fine-tuned) and write a run file.

WHAT YOU NEED TO DO
-------------------
Run it for each candidate embedder. Log every result, including the bad ones —
zero-shot numbers are ablation rows in the paper, not throwaways.

The input format (segmented vs not) is chosen automatically from
``src/dense.REGISTRY``. Do NOT segment by hand before calling this; you will
double-segment a PhoBERT model or wrongly segment a BGE-M3 one, and neither
raises an error.

HOW TO READ THE OUTPUT
----------------------
``recall@100`` is the number that matters most here, not the official score: it
is the ceiling for the reranker in Phase 3. A model with worse recall@10 but
better recall@100 is the better *retriever* in a retrieve-then-rerank pipeline.

Corpus embeddings are cached under ``data/processed/.emb_*.npy``. Delete the
cache after fine-tuning, or pass ``--no-cache``, or you will silently evaluate
the old weights.

USAGE
  python phases/2_dense/zero_shot_eval.py --model AITeamVN/Vietnamese_Embedding
  python phases/2_dense/zero_shot_eval.py --model models/biencoder-r2 --run-id dense-r2
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import chunking, cutoff, dense, exp_log, io_utils, metrics  # noqa: E402


def embed_corpus(model, model_name, corpus_path, batch_size, cache=True):
    import numpy as np
    doc_ids, texts, _ = io_utils.load_corpus(corpus_path)
    tag = hashlib.sha1(f"{model_name}|{corpus_path}".encode()).hexdigest()[:10]
    cache_path = f"data/processed/.emb_{tag}.npy"
    if cache and os.path.exists(cache_path):
        print(f"loading cached embeddings {cache_path} "
              f"(delete it if the model changed)")
        return doc_ids, np.load(cache_path)
    t0 = time.time()
    emb = dense.encode(model, texts, model_name, is_query=False,
                       batch_size=batch_size)
    print(f"embedded {len(texts)} docs in {time.time()-t0:.0f}s -> {emb.shape}")
    if cache:
        np.save(cache_path, emb)
    return doc_ids, emb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--registry-as", default=None,
                    help="for a fine-tuned local path: which REGISTRY entry's "
                         "input format it inherits (defaults to --model)")
    ap.add_argument("--corpus", default="data/processed/corpus_article.jsonl")
    ap.add_argument("--queries", default="data/processed/queries_dev.jsonl")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--aggregate", default=None, choices=[None, "max", "sum", "mean"])
    # Default top_k/5, not ratio/10. 92% of questions have exactly ONE gold
    # document, recall is the primary metric, and recall never decreases with
    # more ids up to BTC's cap of 5 — so 5 is optimal and k>5 zeroes the
    # question. See docs/reference/10_data_facts.md §2.
    ap.add_argument("--cutoff", default="top_k")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    fmt_name = a.registry_as or a.model
    sp = dense.spec(fmt_name)
    print(f"model      : {a.model}")
    print(f"input fmt  : {sp.backbone} -> segmented={sp.segmented}, "
          f"max_seq={sp.max_seq}")

    model = dense.load_encoder(a.model, device=a.device)
    doc_ids, doc_emb = embed_corpus(model, fmt_name, a.corpus, a.batch_size,
                                    cache=not a.no_cache)

    queries = io_utils.load_queries(a.queries)
    qids = [q["qid"] for q in queries]
    q_emb = dense.encode(model, [q["text"] for q in queries], fmt_name,
                         is_query=True, batch_size=a.batch_size)

    run = dense.search(q_emb, doc_emb, doc_ids, qids, top_k=a.depth)
    if a.aggregate:
        pmap = chunking.parent_map(list(io_utils.read_jsonl(a.corpus)))
        run = chunking.aggregate_to_parent(run, pmap, a.aggregate, a.depth)

    run_id = a.run_id or f"dense-{os.path.basename(a.model).replace('/','_')}"
    io_utils.write_run(f"work/experiments/runs/{run_id}.jsonl", run)

    qrels = io_utils.qrels(queries)
    if not any(qrels.values()):
        print(f"\nwrote work/experiments/runs/{run_id}.jsonl")
        print("queries are UNLABELLED (test split) — no scores to report. Expected.")
        return
    preds = cutoff.apply_to_run(run, rule=a.cutoff, alpha=a.alpha, k=a.k)
    s = metrics.official(preds, qrels)
    d = metrics.diagnostics(run, qrels)

    print(f"\nrun_id : {run_id}")
    print(f"  RECALL (primary)   : {s['primary_recall']:.4f}")
    print(f"  PRECISION(tiebreak): {s['tiebreak_precision']:.4f}")
    print("  --- ceiling for the Phase 3 reranker ---")
    for k in (10, 50, 100):
        print(f"  recall@{k:<4}: {d[f'recall@{k}']:.4f}")
    print(f"  mrr@10     : {d['mrr@10']:.4f}   ndcg@10: {d['ndcg@10']:.4f}")

    exp_log.log_run({
        "run_id": run_id, "phase": "2", "task": "1",
        "chunking": os.path.basename(a.corpus).replace("corpus_", "").replace(".jsonl", ""),
        "retriever": a.model, "negatives": "-", "reranker": "-",
        "cutoff_rule": f"{a.cutoff}:{a.alpha}",
        "dev_P": s["tiebreak_precision"], "dev_R": s["primary_recall"],
        "dev_official": s["primary_recall"], "n_params": sp.params,
        "notes": f"recall@100={d['recall@100']:.4f} mrr@10={d['mrr@10']:.4f}",
    })
    print(f"\nlogged to {exp_log.LOG_PATH}")


if __name__ == "__main__":
    main()
