#!/usr/bin/env python3
"""Task B1 — rerank a run's top-k with a cross-encoder.

WHAT THIS DOES
--------------
Takes a run file (full ranking from the retriever), keeps the top ``--depth``
candidates per query, scores each (query, document) pair jointly with a
cross-encoder, and re-sorts. Documents beyond ``--depth`` keep their original
relative order BELOW everything reranked, so the run stays a complete ranking and
recall@100 is never damaged by reranking.

DEPTH IS A HYPERPARAMETER, NOT A CONSTANT
-----------------------------------------
Deeper = higher ceiling, more noise, linearly more compute. The ceiling is
literally ``recall@depth`` of the input run — this script prints it, and the
final score can never exceed it. If the printed ceiling is already low, stop:
you have a retriever problem and reranking cannot fix it.

SEGMENTATION
------------
PhoRanker and ViRanker are PhoBERT-backbone and REQUIRE segmented input;
Vietnamese_Reranker and bge-reranker-v2-m3 must NOT get it. ``src/dense.REGISTRY``
decides per model. Do not pre-segment.

USAGE
  python phases/3_rerank/rerank.py --run work/experiments/runs/hybrid-best.jsonl \
      --model AITeamVN/Vietnamese_Reranker --depth 50
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import cutoff, dense, exp_log, io_utils, metrics, normalize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--registry-as", default=None)
    ap.add_argument("--corpus", default="data/processed/corpus_article.jsonl")
    ap.add_argument("--queries", default="data/processed/queries_dev.jsonl")
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--cutoff", default="ratio")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    from sentence_transformers import CrossEncoder  # type: ignore

    fmt = a.registry_as or a.model
    sp = dense.spec(fmt)
    print(f"reranker  : {a.model}")
    print(f"input fmt : {sp.backbone} -> segmented={sp.segmented}")

    run = io_utils.load_run(a.run)
    queries = {q["qid"]: q for q in io_utils.load_queries(a.queries)}
    qrels = io_utils.qrels(list(queries.values()))
    doc_ids, texts, _ = io_utils.load_corpus(a.corpus)
    dtext = dict(zip(doc_ids, texts))

    labelled = any(qrels.values())
    ceiling = metrics.recall_at_k(run, qrels, a.depth) if labelled else float("nan")
    if labelled:
        print(f"CEILING   : recall@{a.depth} of the input run = {ceiling:.4f}")
        print(f"            the final score CANNOT exceed this.\n")
    else:
        print("queries are UNLABELLED (test split) — reranking will run, but there\n"
              "is no ceiling or score to report. Expected.\n")

    model = CrossEncoder(a.model, max_length=a.max_length, device=a.device)

    pairs, index = [], []
    for qid, ranked in run.items():
        if qid not in queries:
            continue
        qtext = normalize.encoder_text(queries[qid]["text"], sp.segmented)
        for doc, _ in ranked[:a.depth]:
            pairs.append((qtext, normalize.encoder_text(dtext.get(doc, ""), sp.segmented)))
            index.append((qid, doc))

    print(f"scoring {len(pairs)} (query, doc) pairs "
          f"= {len(run)} queries x depth {a.depth}")
    t0 = time.time()
    scores = model.predict(pairs, batch_size=a.batch_size, show_progress_bar=True)
    dt = time.time() - t0
    print(f"done in {dt:.0f}s ({len(pairs)/max(dt,1e-9):.0f} pairs/s)")

    reranked = {}
    for (qid, doc), s in zip(index, scores):
        reranked.setdefault(qid, []).append((doc, float(s)))
    # keep the tail below everything reranked, so recall@100 is preserved
    out = {}
    for qid, ranked in run.items():
        head = sorted(reranked.get(qid, []), key=lambda x: -x[1])
        seen = {d for d, _ in head}
        floor = min((s for _, s in head), default=0.0)
        tail = [(d, floor - 1e-6 * (i + 1))
                for i, (d, _) in enumerate(ranked[a.depth:]) if d not in seen]
        out[qid] = head + tail

    run_id = a.run_id or (f"rerank-{os.path.basename(a.model).replace('/','_')}"
                          f"-d{a.depth}")
    io_utils.write_run(f"work/experiments/runs/{run_id}.jsonl", out)

    if not labelled:
        print(f"\nrun_id : {run_id}\nwrote work/experiments/runs/{run_id}.jsonl")
        print("no scores (unlabelled queries).")
        return
    preds = cutoff.apply_to_run(out, rule=a.cutoff, alpha=a.alpha, k=a.k)
    s = metrics.official(preds, qrels)
    dg = metrics.diagnostics(out, qrels)
    print(f"\nrun_id : {run_id}")
    print(f"  RECALL (primary)   : {s['primary_recall']:.4f}  "
          f"(ceiling {ceiling:.4f}, {100*s['primary_recall']/max(ceiling,1e-9):.0f}% of it)")
    print(f"  PRECISION(tiebreak): {s['tiebreak_precision']:.4f}")
    print(f"  mrr@10 : {dg['mrr@10']:.4f}   ndcg@10: {dg['ndcg@10']:.4f}")
    print(f"  avg answer set     : {s['avg_pred_size']:.2f}")

    exp_log.log_run({
        "run_id": run_id, "phase": "3", "task": "1",
        "retriever": os.path.basename(a.run), "reranker": f"{a.model}@{a.depth}",
        "cutoff_rule": f"{a.cutoff}:{a.alpha}",
        "dev_P": s["tiebreak_precision"], "dev_R": s["primary_recall"],
        "dev_official": s["primary_recall"], "n_params": sp.params,
        "notes": f"ceiling@{a.depth}={ceiling:.4f} mrr@10={dg['mrr@10']:.4f} "
                 f"{len(pairs)/max(dt,1e-9):.0f}pairs/s",
    })
    print(f"\nlogged. NOW RE-SWEEP THE CUTOFF — cross-encoder scores are much "
          f"sharper than retriever scores:\n"
          f"  python phases/1_bm25/cutoff_sweep.py --run work/experiments/runs/{run_id}.jsonl --plot")


if __name__ == "__main__":
    main()
