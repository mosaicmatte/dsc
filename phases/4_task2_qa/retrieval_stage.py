#!/usr/bin/env python3
"""Task B2 — port the Task 1 pipeline to Task 2 questions, and MEASURE it.

WHY THIS RUNS BEFORE ANY READER WORK
------------------------------------
The reader can only answer from what it is handed. If the gold passage is not in
the top-k, the model will produce a fluent, confident, wrong answer — and you
will spend two days tuning the generator for a retrieval problem.

Task 2 questions are NOT the same distribution as Task 1 queries: often longer,
more scenario-shaped, less likely to quote statute numbers verbatim. A retriever
tuned on Task 1 may transfer poorly. Find that out as a number, now.

WHAT TO RECORD
--------------
recall@1, @3, @5 — these are the ceilings for a reader given 1, 3 or 5 passages.
If recall@5 is 0.60, then 40% of your questions are unanswerable no matter what
the generator does, and no prompt engineering will move them.

USAGE
  python phases/4_task2_qa/retrieval_stage.py \
      --queries data/processed/task2_dev.jsonl \
      --model AITeamVN/Vietnamese_Embedding \
      --reranker AITeamVN/Vietnamese_Reranker --depth 100 --rerank-depth 50
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import dense, exp_log, io_utils, metrics, normalize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default="data/processed/task2_dev.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus_article.jsonl")
    ap.add_argument("--model", required=True, help="bi-encoder from Task 1")
    ap.add_argument("--registry-as", default=None)
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--reranker-as", default=None)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--rerank-depth", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--run-id", default="task2-retrieval")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    fmt = a.registry_as or a.model
    queries = io_utils.load_queries(a.queries)
    doc_ids, texts, _ = io_utils.load_corpus(a.corpus)

    model = dense.load_encoder(a.model, device=a.device)
    doc_emb = dense.encode(model, texts, fmt, batch_size=a.batch_size)
    q_emb = dense.encode(model, [q["text"] for q in queries], fmt, is_query=True,
                         batch_size=a.batch_size)
    run = dense.search(q_emb, doc_emb, doc_ids, [q["qid"] for q in queries],
                       top_k=a.depth)

    qrels = io_utils.qrels(queries)
    has_labels = any(qrels.values())
    if has_labels:
        d = metrics.diagnostics(run, qrels, ks=(1, 3, 5, 10, 50, 100))
        print("retriever only:")
        for k in (1, 3, 5, 10, 50, 100):
            print(f"  recall@{k:<4}: {d[f'recall@{k}']:.4f}")

    if a.reranker:
        from sentence_transformers import CrossEncoder  # type: ignore
        rfmt = a.reranker_as or a.reranker
        rsp = dense.spec(rfmt)
        dtext = dict(zip(doc_ids, texts))
        qtext = {q["qid"]: q["text"] for q in queries}
        pairs, index = [], []
        for qid, ranked in run.items():
            qt = normalize.encoder_text(qtext[qid], rsp.segmented)
            for doc, _ in ranked[:a.rerank_depth]:
                pairs.append((qt, normalize.encoder_text(dtext.get(doc, ""),
                                                         rsp.segmented)))
                index.append((qid, doc))
        ce = CrossEncoder(a.reranker, max_length=512, device=a.device)
        scores = ce.predict(pairs, batch_size=a.batch_size, show_progress_bar=True)
        rr = {}
        for (qid, doc), s in zip(index, scores):
            rr.setdefault(qid, []).append((doc, float(s)))
        for qid in run:
            head = sorted(rr.get(qid, []), key=lambda x: -x[1])
            seen = {d for d, _ in head}
            floor = min((s for _, s in head), default=0.0)
            run[qid] = head + [(d, floor - 1e-6 * (i + 1))
                               for i, (d, _) in enumerate(run[qid][a.rerank_depth:])
                               if d not in seen]

    io_utils.write_run(f"work/experiments/runs/{a.run_id}.jsonl", run)
    print(f"\nwrote work/experiments/runs/{a.run_id}.jsonl")

    if has_labels:
        d = metrics.diagnostics(run, qrels, ks=(1, 3, 5, 10, 50, 100))
        print("\nFINAL — these are the CEILINGS for the reader:")
        for k in (1, 3, 5):
            print(f"  recall@{k}  -> a reader given top-{k} passages "
                  f"cannot exceed {d[f'recall@{k}']:.4f}")
        for k in (10, 50, 100):
            print(f"  recall@{k:<4}: {d[f'recall@{k}']:.4f}")
        exp_log.log_run({
            "run_id": a.run_id, "phase": "4", "task": "2",
            "retriever": a.model, "reranker": a.reranker or "-",
            "dev_R": d["recall@5"],
            "notes": f"TASK2 RETRIEVAL CEILING r@1={d['recall@1']:.4f} "
                     f"r@3={d['recall@3']:.4f} r@5={d['recall@5']:.4f}",
        })
        if d["recall@5"] < 0.7:
            print("\nWARNING: recall@5 below 0.70. More than 30% of questions are\n"
                  "unanswerable from the top-5 context. Fix retrieval before\n"
                  "spending any time on the reader.")
    else:
        print("\n(no labels in this file — run on a labelled split to get the ceiling)")


if __name__ == "__main__":
    main()
