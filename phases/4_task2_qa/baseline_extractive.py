#!/usr/bin/env python3
"""Task B3 — Baseline A: extractive reader.

USE THIS ONLY IF gold answers are verbatim spans of the corpus. Question Q2 in
`00_task2_eval_notes.md` settles that mechanically — do not eyeball it.

WHY IT CAN BEAT A GENERATIVE MODEL
-----------------------------------
* It cannot hallucinate: the output is a substring of a retrieved passage.
* ~135M parameters against 1.5B, leaving budget elsewhere.
* If the metric is exact match against a span, a generative model's paraphrase
  scores zero even when it is factually right.

WHEN IT CANNOT WORK
-------------------
If gold answers are rephrased or synthesised across passages, extraction has a
hard ceiling below 1.0 and no amount of tuning reaches it. Measure that ceiling
first: `--oracle-ceiling` reports the fraction of gold answers that appear
verbatim in ANY retrieved passage. That number is the maximum this baseline can
possibly score.

TWO MODES
---------
--mode span   a QA head predicts start/end offsets. Needs a QA-finetuned
              checkpoint; there is no reliable Vietnamese legal one, so expect to
              train it on train.json if BTC's answers are spans.
--mode passage  return the whole top-1 passage as the answer. A crude but real
              baseline, and sometimes competitive when the metric is token-F1
              over long gold answers. Costs zero parameters — run it first.

USAGE
  python phases/4_task2_qa/baseline_extractive.py --mode passage --top-k 1
  python phases/4_task2_qa/baseline_extractive.py --oracle-ceiling
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import exp_log, io_utils, normalize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="work/experiments/runs/task2-retrieval.jsonl")
    ap.add_argument("--queries", default="data/processed/task2_dev.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus_article.jsonl")
    ap.add_argument("--mode", default="passage", choices=["passage", "span"])
    ap.add_argument("--model", default=None, help="QA checkpoint, for --mode span")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--answer-field", default="answer",
                    help="field in the queries file holding the gold answer")
    ap.add_argument("--oracle-ceiling", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()

    run = io_utils.load_run(a.run)
    queries = io_utils.load_queries(a.queries)
    doc_ids, texts, _ = io_utils.load_corpus(a.corpus)
    dtext = dict(zip(doc_ids, texts))

    if a.oracle_ceiling:
        # The hard limit on any extractive approach: can the gold answer even be
        # found verbatim in the retrieved text?
        hit = tot = 0
        for q in queries:
            gold = q.get(a.answer_field)
            if not gold:
                continue
            tot += 1
            ctx = " ".join(dtext.get(d, "") for d, _ in run.get(q["qid"], [])[:a.top_k])
            if normalize.normalize(str(gold)) in normalize.normalize(ctx):
                hit += 1
        if not tot:
            print(f"no '{a.answer_field}' field found in {a.queries} — "
                  f"pass --answer-field")
            return
        print(f"ORACLE EXTRACTIVE CEILING @top-{a.top_k}: {hit}/{tot} = "
              f"{hit/tot:.4f}")
        print("This is the maximum an extractive reader can score. If it is low,\n"
              "gold answers are not verbatim spans — use the generative baseline.")
        return

    if a.mode == "passage":
        answers = {}
        for q in queries:
            top = run.get(q["qid"], [])[:a.top_k]
            answers[q["qid"]] = " ".join(dtext.get(d, "") for d, _ in top)
    else:
        if not a.model:
            raise SystemExit("--mode span needs --model (a QA checkpoint)")
        from transformers import pipeline  # type: ignore
        qa = pipeline("question-answering", model=a.model)
        answers = {}
        for q in queries:
            ctx = " ".join(dtext.get(d, "") for d, _ in run.get(q["qid"], [])[:a.top_k])
            answers[q["qid"]] = qa(question=q["text"], context=ctx[:4000])["answer"] \
                if ctx else ""

    run_id = a.run_id or f"task2-extractive-{a.mode}-k{a.top_k}"
    out = a.out or f"work/experiments/predictions/{run_id}.jsonl"
    io_utils.write_jsonl(out, ({"qid": q["qid"], "question": q["text"],
                                "answer": answers.get(q["qid"], "")}
                               for q in queries))
    print(f"wrote {out}")
    exp_log.log_run({"run_id": run_id, "phase": "4", "task": "2",
                     "retriever": os.path.basename(a.run),
                     "cutoff_rule": f"top_k={a.top_k}",
                     "n_params": 0 if a.mode == "passage" else "",
                     "notes": f"extractive mode={a.mode} model={a.model}"})


if __name__ == "__main__":
    main()
