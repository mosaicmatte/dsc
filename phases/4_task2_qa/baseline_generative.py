#!/usr/bin/env python3
"""Task B4 — Baseline B: generative reader over the retrieved passages.

BEFORE YOU RUN THIS
-------------------
1. `phases/4_task2_qa/00_task2_eval_notes.md` must be filled in. The answer format
   decides whether this baseline or the extractive one is even appropriate.
2. `retrieval_stage.py` must have been run, and you must know recall@k. This
   script prints it as a ceiling reminder; if it is low, stop and fix retrieval.
3. `python src/params.py` must show the full stack under 4B.

WHAT IT DOES
------------
For each question: take the top-k retrieved passages, build a prompt (see
`prompts.py`), generate, post-process, write predictions.

Greedy decoding by default (`--temperature 0`). For an extraction-shaped task
sampling adds variance without adding accuracy, and it makes your run
non-reproducible — which matters for the Phase 5 package.

WHAT TO WATCH
-------------
`--max-new-tokens` interacts with the metric. If Task 2 is scored by token-F1, a
long correct answer is penalised on precision against a short gold string; the
`concise` prompt variant plus a low token cap often beats a better-reasoned long
answer. Measure it — that is Task B6.

USAGE
  python phases/4_task2_qa/baseline_generative.py \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --run work/experiments/runs/task2-retrieval.jsonl --top-k 3 --prompt grounded
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prompts as P  # noqa: E402

from src import config, exp_log, io_utils, metrics  # noqa: E402


def postprocess(text: str, strip_prefixes=("Trả lời:", "Answer:")) -> str:
    """Trim the model's preamble. Keep this deterministic and logged — an
    undocumented post-processing rule is a reproduction-package failure."""
    t = text.strip()
    for p in strip_prefixes:
        if t.startswith(p):
            t = t[len(p):].strip()
    return t.split("\n\n")[0].strip()


def generate_answers(model_name, questions, contexts, ctx_ids, variant,
                     max_new_tokens, temperature, batch_size, device,
                     max_ctx_chars):
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # required for correct batched generation
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device or ("auto" if torch.cuda.is_available() else None))
    model.eval()

    texts = []
    for q, ctx, ids in zip(questions, contexts, ctx_ids):
        msgs = P.as_chat(q, ctx, variant, ids, max_ctx_chars)
        try:
            texts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True))
        except Exception:      # base (non-instruct) model without a chat template
            texts.append(P.build(q, ctx, variant, ids, max_ctx_chars))

    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=4096).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(postprocess(tok.decode(new, skip_special_tokens=True)))
        print(f"  generated {min(i+batch_size, len(texts))}/{len(texts)}",
              end="\r", flush=True)
    print()
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--run", default="work/experiments/runs/task2-retrieval.jsonl")
    ap.add_argument("--queries", default="data/processed/task2_dev.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus_article.jsonl")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--prompt", default="grounded", choices=sorted(P.TEMPLATES))
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-ctx-chars", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="debug on N questions")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    config.set_seed(a.seed)

    run = io_utils.load_run(a.run)
    queries = io_utils.load_queries(a.queries)
    if a.limit:
        queries = queries[:a.limit]
    doc_ids, texts, _ = io_utils.load_corpus(a.corpus)
    dtext = dict(zip(doc_ids, texts))

    qrels = io_utils.qrels(queries)
    if any(qrels.values()):
        ceil = metrics.recall_at_k(run, qrels, a.top_k)
        print(f"CEILING: recall@{a.top_k} = {ceil:.4f} — "
              f"{100*(1-ceil):.0f}% of questions have no gold passage in context "
              f"and cannot be answered correctly.\n")

    qs, ctxs, cids = [], [], []
    for q in queries:
        top = run.get(q["qid"], [])[:a.top_k]
        qs.append(q["text"])
        ctxs.append([dtext.get(d, "") for d, _ in top])
        cids.append([d for d, _ in top])

    print(f"model {a.model}  prompt={a.prompt}  top_k={a.top_k}  "
          f"max_new_tokens={a.max_new_tokens}  temperature={a.temperature}")
    answers = generate_answers(a.model, qs, ctxs, cids, a.prompt,
                               a.max_new_tokens, a.temperature, a.batch_size,
                               a.device, a.max_ctx_chars)

    run_id = a.run_id or (f"task2-gen-{os.path.basename(a.model)}-"
                          f"k{a.top_k}-{a.prompt}")
    out = a.out or f"work/experiments/predictions/{run_id}.jsonl"
    io_utils.write_jsonl(out, (
        {"qid": q["qid"], "question": q["text"], "answer": ans,
         "context_ids": ids}
        for q, ans, ids in zip(queries, answers, cids)))

    print(f"\nwrote {out}")
    print("\n--- 3 samples (READ THESE before trusting any metric) ---")
    for q, ans, ids in list(zip(queries, answers, cids))[:3]:
        print(f"Q: {q['text']}")
        print(f"A: {ans}")
        print(f"   (from {ids})\n")

    exp_log.log_run({
        "run_id": run_id, "phase": "4", "task": "2",
        "retriever": os.path.basename(a.run), "reranker": "-",
        "cutoff_rule": f"top_k={a.top_k}",
        "notes": f"generator={a.model} prompt={a.prompt} "
                 f"max_new={a.max_new_tokens} temp={a.temperature}",
    })
    print(f"logged. Score it with BTC's Task 2 metric "
          f"(see phases/4_task2_qa/00_task2_eval_notes.md).")


if __name__ == "__main__":
    main()
