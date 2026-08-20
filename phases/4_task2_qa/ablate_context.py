#!/usr/bin/env python3
"""Tasks B5/B6 — ablate context size and prompt format. Cheapest wins in Phase 4.

WHY CONTEXT SIZE IS NOT MONOTONIC
---------------------------------
Adding passages pulls in two opposite directions:
  + the gold passage is more likely to be present (recall@k rises with k)
  - more distractors, and small models attend badly over long contexts. A gold
    passage at position 4 of 5 can be effectively invisible ("lost in the
    middle").

For a 1.5B model the optimum is often top-3. The point of this script is that you
report the curve, not a guess — including the case where more context made it
worse, which is the interesting result and a paper row.

SCORING
-------
This script needs BTC's Task 2 metric. Fill in `score_answers()` below from
`00_task2_eval_notes.md`. Until then it reports the built-in token-F1 and exact
match, clearly labelled as UNOFFICIAL — do not put unofficial numbers in the
paper without saying so.

USAGE
  python phases/4_task2_qa/ablate_context.py --model Qwen/Qwen2.5-1.5B-Instruct --top-k 1 3 5
  python phases/4_task2_qa/ablate_context.py --model <m> --prompts all --top-k 3
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prompts as P  # noqa: E402

from src import io_utils, normalize  # noqa: E402


def _tokens(s):
    return normalize.tokenize(s or "")


def token_f1(pred: str, gold: str) -> float:
    """UNOFFICIAL. Replace with BTC's metric once known."""
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return float(p == g)
    common = 0
    gc = list(g)
    for t in p:
        if t in gc:
            gc.remove(t)
            common += 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize.normalize(pred or "") == normalize.normalize(gold or ""))


def score_answers(pred_path: str, queries, answer_field: str):
    """TODO(BLOCKER/phase4-B1): swap in BTC's official Task 2 metric.

    HOW: same pattern as `phases/0_harness/evaluate.py:btc_official_score` —
    import their scorer, or shell out to their CLI and parse stdout.
    """
    gold = {q["qid"]: q.get(answer_field, "") for q in queries}
    preds = {r["qid"]: r.get("answer", "") for r in io_utils.read_jsonl(pred_path)}
    n = len(gold) or 1
    return {
        "token_f1_UNOFFICIAL": sum(token_f1(preds.get(q, ""), g)
                                   for q, g in gold.items()) / n,
        "exact_match_UNOFFICIAL": sum(exact_match(preds.get(q, ""), g)
                                      for q, g in gold.items()) / n,
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--run", default="work/experiments/runs/task2-retrieval.jsonl")
    ap.add_argument("--queries", default="data/processed/task2_dev.jsonl")
    ap.add_argument("--answer-field", default="answer")
    ap.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--prompts", nargs="+", default=["grounded"],
                    help="prompt variants, or 'all'")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="work/analysis/task2_ablation.md")
    a = ap.parse_args()

    variants = sorted(P.TEMPLATES) if a.prompts == ["all"] else a.prompts
    queries = io_utils.load_queries(a.queries)
    if a.limit:
        queries = queries[:a.limit]

    gen = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "baseline_generative.py")
    rows = []
    for variant in variants:
        for k in a.top_k:
            run_id = f"abl-k{k}-{variant}"
            cmd = [sys.executable, gen, "--model", a.model, "--run", a.run,
                   "--queries", a.queries, "--top-k", str(k),
                   "--prompt", variant, "--run-id", run_id,
                   "--max-new-tokens", str(a.max_new_tokens)]
            if a.limit:
                cmd += ["--limit", str(a.limit)]
            print(f"\n=== {run_id} ===")
            subprocess.run(cmd, check=True)
            s = score_answers(f"work/experiments/predictions/{run_id}.jsonl",
                              queries, a.answer_field)
            rows.append({"prompt": variant, "top_k": k, **s})
            print(f"  -> {s}")

    L = ["# Task 2 ablation — context size x prompt format", "",
         f"Model: `{a.model}`  ·  retrieval: `{os.path.basename(a.run)}`", "",
         "> Metrics marked UNOFFICIAL are this repo's approximation. Replace",
         "> `score_answers()` with BTC's scorer before quoting these anywhere.", "",
         "| prompt | top_k | token F1 (unoff.) | exact match (unoff.) |",
         "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['prompt']} | {r['top_k']} | "
                 f"{r['token_f1_UNOFFICIAL']:.4f} | {r['exact_match_UNOFFICIAL']:.4f} |")
    best = max(rows, key=lambda r: r["token_f1_UNOFFICIAL"]) if rows else {}
    L += ["", f"Best: **{best.get('prompt')} @ top-{best.get('top_k')}**", "",
          "## Interpretation (fill in)", "",
          "- Did more context help monotonically? If not, at which k did it turn,",
          "  and is that consistent with the 'lost in the middle' explanation?",
          "- Which prompt variant won, and does that tell you the failure mode was",
          "  hallucination (grounded wins) or verbosity against a short gold string",
          "  (concise wins)?"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
