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
Uses BTC's own scorer (vendored, `phases/0_harness/btc_eval/scoring_legalqa.py`):
METEOR primary, ROUGE-L secondary, both macro-averaged.

Needs `nltk` and the wordnet corpus — the first run downloads it. If the machine
is offline, pre-fetch with:
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

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

from src import io_utils  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def score_answers(pred_path: str, queries, answer_field: str = "answer"):
    """Score with BTC's OWN Task 2 code (vendored in phases/0_harness/btc_eval/).

    Their metric, verbatim from ``scoring_legalqa.py``:
        METEOR   (primary)  nltk.translate.meteor_score, on `.split()` tokens
        ROUGE-L  (secondary) rouge_score.RougeScorer(['rougeL'], use_stemmer=False)
        both averaged with .mean() over questions

    Two details that shape your whole approach:
      * They do NOT word-segment — the pyvi call is commented out in their
        source. Scoring is over whitespace-split Vietnamese syllables.
      * NLTK's METEOR is heavily RECALL-weighted (alpha=0.9) and penalises
        fragmentation. Long answers that cover the reference's content in the
        reference's order score well; terse answers are punished.
    """
    sys.path.insert(0, os.path.join(REPO, "phases", "0_harness"))
    from btc_eval.scoring_legalqa import eval_qa

    gold = {q["qid"]: str(q.get(answer_field, "")) for q in queries}
    preds = {r["qid"]: str(r.get("answer", "")) for r in io_utils.read_jsonl(pred_path)}
    # their scorer raises on a key-count mismatch; mirror the submission exactly
    y_pred = {q: {"answer": preds.get(q, "")} for q in gold}
    res = eval_qa(y_pred, gold)
    return {"meteor": res["meteor"], "rouge_l": res["rouge"], "n": len(gold)}


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
         "> Scored with BTC's own code (METEOR primary, ROUGE-L secondary).", "",
         "| prompt | top_k | METEOR (primary) | ROUGE-L |",
         "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['prompt']} | {r['top_k']} | "
                 f"{r['meteor']:.4f} | {r['rouge_l']:.4f} |")
    best = max(rows, key=lambda r: r["meteor"]) if rows else {}
    L += ["", f"Best: **{best.get('prompt')} @ top-{best.get('top_k')}**", "",
          "## Interpretation (fill in)", "",
          "- Did more context help monotonically? If not, at which k did it turn,",
          "  and is that consistent with the 'lost in the middle' explanation?",
          "- Which prompt variant won? METEOR is recall-weighted, so `concise`",
          "  losing to `cited`/`grounded` is the expected direction — check whether",
          "  your answer LENGTH tracks the reference length.",
          "- Compare mean answer length against mean reference length. A large gap",
          "  either way is usually worth more METEOR than any model change."]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
