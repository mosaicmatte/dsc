#!/usr/bin/env python3
"""Task B2 — fine-tune a cross-encoder on YOUR retriever's errors.

WHY NOT RANDOM NEGATIVES
------------------------
At inference the reranker only ever sees the retriever's top-k. Training it on
randomly sampled negatives trains it on a distribution it will never encounter:
it becomes excellent at rejecting documents about traffic law when the query is
about labour law, which the retriever already filtered out, and never learns to
separate the two adjacent điều that are actually the hard call.

So negatives here come from the run file — the documents your retriever ranked
highly and got wrong. That is the exact distribution the reranker will face.

OBJECTIVE
---------
Binary cross-entropy on pairs (label 1 for gold, 0 for mined negatives). Chosen
over listwise because the resulting logits are **calibrated across queries**,
which keeps the `threshold` cutoff rule usable alongside `ratio`. Listwise
usually scores a little better but its scores are only comparable within a query.

RATIO CONTROL
-------------
``--neg-per-pos`` sets the class balance. Too few negatives and the model learns
to say "relevant" to everything; too many and it collapses to "irrelevant". 4-8
is the usual range — it is a hyperparameter, so log what you used.

USAGE
  python phases/3_rerank/train_cross_encoder.py \
      --model AITeamVN/Vietnamese_Reranker \
      --run work/experiments/runs/hybrid-best.jsonl --epochs 2 --neg-per-pos 6
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import config, dense, io_utils, normalize  # noqa: E402


def build_examples(run, queries, dtext, sp, n_neg, skip_top, depth, rng):
    """Label 1 for gold, 0 for retriever mistakes. Returns InputExamples."""
    from sentence_transformers import InputExample  # type: ignore

    ex, n_pos, n_neg_tot, no_cand = [], 0, 0, 0
    for q in queries:
        rel = set(q["relevant"])
        if not rel:
            continue
        qt = normalize.encoder_text(q["text"], sp.segmented)
        for pos in rel:
            if pos in dtext:
                ex.append(InputExample(
                    texts=[qt, normalize.encoder_text(dtext[pos], sp.segmented)],
                    label=1.0))
                n_pos += 1
        cands = [d for i, (d, _) in enumerate(run.get(q["qid"], [])[:depth])
                 if d not in rel and i >= skip_top and d in dtext]
        if not cands:
            no_cand += 1
            continue
        take = min(n_neg * max(len(rel), 1), len(cands))
        for d in rng.sample(cands, take):
            ex.append(InputExample(
                texts=[qt, normalize.encoder_text(dtext[d], sp.segmented)],
                label=0.0))
            n_neg_tot += 1
    print(f"examples: {n_pos} positive, {n_neg_tot} negative "
          f"(ratio 1:{n_neg_tot/max(n_pos,1):.1f})")
    if no_cand:
        print(f"WARNING: {no_cand} queries yielded no negatives — "
              f"lower --skip-top or raise --depth")
    return ex


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True, help="YOUR retriever's run — negatives come from here")
    ap.add_argument("--queries", default="data/processed/queries_train_split.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus_article.jsonl")
    ap.add_argument("--out", default="models/cross-encoder-ft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--neg-per-pos", type=int, default=6)
    ap.add_argument("--skip-top", type=int, default=0,
                    help="skip the N top-ranked candidates (false-negative guard)")
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    config.set_seed(a.seed)
    rng = random.Random(a.seed)

    from sentence_transformers import CrossEncoder  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore

    sp = dense.spec(a.model)
    print(f"model: {a.model}  ({sp.backbone}, segmented={sp.segmented})")

    run = io_utils.load_run(a.run)
    queries = io_utils.load_queries(a.queries)
    doc_ids, texts, _ = io_utils.load_corpus(a.corpus)
    dtext = dict(zip(doc_ids, texts))

    ex = build_examples(run, queries, dtext, sp, a.neg_per_pos, a.skip_top,
                        a.depth, rng)
    if not ex:
        raise SystemExit("no training examples built — check --run and --queries")

    model = CrossEncoder(a.model, num_labels=1, max_length=a.max_length)
    loader = DataLoader(ex, shuffle=True, batch_size=a.batch_size, drop_last=True)
    warmup = int(len(loader) * a.epochs * a.warmup_ratio)

    print(f"training: {len(ex)} examples, batch {a.batch_size}, {a.epochs} epochs, "
          f"lr {a.lr}, warmup {warmup}")
    model.fit(train_dataloader=loader, epochs=a.epochs, warmup_steps=warmup,
              optimizer_params={"lr": a.lr}, use_amp=a.fp16, output_path=a.out,
              show_progress_bar=True)

    cfg = vars(a)
    run_id = config.make_run_id(cfg, prefix="train-ce")
    config.freeze(cfg, run_id)
    print(f"\nsaved -> {a.out}\nfroze config -> configs/{run_id}.yaml")
    print(f"\nEVALUATE (keep the zero-shot row for the ablation):\n"
          f"  python phases/3_rerank/rerank.py --run {a.run} --model {a.out} "
          f"--registry-as {a.model} --depth {a.depth} --run-id rerank-ft")


if __name__ == "__main__":
    main()
