#!/usr/bin/env python3
"""Tasks B2/B3/B4 — fine-tune the bi-encoder with MultipleNegativesRankingLoss.

THE THREE ROUNDS
----------------
round 1  (query, positive)                     in-batch negatives only
round 2  (query, positive, *hard_negatives)    negatives mined from BM25
round 3  (query, positive, *hard_negatives)    negatives mined from round 2's own model

Each round starts from the BASE model, not from the previous round's weights,
unless you pass --init. Restarting keeps the ablation clean: "round 3 beat
round 2" then means the negatives were better, not that it trained twice as long.

WHY BATCH SIZE IS THE HYPERPARAMETER THAT MATTERS
-------------------------------------------------
MultipleNegativesRankingLoss treats the other examples in the batch as
negatives. Batch 64 -> 63 negatives per query, free. Batch 16 -> 15. It is a
softmax over the batch, so a larger batch is a strictly harder classification
problem and produces sharper representations. Push it as high as the GPU allows:
enable --fp16 and --grad-checkpointing before lowering it.

WITH hard negatives, each example contributes (1 positive + n_neg hard) columns,
so effective difficulty rises fast and memory rises with it. Batch 16 with 8 hard
negatives is a harder task than batch 64 with none.

WHAT TO DO
----------
    # round 1
    python phases/2_dense/train_biencoder.py --model AITeamVN/Vietnamese_Embedding \
        --round 1 --batch-size 64 --epochs 2
    # rounds 2/3
    python phases/2_dense/train_biencoder.py --model AITeamVN/Vietnamese_Embedding \
        --round 2 --pairs data/processed/train_pairs_bm25neg.jsonl --batch-size 16

Then evaluate with:
    python phases/2_dense/zero_shot_eval.py --model models/biencoder-r2 \
        --registry-as AITeamVN/Vietnamese_Embedding --run-id dense-r2 --no-cache

(--registry-as tells the evaluator which input format the fine-tuned checkpoint
inherits, and --no-cache stops it reusing the base model's cached embeddings —
forgetting that flag means evaluating the OLD weights and concluding fine-tuning
did nothing.)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src import config, dense, io_utils  # noqa: E402


def build_examples(round_no, pairs_path, queries_path, corpus_path, model_name,
                   max_neg):
    """Return sentence-transformers InputExamples in the shape MNRL expects.

    MNRL reads each example's texts as [anchor, positive, neg1, neg2, ...].
    Every example in a batch must have the SAME number of texts, so hard
    negatives are padded/truncated to exactly ``max_neg``.
    """
    from sentence_transformers import InputExample  # type: ignore

    sp = dense.spec(model_name)
    def prep(t, is_q=False):
        return dense.prepare([t], model_name, is_query=is_q)[0]

    ex = []
    if round_no == 1:
        queries = io_utils.load_queries(queries_path)
        doc_ids, texts, _ = io_utils.load_corpus(corpus_path)
        dtext = dict(zip(doc_ids, texts))
        for q in queries:
            for pos in q["relevant"]:
                if pos in dtext:
                    ex.append(InputExample(texts=[prep(q["text"], True),
                                                  prep(dtext[pos])]))
    else:
        if not pairs_path:
            raise SystemExit("rounds 2 and 3 need --pairs from mine_hard_negatives.py")
        for r in io_utils.read_jsonl(pairs_path):
            negs = r["negatives"][:max_neg]
            if len(negs) < max_neg:
                continue     # keep the tensor shape uniform across the batch
            ex.append(InputExample(texts=[prep(r["query"], True),
                                          prep(r["positive"]),
                                          *[prep(n) for n in negs]]))
    print(f"built {len(ex)} training examples "
          f"({'pairs' if round_no == 1 else f'1 pos + {max_neg} hard negs'}), "
          f"segmented={sp.segmented}")
    return ex


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="AITeamVN/Vietnamese_Embedding")
    ap.add_argument("--init", default=None, help="resume from a checkpoint instead of base")
    ap.add_argument("--round", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--pairs", default=None, help="output of mine_hard_negatives.py")
    ap.add_argument("--queries", default="data/processed/queries_train_split.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus_article.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-neg", type=int, default=4)
    ap.add_argument("--scale", type=float, default=20.0,
                    help="MNRL scale = 1/temperature; 20 == tau 0.05")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    config.set_seed(a.seed)

    from sentence_transformers import losses  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore

    out = a.out or f"models/biencoder-r{a.round}"
    model = dense.load_encoder(a.init or a.model)
    if a.grad_checkpointing:
        try:
            model[0].auto_model.gradient_checkpointing_enable()
            print("gradient checkpointing enabled")
        except Exception as e:  # noqa: BLE001
            print(f"could not enable gradient checkpointing: {e}", file=sys.stderr)

    ex = build_examples(a.round, a.pairs, a.queries, a.corpus, a.model, a.max_neg)
    if not ex:
        raise SystemExit("no training examples — check --pairs / --queries")

    loader = DataLoader(ex, shuffle=True, batch_size=a.batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model, scale=a.scale)
    warmup = int(len(loader) * a.epochs * a.warmup_ratio)

    print(f"round {a.round}: {len(ex)} examples, batch {a.batch_size}, "
          f"{a.epochs} epochs, lr {a.lr}, warmup {warmup} steps")
    print(f"in-batch negatives per query: {a.batch_size - 1}"
          + (f" (+{a.max_neg} mined hard)" if a.round > 1 else ""))

    model.fit(train_objectives=[(loader, loss)], epochs=a.epochs,
              warmup_steps=warmup, optimizer_params={"lr": a.lr},
              use_amp=a.fp16, output_path=out, show_progress_bar=True)

    os.makedirs(out, exist_ok=True)
    cfg = vars(a)
    run_id = config.make_run_id(cfg, prefix=f"train-r{a.round}")
    config.freeze(cfg, run_id)
    print(f"\nsaved model -> {out}")
    print(f"froze config -> configs/{run_id}.yaml")
    print(f"\nEVALUATE (note both flags):\n"
          f"  python phases/2_dense/zero_shot_eval.py --model {out} "
          f"--registry-as {a.model} --run-id dense-r{a.round} --no-cache")


if __name__ == "__main__":
    main()
