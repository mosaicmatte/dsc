# 03 — Dense retrieval, in more depth

Prerequisite: `phases/2_dense/README.md` Part A.

---

## 1. The asymmetry that makes retrieve-then-rerank inevitable

Let `N` = corpus size, `Q` = queries, `f` = one encoder forward pass.

| | offline cost | per-query cost |
|---|---|---|
| bi-encoder | `N·f` (once, ever) | `1·f` + one `N`-row dot product |
| cross-encoder | none possible | `N·f` |

The bi-encoder's document vector `v_d` does not depend on the query, so it is computed once
and reused for every query for the rest of the competition. The cross-encoder's
representation of `d` *does* depend on `q` — that is precisely why it is more accurate, and
precisely why nothing can be cached.

Concretely, with `N = 60,000` and a 568M cross-encoder at ~500 pairs/s:

```
full cross-encoder scan, 1 query : 60,000 / 500 = 120 s
                        600 queries        = 20 hours
retrieve-then-rerank, 600 queries  : 600 × 100 / 500 = 2 minutes
```

600× less work, and the accuracy loss is bounded by `recall@100` of the retriever — which
you can measure. That measured bound is the retrieval ceiling.

## 2. InfoNCE, and why batch size changes the task

```
L = − log [ exp(sim(q, d⁺)/τ) / Σ_j exp(sim(q, d_j)/τ) ]
```

The denominator runs over one positive and all negatives available. With in-batch
negatives, "available" means *the other examples in this batch*.

So batch size does not merely reduce gradient noise here — it changes the **number of
classes** in the classification problem:

| batch | task | difficulty |
|---|---|---|
| 16 | pick the right document out of 16 | easy; a coarse representation suffices |
| 64 | out of 64 | harder |
| 256 | out of 256 | much harder; forces fine distinctions |

At inference the model faces ~60,000-way discrimination. The closer training gets to that
regime, the better the transfer. **Exhaust fp16 and gradient checkpointing before lowering
batch size.**

Adding `n` mined hard negatives per example adds `n` columns per example to the softmax, so
batch 16 with 8 hard negatives is a harder problem than batch 64 with none — and it is
harder in the *useful* direction, because the extra columns are near-misses rather than
random documents.

**Temperature.** `τ` scales the logits. Small `τ` (0.02–0.05) sharpens the softmax, so
near-misses produce large gradients — good with clean labels, dangerous with false
negatives, because a mislabelled near-miss now dominates the update.
`MultipleNegativesRankingLoss` exposes `scale = 1/τ`, default 20 (τ = 0.05).

## 3. Why random negatives stop teaching after one epoch

The gradient of the InfoNCE loss with respect to a negative is proportional to its softmax
weight `exp(sim(q,d_j)/τ) / Σ`. A random negative from a different legal domain scores far
below the positive, so its weight is ~0 — it contributes essentially nothing to the update.

After the first epoch, virtually all random negatives are in that regime. Training
continues, loss keeps ticking down, and the model stops improving on anything that matters.
**Hard negatives are the fix precisely because their softmax weight is non-negligible.**

## 4. The mining ladder, and the false-negative trap

```
round 1   in-batch only          free negatives, gets you most of the way
round 2   BM25 top-k minus gold  lexically confusable documents
round 3   your own top-k minus gold   your model's actual current mistakes
```

Each round's negatives are drawn from a better retriever, so they sit closer to the
decision boundary.

**The trap.** A mined "negative" that is actually relevant but unlabelled teaches the model
that a correct answer is wrong. In legal corpora this is *common*, not hypothetical:
near-duplicate provisions appear across amended versions and parallel statutes.

Two defences, both in `mine_hard_negatives.py`:

- `--skip-top N` — discard the top N candidates. Ranks 1–2 of a good retriever are the most
  likely unlabelled positives. Raise this as the retriever improves: 2 for BM25, 3+ for
  round-3 self-mining.
- `--margin EPS` — discard candidates scoring within `EPS` of the gold document. If the
  model cannot separate them, they are probably duplicates.

And one that is not optional: **read ten of them.**
```bash
python phases/2_dense/mine_hard_negatives.py --run <run> --out <pairs> --inspect 10
```

## 5. Diagnosing a fine-tune that did not help

| Symptom | Likely cause | Test |
|---|---|---|
| dev identical to zero-shot, to 4 decimals | you evaluated cached embeddings | pass `--no-cache` |
| training loss near zero within 100 steps | negatives too easy | check batch size; move to round 2 |
| loss noisy, dev degrades | negatives too hard / false negatives | raise `--skip-top`, `--inspect` them |
| recall@10 up, recall@100 flat | model is reordering, not finding | expected and fine — the reranker will use it |
| recall@100 **down** vs zero-shot | over-fitted to the training distribution | fewer epochs, lower lr, more data |
| everything worse, uniformly | input format wrong (segmentation) | check `src/dense.REGISTRY` |

The first row is worth internalising. `zero_shot_eval.py` caches corpus embeddings keyed by
`(model_name, corpus_path)`; a fine-tuned checkpoint at a *new* path gets a new key, but if
you overwrite the same output directory the cache goes stale. `--no-cache` after every
training run.

## 6. Fusion: why margins matter downstream

RRF discards score magnitudes and keeps only ranks. That makes it robust — no calibration
needed between an unbounded BM25 score and a cosine similarity — but it also **flattens the
score distribution**, because `1/(K+rank)` changes slowly.

That has a concrete downstream consequence: the `ratio` cutoff rule keeps documents scoring
above `α × top`. With RRF's compressed range, almost everything clears that bar, and answer
sets balloon. You can see it in the Phase 2 fusion smoke test: RRF's precision collapses
under the same α that worked for raw BM25 scores.

So: **RRF often needs a different (higher) α, or a `top_k` cutoff instead.** Weighted score
fusion preserves margins and tends to play better with `ratio`. Sweep both, and re-sweep the
cutoff after choosing — this is the single most common place people forget to.

## 7. When the best fusion weight is 0 or 1

`hybrid.py --sweep` warns about this. It means one system dominates and fusion is not
helping. Do not ship a "hybrid" that is really one model — report it. Common reasons:

- one system is simply much stronger on this data (say so; it is a result)
- the weak system's run was built over a different split or granularity (a bug — check ids)
- score normalisation is inappropriate: `minmax` on a run where one query has all-equal
  scores maps everything to 1.0

---

## Check yourself

1. Why does a bi-encoder let you precompute the corpus while a cross-encoder does not — in
   one sentence about `v_d`?
2. You raise batch size from 32 to 128 and dev recall improves 4 points, with no other
   change. Is that a lucky seed?
3. Round-3 self-mined negatives make dev *worse* than round 2. Give the two explanations and
   the cheapest test that separates them.

<details><summary>answers</summary>

1. The bi-encoder's `v_d` is a function of `d` alone, so it is query-independent and can be
   computed once and indexed; the cross-encoder's representation of `d` is a function of
   `(q, d)` jointly, so it does not exist until the query arrives.
2. **Almost certainly not luck.** With `MultipleNegativesRankingLoss` the batch *is* the
   negative set, so 32→128 turns a 32-way classification problem into a 128-way one. The
   task got harder and the representations had to get sharper. This is the expected
   direction and magnitude; re-run with a second seed if you want to be sure, but do not be
   surprised.
3. (a) **False negatives** — the round-2 retriever ranks unlabelled positives highly and you
   mined them as negatives. (b) **Too hard** — negatives sit on the decision boundary,
   gradients get noisy, training degrades. **Cheapest test: `--inspect 20` and read them.**
   If any plainly answers its query, it is (a). Failing that, raise `--skip-top`: if the
   score recovers it is (a); if you have to mine from deeper ranks instead, it is (b).
</details>
