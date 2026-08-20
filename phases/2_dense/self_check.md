# Phase 2 — self-check answer key

### 1. How many pairs would a cross-encoder need to score to rank the whole corpus?

> **Use your own numbers** from `analysis/dataset_stats.md` — the point is to feel
> the size, not to recite a formula.
>
> `pairs = n_queries × n_corpus_units`.
>
> Worked example with plausible figures: 60 dev queries × 60,000 điều-level units
> = **3.6 million pairs**. At ~500 pairs/second on one GPU (a realistic rate for a
> 568M-parameter cross-encoder at 256 tokens), that is 7,200 seconds ≈ **2 hours
> for the dev split alone**. Scale to the full public test set and re-run it after
> every hyperparameter change and the approach is simply impossible.
>
> Now the same corpus through retrieve-then-rerank: 60 queries × 100 candidates =
> **6,000 pairs ≈ 12 seconds**. That is a 600× reduction, and it is why the
> architecture is retrieve-then-rerank rather than rerank-everything.
>
> The bi-encoder makes this work because `vd` does not depend on `q`, so all
> 60,000 document vectors are computed **once, offline**, and every subsequent
> query is a single forward pass plus a matrix multiply.

### 2. Why does batch size improve a bi-encoder trained with in-batch negatives?

> Normally batch size only trades gradient noise against step count. With
> `MultipleNegativesRankingLoss` it changes **the task itself**.
>
> The loss is a softmax over the batch: "which of these N documents answers this
> query?" Batch 16 is a 16-way classification problem; batch 256 is a 256-way one.
> The larger batch is genuinely harder, so the representations must be sharper to
> solve it — and sharper representations are exactly what retrieval over a large
> corpus needs, because at inference the model is effectively doing
> 60,000-way discrimination.
>
> This is why you should exhaust fp16 and gradient checkpointing before lowering
> batch size. It is also why hard negatives help so much at *small* batch: each
> mined negative adds a column to the softmax, buying difficulty that you could
> not afford to buy with memory.

### 3. You mine hard negatives from your own retriever and dev score drops. Two explanations?

> **Explanation A — false negatives.** Your retriever ranked unlabelled-but-correct
> documents highly, you mined them as negatives, and training taught the model
> that correct answers are wrong. Most likely when the corpus contains
> near-duplicate provisions, which legal corpora always do.
>
> **Explanation B — the negatives are too hard.** Round-3 negatives are drawn from
> a model that is already good, so they sit right at the decision boundary. The
> loss is dominated by examples that are nearly indistinguishable from positives;
> gradients get noisy and the model degrades rather than sharpens.
>
> **How to tell them apart:**
> - *Read them.* `mine_hard_negatives.py --inspect 20`. If mined "negatives"
>   plainly answer their query, it is A. This check costs ten minutes.
> - *Vary `--skip-top`.* If raising it from 2 to 10 recovers the score, it is A —
>   you were mining unlabelled positives off the top of the ranking.
> - *Vary `--depth`.* If mining from ranks 20–50 instead of 3–20 recovers the
>   score, it is B — you needed easier negatives, not fewer false ones.
> - *Check per-query deltas.* A implies a few queries collapse catastrophically
>   (their gold document is now trained against); B implies a broad, mild decline
>   across many queries.
>
> Either way: **log the negative result.** "We mined harder negatives and it
> stopped helping at round 3" is a legitimate paper row and directly answers
> BTC's question about where deep learning stops paying off.

---

### Bonus

**Zero-shot, a 135M PhoBERT model beats a 568M BGE-M3 model. Should you use the small one?**
> Not automatically. Check `recall@100`, not the official score — that is the
> ceiling the Phase 3 reranker inherits. Also check whether the BGE-M3 model was
> handicapped by input format (was it wrongly given segmented text?) and whether
> PhoBERT's 256-token limit is truncating your chunks. Both are silent failures.
> And note the budget: the 135M model leaves 2.9B of headroom for Task 2's
> generator, which may matter more than a point of recall.

**Why L2-normalise embeddings?**
> It makes the dot product exactly the cosine similarity, so one similarity
> function works everywhere — search, fusion, thresholding — and score ranges stay
> comparable across models, which the `ratio` cutoff rule depends on.
