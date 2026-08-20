# Phase 3 — Reranking
**04–08/09 · 5 days**

> Goal: the full Task 1 pipeline, plus the **retrieval ceiling table** — the analysis that
> tells you whether to spend your remaining time on the retriever or the reranker, and
> which doubles as the "why was this method insufficient" section BTC asked for.

---

## PART A — Learn

### A1. The retrieval ceiling — the most important idea in this phase

A reranker only reorders the candidates it is given. It cannot retrieve.

> If your retriever's Recall@50 is 0.85, then **no reranker operating on that top-50 can
> ever exceed 0.85**, no matter how accurate it is. The missing 15% of relevant documents
> are not in the list.

So there are exactly two ways to improve, and they are not interchangeable:

| Symptom | Diagnosis | Fix |
|---|---|---|
| Recall@100 is high, final score is low | relevant docs are retrieved but ranked badly | **reranker** |
| Recall@100 is itself low | relevant docs are never retrieved | **retriever** — reranking is wasted effort |

The ceiling table (Task B4) measures exactly this. Build it before deciding where to
spend days 3–5 of this phase.

**Reranking depth is a genuine hyperparameter with a real trade-off.** Deeper = higher
ceiling but more noise for the reranker to sift and linearly more compute. Depth 50 vs
100 vs 200 is a measurement, not a guess.

### A2. Cross-encoder training objectives

**Binary cross-entropy on pairs.** Each `(query, doc)` gets label 1 or 0; the model
outputs a relevance logit. Simple, stable, and — usefully for us — the logits are
**calibrated across queries**, which means the `threshold` cutoff rule becomes viable
alongside `ratio`.

**Listwise / localised contrastive.** Score one positive against n negatives for the same
query with a softmax. Optimises ordering directly, usually a bit better, but the scores
are only meaningful *within* a query, which breaks cross-query thresholding.

Either way, **the negatives must come from your own retriever's errors**, not random
sampling. At inference the reranker only ever sees the retriever's top-k, so training it
on random negatives trains it on a distribution it will never encounter. This is the same
argument as Phase 2's hard negatives, but even sharper here.

### A3. Why the cutoff must be re-swept after reranking

Cross-encoder score distributions are **much sharper** than retriever ones — the model
attends across the pair, so it separates relevant from irrelevant far more decisively.
Consequence: the `ratio` rule at the same α returns **smaller** sets, and the optimum
usually moves to a *lower* α (or a *higher* α with fewer documents kept — measure it, do
not reason it out). Post-rerank optimal sets are typically smaller than post-retrieval ones.

### A4. Budget

A BGE-M3-class embedder (~0.57B) + a bge-reranker-v2-m3-class reranker (~0.57B) ≈ **1.14B**,
leaving ~2.86B for Task 2's generator. That fits Qwen2.5-1.5B comfortably. It does **not**
fit Qwen2.5-3B (3.09B) — check with `python src/params.py` before committing.

---

### Going deeper (optional)

[`docs/reference/05_reranking.md`](../../docs/reference/05_reranking.md) — the ceiling stated precisely, the
two-number diagnosis, depth as a trade, BCE vs listwise, and why the un-reranked tail is kept.

---

## PART B — Do

### Task B1 — Zero-shot reranking of the hybrid top-50 and top-100
```bash
python phases/3_rerank/rerank.py --run experiments/runs/hybrid-best.jsonl \
    --model AITeamVN/Vietnamese_Reranker --depth 50
python phases/3_rerank/rerank.py --run experiments/runs/hybrid-best.jsonl \
    --model itdainb/PhoRanker --depth 50
python phases/3_rerank/rerank.py --run experiments/runs/hybrid-best.jsonl \
    --model namdp-ptit/ViRanker --depth 50
```
Then repeat the best one at `--depth 100`. Remember PhoRanker and ViRanker are
PhoBERT-backbone and **require segmented input** — `src/dense.REGISTRY` handles it, so do
not segment by hand.
**Done when:** 4 rows logged and you know which reranker and which depth won.

### Task B2 — Fine-tune the best reranker on your retriever's errors
```bash
python phases/3_rerank/train_cross_encoder.py --model <best> \
    --run experiments/runs/hybrid-best.jsonl --epochs 2
```
Negatives are mined from *your* retriever's top-k, not sampled randomly.
**Done when:** fine-tuned reranker logged; keep the zero-shot row for the ablation.

### Task B3 — Re-sweep the cutoff
```bash
python phases/1_bm25/cutoff_sweep.py --run experiments/runs/rerank-best.jsonl --plot \
    --out analysis/fig_cutoff_postrerank.png
```
**Done when:** you can state how the optimal set size changed and why.

### Task B4 — The ceiling table  ← the analysis BTC asked for
```bash
python phases/3_rerank/ceiling_table.py --retriever experiments/runs/hybrid-best.jsonl \
    --reranked experiments/runs/rerank-best.jsonl
```
Retriever Recall@10/@50/@100 against the final score after reranking.
**Done when:** `analysis/ceiling_table.md` exists and you have written one paragraph
answering: *retriever or reranker for the remaining days?*

---

## PART C — Self-check

1. Retriever Recall@50 = 0.91, Recall@100 = 0.93; after reranking the top-50 your final
   Recall is 0.78. Where is the loss, and what do you fix?
2. Same retriever, but final Recall after reranking is 0.90. Now what do you fix?
3. Why must reranker training negatives come from your own retriever rather than random
   sampling — and what specifically goes wrong if they do not?

Key in [`self_check.md`](self_check.md).

---

## Definition of done for Phase 3

- [ ] 3 zero-shot rerankers compared at depth 50, best one also at 100
- [ ] Fine-tuned reranker logged alongside its zero-shot row
- [ ] Cutoff re-swept post-rerank, figure saved
- [ ] `analysis/ceiling_table.md` written, with the retriever-vs-reranker paragraph
- [ ] `python src/params.py` confirms the Task 1 pipeline is under 4B
- [ ] 20 dev failures categorised in `analysis/error_analysis_phase3.md`
