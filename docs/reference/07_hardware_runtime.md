# 07 — Hardware, runtime, and planning a day

Order-of-magnitude figures for planning. Measure your own; these exist so that "this has been
running for three hours" triggers a check rather than a shrug.

---

## 1. What needs a GPU

| Stage | GPU | Notes |
|---|---|---|
| Phase 0 (ingest, split, stats) | no | seconds |
| Phase 1 (BM25, grid, cutoff) | no | CPU only; the grid is minutes |
| Phase 2 embedding a corpus | strongly preferred | CPU is ~20–50× slower |
| Phase 2 fine-tuning | **yes** | practically required |
| Phase 3 reranking | **yes** | it is the compute-heaviest inference step |
| Phase 3 cross-encoder fine-tuning | **yes** | |
| Phase 4 generation | **yes** for a 1.5B+ model | CPU generation is possible but painfully slow |

**Phases 0–1 are fully doable on a laptop.** That is deliberate: it means the harness,
the dev split, the cutoff logic and the first submission are never blocked on GPU access,
and whoever has the GPU is never blocked on the harness.

## 2. Rough throughputs (single mid-range GPU, fp16)

| Operation | Rate |
|---|---|
| bi-encoder embedding, 135M PhoBERT, 256 tok | ~1,500–3,000 docs/s |
| bi-encoder embedding, 568M BGE-M3, 512 tok | ~200–500 docs/s |
| cross-encoder scoring, 568M, 512 tok | ~300–800 pairs/s |
| cross-encoder scoring, 135M, 256 tok | ~1,500–3,000 pairs/s |
| generation, 1.5B, 128 new tokens, batch 8 | ~2–6 questions/s |

`rerank.py` prints its own measured `pairs/s` — compare it against this table. An order of
magnitude below it usually means you are on CPU, or `max_length` is much larger than needed.

## 3. Planning arithmetic that matters

**Embedding the corpus** = `N × granularity_multiplier / rate`. At `N=10,000` documents,
6 chunks each, 568M model at 300 docs/s: `60,000/300 = 200 s`. Fine — and it is cached, so
you pay it once per (model, corpus) pair.

**Reranking** = `queries × depth / rate`. 600 dev queries × depth 50 at 500 pairs/s =
`30,000/500 = 60 s`. Also fine. Now depth 100 over 3,000 test queries: `300,000/500 = 10 min`
— still fine, but note it is per *configuration*, so a five-way comparison is an hour.

**The one that bites: a full cross-encoder scan.** 600 × 60,000 = 36M pairs at 500/s = **20
hours**. This is the number to have in your head; it is why retrieve-then-rerank exists, and
it is Phase 2's self-check question.

## 4. Memory, and what to do when you run out

Fine-tuning memory ≈ model + gradients + optimiser states + activations. AdamW keeps two
extra copies of every trainable parameter, so full fine-tuning of a 568M model in fp32 needs
roughly 568M × (4 + 4 + 8) = ~9 GB before activations.

In order of what to try:

1. **`--fp16`** — roughly halves activation memory. Try first, costs nothing.
2. **`--grad-checkpointing`** — recomputes activations in the backward pass. Big memory saving,
   ~30% slower.
3. **Shorter `max_length`** — attention memory is quadratic in sequence length. Check your
   chunk-length p90 from `dataset_stats.md`; if it is 120 words, `max_length=512` is wasted.
4. **Gradient accumulation** — keeps the *optimisation* batch large while the *forward* batch
   shrinks. **But note this does NOT preserve in-batch negatives**: MNRL's negatives come from
   the forward batch. Accumulation is not a substitute for a large batch here, and this trips
   people up.
5. **Lower batch size** — last resort for a bi-encoder, because it directly weakens the
   training signal (see `docs/reference/03_dense_retrieval.md` §2).

## 5. Caches, and the trap in them

| Cache | Key | Invalidate when |
|---|---|---|
| `data/processed/.bm25_*.pkl` | corpus + segmenter + stopwords | corpus or tokenisation changes |
| `data/processed/.emb_*.npy` | model name + corpus path | **the model's weights change** |
| `src/normalize` segmentation cache | in-process only | never (per run) |

**The trap:** you fine-tune, save to the same output directory, evaluate, and get exactly the
zero-shot number back. The embedding cache key is the *path*, not the weights. Pass
`--no-cache` after every training run. `train_biencoder.py` prints the reminder in its final
output for this reason.

Symptom to recognise: dev score identical to four decimal places. Models do not do that.

## 6. If you have no GPU at all

You can still do real work:

- All of Phases 0 and 1, including a valid submission.
- Phase 2 **zero-shot** evaluation of a small model (135M PhoBERT) on CPU — slow but feasible
  on a modest corpus, and it gets you real ablation rows.
- All error analysis, all cutoff sweeps, all fusion experiments (they operate on run files,
  not models).
- All of the writing.

Then borrow a GPU for the fine-tuning runs specifically, with configs already prepared and
tested on CPU with `--limit`.

## 7. Reproducibility caveat

`src/config.set_seed` seeds Python, NumPy and torch. GPU kernels can still introduce
nondeterminism in the last bits of a float. Usually harmless at four decimal places — but
**verify it** during the freeze rather than assuming: run the pipeline twice and diff the
submission hashes (`freeze_checklist.md` §3). If they differ, find out where before the
Private Test, not during it.

---

## Check yourself

1. Your reranker reports 40 pairs/s. Table says 300–800. Name two likely causes.
2. Why is gradient accumulation not a substitute for a large batch when training a bi-encoder
   with `MultipleNegativesRankingLoss`?
3. Fine-tuning finishes, dev score is identical to zero-shot at four decimals. What happened?

<details><summary>answers</summary>

1. (a) It is running on **CPU** — check device selection and that CUDA is visible. (b)
   `max_length` is far larger than your passages need, and attention cost is quadratic in
   sequence length — check chunk-length p90 in `analysis/dataset_stats.md`. A third
   possibility is a batch size of 1 leaving the GPU idle between passes.
2. Because MNRL's negatives are **the other examples in the forward batch**. Accumulation
   makes the *optimiser step* see more examples but each forward pass still contains only
   `micro_batch` documents, so the softmax stays that many ways wide. It fixes gradient noise;
   it does not make the discrimination task harder, which is the thing that actually matters here.
3. You evaluated **cached embeddings from the base model**. The cache key is
   `(model_name/path, corpus_path)`, not the weights, so overwriting a checkpoint in place
   reuses the old vectors. Re-run with `--no-cache`. Identical-to-four-decimals is the
   signature — real model changes never produce that.
</details>
