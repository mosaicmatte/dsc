# Phase 2 — Dense retrieval
**28/08 – 03/09 · 7 days**

> Goal: a fine-tuned bi-encoder that beats BM25, and a hybrid that beats both.
> Four ablation rows minimum: zero-shot → fine-tuned → + hard negatives → + hybrid.

---

## PART A — Learn

### A1. Bi-encoder vs cross-encoder — and why the standard architecture is what it is

**Bi-encoder.** Query and document are encoded *independently* into vectors;
relevance is their cosine similarity.

```
q ──[encoder]──► vq ┐
                    ├── cos(vq, vd)
d ──[encoder]──► vd ┘
```

Because `vd` does not depend on `q`, the whole corpus is embedded **once**, offline, and
a query is answered by one vector search. Cost per query: one encoder forward pass plus a
similarity scan.

**Cross-encoder.** Query and document are concatenated and encoded *jointly*; the model
attends across the pair at every layer.

```
[CLS] q [SEP] d [SEP] ──[encoder]──► relevance score
```

Nothing can be precomputed — `d`'s representation depends on `q`. Cost per query: one
forward pass **per candidate document**.

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Corpus precomputable | yes | **no** |
| Forward passes per query | 1 | N (one per candidate) |
| Query–document interaction | one dot product at the end | full attention, every layer |
| Accuracy | good | **markedly better** |
| Usable over a full corpus | yes | **impossible** |

That table *is* the retrieve-then-rerank architecture. Use the cheap, precomputable model
to cut the corpus down to ~50–100 candidates, then spend the expensive model only on
those. Phase 3 does the second half; run the arithmetic yourself in the self-check.

### A2. Contrastive training and InfoNCE

Fine-tuning teaches the encoder *this corpus's* notion of similarity. The loss:

```
L = − log  exp(sim(q, d⁺)/τ)  /  Σ_j exp(sim(q, d_j)/τ)
```

A softmax over one positive and many negatives — literally a classification problem
where the classes are "which of these documents answers the query?"

- **In-batch negatives.** The other documents in the same batch serve as negatives for
  free. So a batch of 64 gives 63 negatives per query at no extra cost. **This is why
  batch size matters so much here**: a bigger batch is a harder classification problem
  and forces sharper representations. Maximise it — gradient checkpointing and fp16 exist
  for exactly this reason.
- **Temperature τ.** Scales the logits. Low τ (0.02–0.05) sharpens the distribution and
  punishes near-misses hard; high τ softens it. `MultipleNegativesRankingLoss` in
  sentence-transformers uses a `scale` parameter = 1/τ, defaulting to 20.

### A3. Hard negatives — the single biggest lever in this phase

A **random** negative from a legal corpus is a document about traffic fines when the
query is about annual leave. The model separates those after roughly one epoch, and every
subsequent step teaches it nothing. Gradients go to zero and training plateaus.

A **hard** negative is a document that looks right and is not: same statute, adjacent
điều, same terminology, wrong condition. Learning to reject *those* is what actually
raises precision.

Standard mining ladder, and the roadmap follows it exactly:
1. **BM25-mined** — take BM25's top-k, remove the gold documents, the rest are hard
   negatives. Cheap and effective.
2. **Self-mined** — take *your own round-2 retriever's* top-k, minus gold. Harder still,
   because they are precisely the mistakes your current model makes.

> **The false-negative trap.** A mined "negative" that is actually relevant but simply
> unlabelled will actively damage training — you are teaching the model that a correct
> answer is wrong. Two standard defences, both implemented in `mine_hard_negatives.py`:
> **skip the top-n ranks** (rank 1–2 of a good retriever are often unlabelled positives)
> and **cap by score margin** (reject candidates scoring within ε of the gold document).
> Legal corpora are full of near-duplicate provisions, so this is not a theoretical risk.

### A4. Hybrid fusion — why it wins here specifically

BM25 and a dense encoder fail on **disjoint** query types:

| Query type | BM25 | Dense |
|---|---|---|
| `Nghị định 100/2019/NĐ-CP` | exact hit, max IDF | blurs `100/2019` and `100/2020` |
| "nghỉ phép năm" vs corpus "nghỉ hằng năm" | miss — no shared term | hit — paraphrase |

Two methods, both in `src/fusion.py`:
- **RRF** — combines *ranks*: `Σ w_i /(K + rank_i)`. Scale-free, so unbounded BM25 scores
  and cosine similarities mix without calibration. `K = 60` damps the top rank so one
  over-confident system cannot dominate. Safe default.
- **Weighted score fusion** — normalise each system's scores per query, then weighted sum.
  Can beat RRF because it keeps score *margins* (the gap between rank 1 and 2 is
  information that ranks throw away) — and margins are exactly what the `ratio` cutoff
  rule consumes downstream.

Tune the weight on dev. Do not assume 0.5.

---

### Going deeper (optional)

[`docs/reference/03_dense_retrieval.md`](../../docs/reference/03_dense_retrieval.md) — why random negatives
stop teaching (the gradient argument), the false-negative trap, a diagnostic table for
fine-tunes that did not help, and why RRF interacts badly with the `ratio` cutoff.
[`docs/reference/07_hardware_runtime.md`](../../docs/reference/07_hardware_runtime.md) — throughputs, memory,
and the embedding-cache trap that makes a fine-tune look like a no-op.

---

## PART B — Do

### Task B1 — Zero-shot evaluation of 2–3 embedders
```bash
python phases/2_dense/zero_shot_eval.py --model AITeamVN/Vietnamese_Embedding
python phases/2_dense/zero_shot_eval.py --model bkai-foundation-models/vietnamese-bi-encoder
python phases/2_dense/zero_shot_eval.py --model BAAI/bge-m3     # multilingual contrast
```
Record **every** score. Zero-shot numbers are ablation rows, not throwaways.
Segmentation is handled automatically by `src/dense.REGISTRY` — do not do it by hand.
**Done when:** 3 rows in the log, plus your read on why the ranking came out that way.

### Task B2 — Fine-tune round 1: in-batch negatives only
```bash
python phases/2_dense/train_biencoder.py --model <best> --round 1 --batch-size 64
```
`MultipleNegativesRankingLoss` on (query, positive) pairs. Batch size as large as the GPU
allows — this is the parameter that matters most in round 1.
**Done when:** round-1 model beats its own zero-shot number on dev.

### Task B3 — Fine-tune round 2: BM25-mined hard negatives
```bash
python phases/2_dense/mine_hard_negatives.py --run experiments/runs/<bm25-best>.jsonl \
    --out data/processed/train_pairs_bm25neg.jsonl --skip-top 2
python phases/2_dense/train_biencoder.py --round 2 --pairs data/processed/train_pairs_bm25neg.jsonl
```
**Done when:** you have inspected 10 mined negatives by hand and confirmed none is
actually a valid answer. Do this. It takes ten minutes and catches false negatives.

### Task B4 — Fine-tune round 3: self-mined hard negatives
```bash
python phases/2_dense/zero_shot_eval.py --model models/biencoder-r2 --run-id dense-r2
python phases/2_dense/mine_hard_negatives.py --run experiments/runs/dense-r2.jsonl \
    --out data/processed/train_pairs_selfneg.jsonl --skip-top 3
python phases/2_dense/train_biencoder.py --round 3 --pairs data/processed/train_pairs_selfneg.jsonl
```
**Done when:** logged. If round 3 does not beat round 2, say so in the log — a negative
result is a paper row, and "we mined harder and it stopped helping" is a real finding.

### Task B5 — Hybrid fusion
```bash
python phases/2_dense/hybrid.py --dense experiments/runs/dense-r3.jsonl \
    --lexical experiments/runs/<bm25-best>.jsonl --sweep
```
Both RRF and weighted score normalisation. Tune the weight on dev.
**Done when:** the weight sweep table is in `work/analysis/` and the best hybrid is logged.

### Task B6 — Re-sweep the cutoff, and check the budget
```bash
python phases/1_bm25/cutoff_sweep.py --run experiments/runs/hybrid-best.jsonl --plot
python src/params.py
```
The optimal set size shifts every time the score distribution changes. Keep a running
parameter total as you go.

---

### Task B7 — Write your own code  ← `TODO(YOU/phase2)`

[`phases/2_dense/mine_hard_negatives.py`](mine_hard_negatives.py) `select_negatives()` —
choose which candidates become training negatives. Which negatives you train on matters
more than almost any hyperparameter here, and the default (take the top `n_neg`) is the
obvious thing rather than the best thing.

```bash
python tools/todo.py --yours
```

---

## PART C — Self-check

1. Using **your actual numbers** from `analysis/dataset_stats.md`: how many
   (query, document) pairs would a cross-encoder have to score to rank the entire corpus
   for every dev query? At ~500 pairs/second on one GPU, how long is that?
2. Why does increasing batch size improve a bi-encoder trained with in-batch negatives,
   when batch size normally only affects gradient noise?
3. You mine hard negatives from your own retriever and dev score *drops*. Give two
   distinct explanations and how you would tell them apart.

Key in [`self_check.md`](self_check.md).

---

## Definition of done for Phase 2

- [ ] 3 zero-shot rows logged
- [ ] Rounds 1/2/3 trained and logged (including any negative result)
- [ ] 10 mined negatives manually inspected for false negatives
- [ ] Hybrid weight sweep table in `work/analysis/`
- [ ] Cutoff re-swept on the hybrid run
- [ ] Running parameter total recorded — still under 4B with room for Task 2
- [ ] 20 dev failures categorised in `analysis/error_analysis_phase2.md`
