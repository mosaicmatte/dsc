# Phase 1 — Lexical baseline and a live submission
**24–27/08 · 4 days**

> Goal: a submitted, valid Codabench score, and proof that dev predicts the leaderboard.
> A strong BM25 baseline is not a formality — in Vietnamese legal IR it routinely beats
> zero-shot dense retrieval, and it is half of the hybrid system you will ship.

---

## PART A — Learn

### A1. BM25, term by term

```
score(q, d) = Σ_t∈q  idf(t) · qtf(t) ·      tf(t,d) · (k1 + 1)
                                        ─────────────────────────────────
                                        tf(t,d) + k1·(1 − b + b·|d|/avgdl)

idf(t) = log( 1 + (N − df(t) + 0.5) / (df(t) + 0.5) )
```

**`idf` — how surprising is this term?** A term in 3 of 50,000 documents is enormously
informative; one in 40,000 is nearly worthless. The `log(1 + …)` form keeps IDF positive
even for terms in more than half the corpus, unlike the classic Robertson form which goes
negative and lets a common word *subtract* score.

**`k1` — term-frequency saturation.** Without it, a document saying "người lao động"
ten times would score ten times higher than one saying it once. It is not ten times more
relevant. `k1` sets how quickly extra occurrences stop helping: `k1 = 0` makes term
frequency purely binary; large `k1` approaches raw counts. Typical 1.2–2.0.

**`b` — document-length normalisation.** Long documents contain more terms *by accident*,
so they match more queries by accident. `b = 1` fully divides by relative length; `b = 0`
disables normalisation entirely. Typical 0.75.

### A2. Why legal retrieval rewards exact lexical match

Legal language is **not** paraphrased. A question about `Nghị định 100/2019/NĐ-CP` uses
that exact string, and so does the source. `Điều 113`, `khoản 2`, and defined terms
(`người lao động`, `hợp đồng lao động`) appear verbatim on both sides. These are precisely
the tokens with the highest IDF, and precisely the tokens a dense embedder is worst at —
embeddings blur `100/2019` and `100/2020` into near-identical vectors.

This is why the hybrid in Phase 2 wins: BM25 nails the identifiers, dense handles the
paraphrase. Neither does both.

Consequence for tokenisation: `src/normalize.tokenize` deliberately keeps `/`, `.` and
`-` inside tokens, so `100/2019/nđ-cp` stays a single high-IDF term instead of shattering
into `100`, `2019`, `nđ`, `cp`.

### A3. Vietnamese preprocessing

| Step | Why | Where |
|---|---|---|
| Unicode NFC | `ố` can be one codepoint or two; they are different strings to BM25 | `src/normalize.normalize` |
| Tone placement | `hoà` vs `hòa` — both are correct Vietnamese, and a corpus mixes them | `src/normalize.normalize_tone` |
| Lowercasing | `Điều` vs `ĐIỀU` in headings | `normalize(lower=True)` |
| Word segmentation | `học sinh` is one word, not two | `src/normalize.segment` |
| Stopwords | marginal — IDF already handles it; legal phrasing needs function words | off by default |

**The tone rule is subtler than a find-and-replace.** The two styles only differ in *open*
syllables: `hoà`→`hòa`, `khoẻ`→`khỏe`, `thuỷ`→`thủy`. But `toàn` and `hoàn` are spelled
identically in both styles — a naive replacement corrupts them into `tòan`/`hòan`. Our
implementation guards with a lookahead requiring the syllable to end there.

**Segmentation and model backbone — get this right or lose 5–15 points silently:**
- PhoBERT-backbone (`vietnamese-bi-encoder`, `PhoRanker`, `ViRanker`) → **requires**
  segmented input (`học_sinh`).
- BGE-M3 / XLM-R-backbone (`Vietnamese_Embedding`, `bge-m3`, `Vietnamese_Reranker`) →
  **must not** be given segmented input.

Neither mistake raises an exception. Both just quietly score worse.

For BM25 itself, segmentation is an **ablation, not a given**: it makes matches more
precise but also more brittle (a segmenter error breaks a match that a syllable-level
index would have caught). Measure it — Task B2 below.

---

### Going deeper (optional)

[`docs/reference/02_bm25.md`](../../docs/reference/02_bm25.md) — where the formula comes from, how to read a
flat grid, what tokenisation choices cost, and BM25's known failure modes in legal IR.
[`docs/reference/04_vietnamese_nlp.md`](../../docs/reference/04_vietnamese_nlp.md) — Unicode, the tone-mark
rule and why it needs a lookahead, segmentation, and the legal vocabulary that must survive
tokenisation.

---

## PART B — Do

### Task B1 — Chunk the corpus at two granularities  ← ablation #1
```bash
python phases/1_bm25/chunk_corpus.py --granularity document
python phases/1_bm25/chunk_corpus.py --granularity article
```
This usually matters more than the model choice. If the gold labels are at document level
but you index at article level, `--aggregate max` collapses chunk scores back up — the
script handles it, but you must know which case you are in (Phase 0, Task B1).
**Done when:** both corpora exist in `data/processed/` and their sizes are in the log.

### Task B2 — BM25 baseline, both granularities × segmented/unsegmented
```bash
python phases/1_bm25/bm25_baseline.py --corpus data/processed/corpus_article.jsonl \
    --segmenter none --run-id bm25-art-nosegorig
```
Four runs. Record all four — they are ablation rows, not throwaways.
**Done when:** four rows in `work/experiments/runs.csv`.

### Task B3 — Grid search `k1` × `b`
```bash
python phases/1_bm25/grid_search.py --corpus data/processed/corpus_article.jsonl
```
`k1 ∈ {0.9, 1.2, 1.5, 2.0}` × `b ∈ {0.3, 0.5, 0.75, 1.0}` = 16 configs, one index build.
**Done when:** the printed heatmap is in `work/analysis/` and the winner is in the log.

### Task B4 — Sweep the answer-set cutoff  ← the figure for the paper
```bash
python phases/1_bm25/cutoff_sweep.py --run experiments/runs/<best>.jsonl --plot
```
Score top-k for k = 1…20 **and** the score-ratio rule (keep docs above α × top score).
Plot Precision, Recall and the official score against the cutoff.
**Done when:** `analysis/fig_cutoff_sweep.png` exists and you can explain the crossover.

### Task B5 — Submit to Codabench, then check correlation  ← the gate
```bash
python phases/1_bm25/make_submission.py --run experiments/runs/<best>.jsonl \
    --cutoff ratio --alpha 0.85
# submit, then record what the leaderboard said:
python -c "from src.exp_log import update_leaderboard as u; u('<run_id>', 0.xxxx)"
python -c "from src.exp_log import correlation as c; print(c())"
```
**If dev and leaderboard diverge, stop all modelling work and fix the harness.**
A dev split that does not predict the leaderboard is worse than no dev split.
**Done when:** ≥3 submitted runs and `correlation()` reports `healthy`.

---

## PART C — Self-check

1. What does `b = 0` mean, mechanically?
2. Legal articles vary enormously in length. Argue for **or** against `b < 0.75` here.
3. Your ratio cutoff at α = 0.85 returns 1 document for some queries and 30 for others.
   Is that a bug?

Key in [`self_check.md`](self_check.md).

---

## Definition of done for Phase 1

- [ ] Both granularities built and indexed
- [ ] 4 baseline rows + 16 grid rows in `work/experiments/runs.csv`
- [ ] Cutoff sweep figure in `work/analysis/`
- [ ] ≥1 valid Codabench submission through the registered Organization
- [ ] `src.exp_log.correlation()` reports `healthy`
- [ ] 20 dev failures categorised in `analysis/error_analysis_phase1.md`
