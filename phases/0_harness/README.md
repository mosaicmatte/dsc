# Phase 0 — Ground truth and instrumentation
**20–23/08 · 4 days · produces no score**

> Skipping this phase is the most common way teams waste the following four weeks.
> Everything after this depends on being able to answer "did that change help?"
> in 30 seconds without spending a submission.

---

## PART A — Learn

### A1. Set-based vs rank-based metrics

Two different questions.

**Rank-based** metrics ask *"is the ranking good?"* and are computed at a fixed depth k:

| Metric | Definition | What it tells you |
|---|---|---|
| `Recall@k` | fraction of relevant docs found in top-k | your **ceiling** — no downstream stage can beat it |
| `Precision@k` | fraction of top-k that is relevant | density of the top of the list |
| `MRR` | mean of 1/(rank of first hit) | how fast a user finds *one* answer |
| `MAP` | mean of averaged precision at each hit | overall ranking quality, all hits |
| `nDCG@k` | discounted gain, log-position weighting | ranking quality with position discount |

**Set-based** metrics ask *"is the answer set good?"* Order inside the set is irrelevant:

```
Precision = |predicted ∩ relevant| / |predicted|
Recall    = |predicted ∩ relevant| / |relevant|
F_beta    = (1+β²)·P·R / (β²·P + R)          F1: balanced.  F2: recall-weighted.
```

### A2. The consequence of Recall being primary — read this twice

Task 1 reports **both** Precision and Recall. That is only informative if teams may
return **different numbers of documents per query**.

Here is the argument. Suppose every team returned exactly top-10. Then for a query with
`r` relevant documents, `Precision = hits/10` and `Recall = hits/r`, so
`Precision = Recall · r / 10` — Precision is a fixed rescaling of Recall, carrying no
independent information. Reporting both would be pure redundancy.

Because BTC reports both, the answer set must be **variable-length**, and therefore:

> **Deciding how many documents to return per query is a model component you tune on
> dev — not a formatting detail you hardcode at the end.**

This is what [`src/cutoff.py`](../../src/cutoff.py) is for, and why every later phase
re-sweeps the cutoff after every change. The score distribution moves whenever the
model does, so yesterday's optimal cutoff is not today's.

**And BTC caps the answer set at 5.** Their scorer gives a question **zero on both
metrics** if it returns more than 5 document ids (or none). So the "return everything"
exploit is closed, and the real problem is sharper and more interesting: you have exactly
**five slots per question**, and Precision breaks ties. Spend one slot where the retriever
is confident (precision 1.0 at no recall cost) and all five where it is not.

That is why `src/cutoff.py` clamps to 1..5 and refuses a larger `max_k`.
See [`docs/reference/09_official_rules.md`](../../docs/reference/09_official_rules.md) §4.

### A3. Micro vs macro averaging

- **Micro**: pool all queries, then compute one ratio. `Σhits / Σpredicted`. Queries with
  many relevant documents dominate.
- **Macro**: compute per query, then average. Every query weighs the same.

These can differ by several points and they reward different behaviour. Macro makes
easy single-answer queries matter as much as hard multi-answer ones. **You cannot tune
against a metric you have not identified.** Find out which BTC uses and set
`OFFICIAL_AVERAGING` in [`src/metrics.py`](../../src/metrics.py:37).

### A4. Why a dev split beats the leaderboard

Ten submissions per day is not a tuning budget — a single BM25 grid is 16 runs. The dev
split is your real feedback loop; the leaderboard is a *sanity check on the dev split*.
Stratify it, or a 10% sample will not carry the same distribution of "queries with 5
relevant documents" as the test set, and it will lie to you.

---

### Going deeper (optional)

[`docs/reference/01_metrics.md`](../../docs/reference/01_metrics.md) — the monotonicity proof written out,
micro/macro worked through with numbers, every harness edge case, and how much a 1-point dev
difference is actually worth.

---

## PART B — Do

Work in order. Each task names its script and what "done" means.

### Task B0a — Verify your install
```bash
python phases/0_harness/smoke_test.py
```
Runs the Phase 0 + 1 chain over synthetic data and checks the metric behaviour. Writes
only to a temp directory. Do this before anything else so an environment problem does not
get mistaken for a data problem later.

### Task B0 — Register the models (blocking, do it first)
→ [`00_model_registration.md`](00_model_registration.md)
Register the **full candidate list**, not just what you plan to use. Approval is not
instant; an unregistered model invalidates the submission that used it.

### Task B1 — Write the schema summary
→ fill in [`01_schema_summary.md`](01_schema_summary.md)
Read `DSC2026_Task1_LegalIR_Data_Overview.docx` and the Task 2 equivalent
(put them in `data/raw/`). One page: fields, corpus granularity, exact submission format.
**Done when:** someone who has never seen the data could write a valid submission file
from your page alone.

### Task B2 — Read BTC's evaluation source line by line
→ answer the questions in [`02_eval_code_notes.md`](02_eval_code_notes.md) **in writing**

Their scoring programs are already vendored verbatim in
[`btc_eval/`](btc_eval/) — read `scoring_legalir.py`; it is 40 lines and the two
lines that compute recall and precision decide your entire cutoff strategy.
**Done when:** all four questions are answered with a line reference into their code.

### Task B3 — Ingest raw data into canonical format
→ `python phases/0_harness/ingest.py --help`
This is the **only** file in the repo that knows BTC's raw layout. Keep it that way.
**Done when:** `data/processed/corpus_document.jsonl` and `queries_train.jsonl` exist
and `ingest.py --validate` passes.

### Task B4 — Verify our scorer against theirs
→ `python phases/0_harness/evaluate.py --run <run> --cross-check`
Already wired: it scores the same predictions through BTC's vendored code and ours.
**Done when:** `--cross-check` prints `AGREE` on your own data.

### Task B5 — Build the stratified dev split
→ `python phases/0_harness/build_dev_split.py`
90/10, stratified by number of relevant documents per query, fixed seed.
**Done when:** the printed stratum table shows train and dev have matching distributions.

### Task B6 — Dataset statistics table
→ `python phases/0_harness/dataset_stats.py`
**Done when:** `analysis/dataset_stats.md` exists. The paper needs this table verbatim,
so write it once, properly.

---

## PART C — Self-check

Answer without looking. Key in [`self_check.md`](self_check.md) — try first, then check.

1. State `Recall@k` from memory.
2. In two sentences: why could returning 100 documents per query raise Recall and lower
   Precision, and what does BTC's evaluation code do about it?
3. Your dev Recall is 0.72 micro and 0.61 macro. What does that gap tell you about which
   queries you are failing?

---

## Definition of done for Phase 0

- [ ] `smoke_test.py` passes
- [ ] Models registered on the BTC form
- [ ] `01_schema_summary.md` filled in
- [ ] `02_eval_code_notes.md` answered with line references
- [ ] you can state the 5-document cap and what happens if you exceed it
- [ ] `data/processed/` populated, `ingest.py --validate` passes
- [ ] `evaluate.py` reproduces BTC's score exactly
- [ ] dev split written to `data/processed/queries_dev.jsonl` with a fixed seed
- [ ] `analysis/dataset_stats.md` written
- [ ] `work/experiments/runs.csv` exists (even if empty)
