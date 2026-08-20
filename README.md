# DSC@UIT 2026 — Legal IR (Task 1) + Legal QA (Task 2)

> **New to this repo? → [`START_HERE.md`](START_HERE.md)**

Working repository for the DSC@UIT 2026 competition. The material is organised as six
sequential phases; each phase folder is both a lesson and a task list.

```bash
make setup      # venv + dependencies
make check      # verify the install end-to-end (no GPU needed)
make help       # every other command
```

---

## Hard boundaries

| Constraint | Value |
|---|---|
| Public Test | **06/08 → end of 18/09/2026** (already open) |
| Private Test | **19–23/09** |
| Parameter ceiling | **< 4B total per task**, every component, embedding layer included |
| LoRA / quantization | do **not** make a >4B model legal. Distillation does, if the distilled model is <4B |
| External APIs | **forbidden**, including free ones |
| Data | **BTC data only**, no augmentation, no external data |
| **Task 1 answer set** | **at most 5 document ids per question — 6 scores ZERO on both metrics** |
| Task 1 metric | **Recall** primary, **Precision** tiebreak, macro-averaged |
| Task 2 metric | **METEOR** primary, **ROUGE-L** secondary, macro-averaged |
| Submissions | 10/day, best-of-day on the leaderboard, **through the registered Organization** |
| Codabench | [Task 1](https://www.codabench.org/competitions/17715/) · [Task 2](https://www.codabench.org/competitions/17716/) |

Full transcription with sources: [`docs/reference/09_official_rules.md`](docs/reference/09_official_rules.md).
**That page overrides every other page in this repo.**

**Do today:** register the full candidate model list —
[`phases/0_harness/00_model_registration.md`](phases/0_harness/00_model_registration.md).
Approval is not instant, and an unregistered model invalidates the submission that used it.

---

## Layout

```
START_HERE.md        read this first
Makefile             make help

phases/              THE WORK — six folders, numbered in the order you do them
  0_harness/         20–23/08   ground truth & instrumentation
  1_bm25/            24–27/08   lexical baseline + first submission
  2_dense/           28/08–03/09 dense retrieval + hybrid fusion
  3_rerank/          04–08/09   cross-encoder reranking + ceiling analysis
  4_task2_qa/        09–14/09   LegalQA
  5_freeze/          15–18/09   freeze, package, submit

docs/                python_for_cpp, onboarding, walkthrough, glossary, troubleshooting
  reference/         optional depth: derivations, failure modes, reading list
src/                 shared library — read it, rarely edit it
tools/               todo list, fixture generator, error analysis
tests/               exhaustive cases; `make test`
data/                put BTC's files in data/raw/ — never committed
work/                everything generated: runs, configs, analysis, submissions
paper/               assembled from work/, not written in October
```

## The shared library

You should rarely need to modify `src/`. Read it, though — it is where the reasoning lives.

| Module | Owns | Read it when |
|---|---|---|
| [`normalize.py`](src/normalize.py) | NFC, tone placement, segmentation, BM25 tokenising | any Vietnamese text question |
| [`chunking.py`](src/chunking.py) | granularity, chunk↔parent mapping | ablation #1 |
| [`metrics.py`](src/metrics.py) | set-based and rank-based metrics | reading any score |
| [`cutoff.py`](src/cutoff.py) | the four answer-set rules | always — this is a model component |
| [`bm25.py`](src/bm25.py) | inverted index; k1/b applied at scoring time | Phase 1 |
| [`dense.py`](src/dense.py) | **model registry** — who needs segmented input | before every model swap |
| [`fusion.py`](src/fusion.py) | RRF and weighted score fusion | Phase 2 |
| [`params.py`](src/params.py) | the 4B budget | before committing to any model |
| [`analysis.py`](src/analysis.py) | **loss decomposition** — where every lost point of recall went | end of every phase |
| [`exp_log.py`](src/exp_log.py) | the run log and the dev↔leaderboard gate | every run |
| [`config.py`](src/config.py) | load, hash, freeze configs | Phase 5 |

## Canonical data formats

Only `phases/0_harness/ingest.py` knows BTC's raw layout. Everything downstream speaks these:

```jsonc
data/processed/corpus_<granularity>.jsonl   {"doc_id","text","meta":{"parent_id",...}}
data/processed/queries_<split>.jsonl        {"qid","text","relevant":[doc_id,...]}
work/experiments/runs/<run_id>.jsonl        {"qid","ranked":[[doc_id,score],...]}
work/experiments/predictions/<run_id>.jsonl {"qid","predicted":[doc_id,...]}
```

A **run** is a full ranking — reusable for fusion, reranking and ceiling analysis.
A **prediction** is the variable-length answer set after a cutoff rule — what gets scored
and submitted. One retrieval pass can be re-cut a hundred ways, so never bake the cutoff
into retrieval.

---

## If you fall behind

The schedule has no slack, so decide **in advance** what gets cut.

**Never cut**, in descending priority: the Phase 0 harness and dev split · one valid
submission through the Organization · the experiment log · the reproduction package ·
20 categorised failures per phase.

**Cut in this order if you must:**

1. Task 2 LoRA fine-tuning (Phase 4 B7) — already optional; ablations pay more per hour
2. Cross-encoder fine-tuning (Phase 3 B2) — keep the zero-shot row, say you ran out of time
3. Round-3 self-mined negatives (Phase 2 B4) — round 2 captures most of the gain
4. The second chunking granularity — pick one on Phase 1 evidence and commit

**Never borrow from Phase 5.** Compressing the freeze is how a working system fails to be
submittable. Behind on 14/09? Cut modelling scope, not the freeze.

### Checkpoints — if you are not here by this date, cut scope

| Date | You should have |
|---|---|
| 23/08 | `evaluate.py` verified against BTC's scorer; dev split built |
| 27/08 | ≥1 valid leaderboard score; `correlation()` reports `healthy` |
| 03/09 | a fine-tuned bi-encoder beating BM25 on dev |
| 08/09 | full Task 1 pipeline + ceiling table |
| 14/09 | ≥1 valid Task 2 submission |
| 16/09 | hyperparameters locked, no new ideas |
