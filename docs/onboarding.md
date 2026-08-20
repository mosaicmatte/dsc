# Onboarding — start here

You have been handed a folder per phase. Each folder is **both a lesson and a task list**:
its `README.md` has a *Learn* section (the concepts, explained), a *Do* section (the
tasks, each naming its script and its definition of done), and a *Self-check* with an
answer key. The scripts carry `WHAT YOU NEED TO DO` / `HOW IT WORKS` comments at the top
and inline notes at the decision points.

## Day one, 30 minutes

```bash
git clone <repo> && cd dsc
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python phases/0_harness/smoke_test.py     # verifies the install end-to-end
```

`smoke_test.py` generates a small synthetic Vietnamese legal corpus, runs the whole
Phase 0 + 1 chain over it, and checks the numbers behave (recall monotone in set size,
precision falling, ratio cutoff producing variable-length sets). It writes only to a
temp directory. If it passes, your environment can run Phases 0–1.

**Coming from C++ and new to Python?** Read
[`python_for_cpp.md`](python_for_cpp.md) first — it covers only what this repo uses.

Read, in this order:
1. this file
2. [`docs/glossary.md`](glossary.md) — every term used in the phase READMEs
3. [`phases/0_harness/README.md`](../phases/0_harness/README.md) Part A

## Day one, the next hour: run it

[`docs/walkthrough.md`](walkthrough.md) takes you through the entire pipeline on synthetic data —
ingest, dev split, chunking, BM25 at two granularities, cutoff sweep, a submission, and error
analysis. It includes the real output of every command, and it makes one classic mistake on
purpose so you recognise it later.

**Do this before BTC's data arrives.** Then, when the real corpus lands, you are debugging
the data rather than the tooling.

## When you need more depth

| You want | Read |
|---|---|
| the concepts a phase needs | that phase's README, Part A |
| to check you understood | that phase's `self_check.md` |
| the derivation, failure modes, "why is it like that" | [`docs/reference/`](reference/) |
| a symptom diagnosed | [`docs/troubleshooting.md`](troubleshooting.md) |
| to know what is unfinished | [`docs/todo.md`](todo.md), or `python tools/todo.py` |

`docs/reference/` is optional depth — nothing in a phase requires it. Reach for it when a result
surprises you and you want to know whether it should have.

Then answer the three Phase 0 self-check questions **before** looking at the key. If you
cannot, re-read Part A. Everything downstream depends on reading your own dev numbers
correctly.

## How the repo fits together

```
data/raw/          BTC files, never committed
       ↓  phases/0_harness/ingest.py   ← the ONLY file that knows BTC's raw layout
data/processed/    canonical jsonl (see below)
       ↓  phase1/2/3 scripts
experiments/runs/  full rankings   {qid, ranked:[[doc_id, score], ...]}
       ↓  src/cutoff.py
experiments/predictions/  answer sets  {qid, predicted:[doc_id, ...]}
       ↓  make_submission.py
submissions/       zip for Codabench
```

Canonical formats — everything downstream of ingest speaks these:

```jsonc
// data/processed/corpus_<granularity>.jsonl
{"doc_id": "L1#dieu113", "text": "...", "meta": {"parent_id": "L1", "title": "..."}}
// data/processed/queries_<split>.jsonl
{"qid": "q42", "text": "...", "relevant": ["L1"]}
```

**Run vs prediction is a distinction worth internalising.** A *run* is a full ranking —
reusable for fusion, reranking, and ceiling analysis. A *prediction* is the
variable-length answer set after a cutoff rule — what gets scored and submitted. One
retrieval pass can be re-cut a hundred ways, so never bake the cutoff into retrieval.

## The shared library

You should rarely need to modify `src/`. Read it, though — it is where the reasoning lives.

| Module | What it owns | Read it when |
|---|---|---|
| `src/normalize.py` | NFC, tone placement, segmentation, BM25 tokenising | any Vietnamese text question |
| `src/chunking.py` | granularity, chunk↔parent mapping | ablation #1 |
| `src/metrics.py` | set-based and rank-based metrics | reading any score |
| `src/cutoff.py` | the four answer-set rules | always — this is a model component |
| `src/bm25.py` | inverted index, k1/b at scoring time | Phase 1 |
| `src/dense.py` | **model registry** — who needs segmented input | Phases 2–4, before every model swap |
| `src/fusion.py` | RRF and weighted score fusion | Phase 2 |
| `src/params.py` | the 4B budget | before committing to any model |
| `src/exp_log.py` | the run log and the dev↔leaderboard gate | every run |
| `src/config.py` | load, hash, freeze configs | Phase 5 |

## Five rules that will save you a week

1. **Never tune on the leaderboard.** Ten submissions a day is not a tuning budget; a
   single BM25 grid is 16 runs. Tune on dev, use the leaderboard to check that dev is
   honest.
2. **Segmentation must match the backbone.** PhoBERT models require segmented input;
   BGE-M3 models must not get it. Neither mistake raises an error — both just score
   worse. `src/dense.REGISTRY` decides; never do it by hand.
3. **Re-sweep the cutoff after every change.** The optimal answer-set size moves whenever
   the score distribution moves.
4. **Log every run, including the bad ones.** A negative result is a paper row. An
   unlogged run is a run you will repeat in September.
5. **`--aggregate max` whenever you retrieve at chunk level and the labels are at
   document level.** Forgetting it produces a submission full of ids BTC has never seen,
   which scores zero. `make_submission.py` pre-flights for this — do not ignore the warning.

## Who does what

Phases are sequential, but inside a phase the tasks parallelise:

| Phase | Splits cleanly into |
|---|---|
| 0 | (a) read BTC eval code + wrap it · (b) ingest + dev split + stats |
| 1 | (a) chunking + BM25 grid · (b) cutoff sweep + submission pipeline |
| 2 | (a) zero-shot comparison + hybrid · (b) fine-tuning rounds 1–3 |
| 3 | (a) zero-shot reranker comparison · (b) cross-encoder fine-tuning |
| 4 | (a) Task 2 retrieval + eval notes · (b) reader baselines + ablations |

Whoever is not on the critical path does the phase's **error analysis** — 20 categorised
failures. It is the most commonly skipped and most directly requested deliverable.

## When something looks wrong

Full symptom→diagnosis list in [`docs/troubleshooting.md`](troubleshooting.md). The short version:

| Symptom | Look here first |
|---|---|
| dev score high, leaderboard low | `src.exp_log.correlation()`, then the pre-flight warnings |
| chunking produced no chunks | newlines lost in ingest — `normalize(..., keep_newlines=True)` |
| gold ids "not in corpus" | `ingest.py --validate`; your doc_id construction disagrees with BTC's labels |
| fine-tuned model scores same as base | you evaluated cached embeddings — pass `--no-cache` |
| reranker helps nothing | check the ceiling: `phases/3_rerank/ceiling_table.py` |
| generator confidently wrong | check recall@k first; it is usually retrieval, not the reader |
