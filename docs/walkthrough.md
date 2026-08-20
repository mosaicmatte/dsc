# Walkthrough — run the whole pipeline today, on synthetic data

> **Do this before BTC's data arrives.** Forty minutes here means that when the real
> corpus lands you are debugging *the data*, not the tooling. Every command below works
> right now, on a fixture generated locally.

The fixture is 16 synthetic Vietnamese decrees (5 điều each) and 60 questions, in BTC's
**actual** shapes: a `selected-contexts.zip` of `context_*.json` records, and question
files that are JSON objects keyed by question id.

**It is not a difficulty proxy.** BM25 scores near-perfect recall on it and will not on
real data. Use it to learn the mechanics only.

---

## Step 0 — generate the fixture

```bash
python tools/make_fixture.py --out data/fixture
```

```
16 documents · 60 Task 1 queries · 60 Task 2 questions · 20 unlabelled test queries
```

Open one `context_*.json` inside `data/fixture/selected-contexts.zip` and read it before
continuing. Its fields — `id`, `name`, `link`, `passage` — are exactly what BTC ships.

## Step 1 — inspect before ingesting

```bash
python phases/0_harness/ingest.py --inspect \
    --raw-corpus data/fixture/selected-contexts.zip
```

`--inspect` prints the detected keys and one full record, and writes nothing. **Always run
this first on real data.** Guessing a field mapping and discovering it three scripts later
is how a day disappears.

## Step 2 — ingest into the canonical format

```bash
python phases/0_harness/ingest.py \
    --raw-corpus  data/fixture/selected-contexts.zip \
    --raw-queries data/fixture/train.json \
    --raw-test    data/fixture/public_official.json \
    --out-dir     data/fixture/processed
```

```
  read 16 records from 16 context file(s)
[corpus] 16 records -> .../corpus_document.jsonl  (id=id, text=passage, title=name)
[queries:train] 60 records -> .../queries_train.jsonl  (qid=__key__, query=question, rel=answer)
[queries:public_test] 20 records -> .../queries_public_test.jsonl  (rel=None)
```

Note it unzipped the corpus, and that `qid=__key__` — BTC's question files are objects
keyed by id, so the key *is* the question id.

**Check the detected mapping in that output every time.** It is the single highest-risk
line in the whole pipeline: a wrong `text` field produces an empty index and a plausible
zero.

```bash
python phases/0_harness/ingest.py --validate
```

The check that matters is *"gold ids not present in corpus"*. If that fires, your doc_id
construction disagrees with BTC's labels, recall is capped below 1.0, and **no model can
fix it**.

## Step 3 — the dev split

```bash
python phases/0_harness/build_dev_split.py \
    --queries data/fixture/processed/queries_train.jsonl \
    --out-dir data/fixture/processed
```

```
 |rel|     all   train    dev    all%  train%    dev%
     1      52      47      5   86.7%   87.0%   83.3%
     2       8       7      1   13.3%   13.0%   16.7%
 TOTAL      60      54      6
```

Read the last three columns. Train and dev carry the same mixture of single-answer and
multi-answer queries — that is what stratification bought you, and it is why the cutoff
you tune on dev will transfer.

## Step 4 — chunking, ablation #1

```bash
python phases/1_bm25/chunk_corpus.py \
    --corpus data/fixture/processed/corpus_document.jsonl --granularity article \
    --out    data/fixture/processed/corpus_article.jsonl
```

```
input docs  : 16
chunks      : 96  (6.0 per input doc)
word length : mean 54  p50 51  p90 95  p99 100  max 100
```

If `chunks` had come back equal to `input docs`, the article regex matched nothing —
almost always because newlines were lost during ingest.

## Step 5 — BM25 at both granularities

```bash
# whole document
python phases/1_bm25/bm25_baseline.py \
    --corpus  data/fixture/processed/corpus_document.jsonl \
    --queries data/fixture/processed/queries_dev.jsonl --run-id fx-doc

# điều level, scores collapsed back to the parent document
python phases/1_bm25/bm25_baseline.py \
    --corpus  data/fixture/processed/corpus_article.jsonl \
    --queries data/fixture/processed/queries_dev.jsonl --aggregate max --run-id fx-art
```

| | avg set size | Recall | Precision |
|---|---|---|---|
| article + `--aggregate max` | 1.50 | 0.9167 | **0.8750** |

**Read this properly, it is the point of the exercise.** When two granularities differ in
precision at the same α, that is usually the cutoff interacting with a different score
distribution — not a fact about granularity. On real data, re-sweep the cutoff for each
granularity before comparing them, or you are comparing cutoffs, not chunkings.

**Forget `--aggregate max` on chunked corpora and your submission contains ids like
`L01#dieu3` that BTC has never seen.** It scores zero. `make_submission.py` pre-flights for
this; do not ignore the warning.

## Step 6 — sweep the cutoff

```bash
python phases/1_bm25/cutoff_sweep.py --run experiments/runs/fx-doc.jsonl \
    --queries data/fixture/processed/queries_dev.jsonl
```

```
rule       param   recall     prec       f1   |set|
ratio       0.80   1.0000   0.5417   0.6444    3.00
ratio       0.75   1.0000   0.4306   0.5611    3.33
ratio       0.60   1.0000   0.4167   0.5444    3.50
top_k          4   1.0000   0.2917   0.4444    4.00
```

Recall never falls as the set grows; precision falls at every step. **The sweep stops at
5** because BTC zeroes any question returning more than 5 ids — and note the winner is a
`ratio` rule with an average set of 3.0, not a fixed k. That is the variable-length answer
set earning its keep: same recall as top-4, far better precision.

## Step 7 — build a submission, and make the classic mistake on purpose

```bash
python phases/1_bm25/make_submission.py --run experiments/runs/fx-doc.jsonl \
    --queries data/fixture/processed/queries_public_test.jsonl --out-dir /tmp/sub
```

```
PRE-FLIGHT ERROR: 18 questions missing from the submission (e.g. ['147011', ...]) —
BTC's scorer RAISES on this, the submission fails outright

Refusing to write an invalid submission. Fix the above, or pass --force if you know better.
```

**That is the error, and it is worth seeing once.** The run was retrieved over the *dev*
queries, so it has no rankings for the test questions. BTC's scorer raises on a key-count
mismatch, so this does not score badly — the submission *fails*. The script refuses to
write it. Retrieve over the split you are submitting:

```bash
python phases/1_bm25/bm25_baseline.py \
    --corpus  data/fixture/processed/corpus_document.jsonl \
    --queries data/fixture/processed/queries_public_test.jsonl --run-id fx-test --no-log

python phases/1_bm25/make_submission.py --run experiments/runs/fx-test.jsonl \
    --queries data/fixture/processed/queries_public_test.jsonl --out-dir /tmp/sub
```

```
answer sets  : mean 1.40, min 1, max 4        (no pre-flight errors)
```

Inspect what actually goes to Codabench:

```bash
unzip -p /tmp/sub/fx-sub.zip submission.json | head -c 200
```
```json
{"147011": {"answer": ["732"]}, "147027": {"answer": ["732", "730", "731", "733"]}}
```

A JSON object keyed by question id, at most 5 ids each. That is BTC's exact format —
`phases/0_harness/btc_eval/scoring_legalir.py` reads it directly.

Note that the test run reports *no scores* — the split is unlabelled. That is expected, not
a failure.

## Step 8 — error analysis

```bash
python tools/error_analysis.py --run experiments/runs/fx-doc.jsonl \
    --queries data/fixture/processed/queries_dev.jsonl --phase demo -n 5 \
    --out /tmp/ea_demo.md
```

Open the file. For each failing query it shows the gold document, **where it ranked in your
run**, and whether it made the answer set. That rank column is the diagnosis:

| rank of gold | meaning | fix |
|---|---|---|
| 1–5, not returned | cutoff problem | re-sweep α |
| 6–20 | ranking problem, and **only 5 slots to fix it** | reranker |
| 20–100 | ranking problem | reranker |
| not in top-100 | retrieval problem | reranker cannot help |
| not in corpus | harness bug | stop, fix ingest |

## Step 9 — Task 2, same shape

```bash
python phases/0_harness/ingest.py --raw-queries data/fixture/task2_train.json \
    --task 2 --out-dir data/fixture/processed
python phases/0_harness/build_dev_split.py \
    --queries data/fixture/processed/task2_train.jsonl \
    --prefix task2 --out-dir data/fixture/processed
```

`--task 2` picks up the gold **prose** answer (Task 2 answers are long structured text, not
document ids) and preserves any field it does not recognise.

Task 2 is scored by **METEOR** (primary) and ROUGE-L, both recall-weighted — so brevity is
punished. `phases/4_task2_qa/prompts.py` includes a `mimic` variant that reproduces the
reference answers' "Theo … quy định: - …" structure for exactly this reason.

## Step 10 — clean up

```bash
rm -rf data/fixture experiments/runs/fx-*.jsonl
python -c "
import csv,os
rows=[r for r in csv.DictReader(open('work/experiments/runs.csv')) if not r['run_id'].startswith('fx-')]
w=csv.DictWriter(open('work/experiments/runs.csv','w',newline=''),fieldnames=rows[0].keys() if rows else None)
" 2>/dev/null || true
```

Or just leave it; `data/fixture/` is gitignored.

---

## What you should now be able to do without looking

1. Name the four canonical file formats and which script produces each.
2. Explain why a run file and a prediction file are different things.
3. State the maximum number of document ids per question, and what happens at 6.
4. Say what `--aggregate max` does and why omitting it zeroes a submission.
5. Explain why recall never decreases as the answer set grows.
6. Diagnose a failure from the rank of the gold document.

If any of those is shaky, re-read the relevant phase README's **Learn** section — that is
what it is there for.
