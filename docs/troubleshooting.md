# Troubleshooting

Symptoms first. Each entry says how to confirm the diagnosis, not just what it might be.

---

## Harness and data

### Gold ids reported "not present in corpus"
```bash
python phases/0_harness/ingest.py --validate
```
**This is the most serious error in the repo.** Your `doc_id` construction disagrees with
BTC's labels: recall is capped below 1.0 and **no model can fix it**.

Confirm: print one gold id and one corpus id side by side. Look for a prefix, a zero-padding
difference, a composite key (`law_id` + `article_id`) that you built in a different order, or
labels pointing at articles while you indexed documents.

Fix in `ingest.py` — it is the only file that should know BTC's layout.

### Chunking produced no chunks (`chunks == input docs`)
Newlines were lost. `src.chunking`'s regexes anchor on `^`, so flattened text has no line
starts to match.

Confirm:
```bash
python -c "
from src import io_utils
r=next(io_utils.read_jsonl('data/processed/corpus_document.jsonl'))
print(repr(r['text'][:200]))"
```
If there are no `\n`, ingest normalised with `keep_newlines=False`. Corpus text must use
`normalize(..., keep_newlines=True)`.

### `could not detect the <x> field`
`ingest.py` prints the available keys. Either pass the field explicitly
(`--text-field noi_dung`) or add it to the corresponding `*_FIELDS` list in `ingest.py`.
Prefer the flag — it keeps the mapping visible in your shell history and reproducible.

### Ingest ran but the index is empty / all scores are zero
You mapped the wrong field as `text`. Check the `(id=…, text=…, title=…)` line ingest prints.
A wrong `text` field produces a plausible-looking empty index and a plausible-looking zero.

---

## Scores

### Dev score high, leaderboard low
```bash
python -c "from src.exp_log import correlation as c; print(c())"
```
In order of frequency:

1. **Chunk ids in the submission.** Forgot `--aggregate max`. Look at your submission file:
   if ids contain `#` or `::`, that is it. `make_submission.py` pre-flights this.
2. **Missing queries.** Scored as zero. `make_submission.py` warns; `evaluate.py` warns.
3. **Wrong averaging.** `src/metrics.OFFICIAL_AVERAGING` does not match BTC's.
4. **Dev leakage.** Dev queries also appear in your training data — check you trained on
   `queries_train_split.jsonl`, not `queries_train.jsonl`.
5. **Retrieved over the wrong split.** The classic: run built on dev, submission built on
   test. See `docs/walkthrough.md` Step 7.

### `RECALL (primary): 0.0000` on a test split
Expected — public/private test queries are unlabelled. The scripts now say so explicitly
instead of printing zeros. If you see actual zeros on a *labelled* split, that is a real
failure; check the gold-id diagnosis above first.

### Fine-tuned model scores identical to zero-shot, to four decimals
You evaluated **cached embeddings**. The cache key is the model path, not the weights.
```bash
python phases/2_dense/zero_shot_eval.py --model models/biencoder-r2 \
    --registry-as <base model> --no-cache
```
Models never produce identical numbers to four decimals. That signature is always a cache.

### Adding a reranker made the score worse
Check `avg_pred_size` before and after. Cross-encoder scores are much sharper, so the same
`ratio` α keeps far fewer documents.
```bash
python phases/1_bm25/cutoff_sweep.py --run experiments/runs/<reranked>.jsonl --plot
```
Only if the cutoff is already optimal is the reranker itself suspect.

### A model swap cost 5–15 points and nothing else changed
Input format. PhoBERT-backbone models require segmented input; BGE-M3-backbone models must
not have it. Check `src/dense.REGISTRY` and confirm the text went through `dense.prepare()`.
Also check the model's `max_seq` against your chunk-length p90 — PhoBERT's 256 tokens
(~180 Vietnamese words) truncates a lot of legal text.

### Hybrid fusion's best weight is 0.0 or 1.0
`hybrid.py --sweep` warns about this. One system dominates. Either that is a genuine result
(report it, do not ship a fake hybrid), or the weaker run is broken — check that both runs
cover the same qids and the same id namespace.

### Grid search is completely flat
Genuine possibility, not a bug. Units are length-uniform and terms do not repeat, so `k1`
and `b` are both inert. Record it and move to Phase 2. See `docs/reference/02_bm25.md` §3.

---

## Training

### Training loss hits ~0 in the first few hundred steps
Negatives are too easy. With `MultipleNegativesRankingLoss`, raise batch size (each example
adds a negative) or move to mined hard negatives.

### Dev degrades after adding hard negatives
Two causes, and one cheap test:
```bash
python phases/2_dense/mine_hard_negatives.py --run <run> --out <pairs> --inspect 20
```
If any mined "negative" plainly answers its query, they are **false negatives** — raise
`--skip-top`. If they all look genuinely irrelevant, they are simply **too hard** — mine from
deeper ranks (`--depth`).

### CUDA out of memory
In order: `--fp16`, `--grad-checkpointing`, shorter `--max-length` (check your p90 first),
then batch size **last** — lowering it directly weakens the training signal for a bi-encoder.
Note gradient accumulation does *not* preserve in-batch negatives; see
`docs/reference/07_hardware_runtime.md` §4.

### Reranking is 10× slower than the reference table
Usually CPU rather than GPU, or `--max-length` far larger than your passages need.
`rerank.py` prints measured pairs/s — compare against `docs/reference/07_hardware_runtime.md` §2.

---

## Task 2

### Fluent, confident, wrong answers
Split failures into "gold passage was in context" and "was not". `retrieval_stage.py` prints
`recall@1/@3/@5`. If `recall@3 = 0.58`, then 42% of questions are unanswerable by any reader
and prompt engineering has at most 42 points of nothing to offer. Fix retrieval first.

### Top-5 scores worse than top-3
Expected, not a bug. See `docs/reference/06_rag_task2.md` §4. Report the turning point.

### Task 2 metrics say `UNOFFICIAL`
Deliberate. `score_answers()` in `ablate_context.py` is a blocker
(`python tools/todo.py --blockers`). The built-in token-F1 and exact match are approximations
so the ablation loop runs before BTC's metric is wired in. Do not quote them without the label.

---

## Process

### "Which run produced this submission?"
```bash
python -c "
import pandas as pd; print(pd.read_csv('work/experiments/runs.csv').tail(20).to_string())"
```
If the run is not in the log, it did not happen — re-run it from its config. This is the
whole reason the log exists.

### "What is left to do?"
```bash
python tools/todo.py --blockers
```

### Something in `src/` looks wrong
Run the smoke test before assuming:
```bash
python phases/0_harness/smoke_test.py
```
It checks tone normalisation, chunking, BM25 parameter sensitivity, metric monotonicity,
variable-length cutoffs, fusion, and the correlation gate. If it passes, the bug is in your
data or your invocation, not the library.
