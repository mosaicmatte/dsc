# What is left to do, and who can do it

Generated view:

```bash
python tools/todo.py              # everything, grouped by severity
python tools/todo.py --blockers   # only the hard gates
python tools/todo.py --phase 0    # only phase 0
python tools/todo.py --count      # one line; exits non-zero if blockers remain
```

---

## The taxonomy

| Marker | Meaning | Consequence if left open |
|---|---|---|
| `TODO(BLOCKER/<phase>-<task>)` | Depends on BTC's data, evaluation code, or model cards. **Nobody could have done it in advance.** | The affected numbers are *unverified*. Do not put them in the paper or trust them on the leaderboard. |
| `TODO(TEAM/<phase>-<task>)` | Analysis and writing the team produces. | Nothing crashes. The paper cannot be assembled. |
| `TODO(OPTIONAL/...)` | Worth doing if time allows. | Nothing. |

Anything left as a bare `TODO` is reported as `UNCATEGORISED` so it cannot hide.

---

## The blockers, in the order they bite

These are the things I could not do for you, because each one requires a file or a
web page that only you have access to. Everything else in the repo is written and tested.

### 1. `src/metrics.py` — micro vs macro averaging
**Phase 0, task B2.** Read BTC's scorer; look for `sum(hits)/sum(rel)` (micro) versus
`mean(hits_i/rel_i)` (macro). Set `OFFICIAL_AVERAGING` to match.
**Until then:** every dev number in the repo is reported under an assumption. They may
still be *useful* for ranking your own runs — but a run that wins under macro can lose
under micro, so the ranking itself is not safe.

### 2. `phases/0_harness/evaluate.py` — `btc_official_score()`
**Phase 0, task B4.** Wire in their published scorer so `--cross-check` works.
**Until then:** our reimplementation is unvalidated. It is careful and tested against
hand-computed examples, but "careful" is not "verified against theirs".

### 3. `phases/1_bm25/make_submission.py` — `format_submission()` and `SUBMISSION_FILENAME`
**Phase 0, task B1 → used in Phase 1, task B5.** The placeholder is the shape most
Vietnamese legal-IR shared tasks use. It is a guess.
**Until then:** submissions will likely be rejected or score zero. This costs a
submission from a budget of ten per day, so fix it before your first upload.

### 4. `src/params.py` — verify parameter counts
**Phase 0, task B0.** Check each number against the model card, and record the HF
revision SHA you will actually use.
```bash
python -c "from src.params import count_hf; print(count_hf('BAAI/bge-m3'))"
```
**Until then:** the 4B budget arithmetic is approximate. It is right to roughly ±5%,
which is fine at 2.7B total and *not* fine if you end up near 3.9B.

### 5. `src/dense.py` — verify the model registry
**Phase 0, task B0.** Three fields per model: `segmented`, `max_seq`, `params`. Plus
check whether the card specifies a required query prefix/instruction — several retrieval
models expect one and quietly underperform without it.
**`segmented` is the dangerous one.** A wrong value costs 5–15 points and raises no
exception. Read the card's usage example: underscored words (`người_lao_động`) or a call
to `ViTokenizer.tokenize` means `segmented=True`.

### 6. `phases/4_task2_qa/ablate_context.py` — `score_answers()`
**Phase 4, task B1.** Swap in BTC's Task 2 metric. The built-in token-F1 and exact match
are labelled `UNOFFICIAL` in every output they produce, deliberately.
**Until then:** the Task 2 ablation tables rank variants under a metric that is not the
one you are scored on.

### 7. `phases/5_freeze/build_package.py` — three blanks in the generated README
**Phase 5, task B2.** Model revisions, the exact reproduction commands, and the expected
sha256. The script prints a reminder of all three when it runs.
**Until then:** the reproduction package does not reproduce anything.

---

## Team work (nothing crashes; the paper depends on it)

| Where | When | What |
|---|---|---|
| `phases/0_harness/01_schema_summary.md` | 20–23/08 | fields, granularity, submission format |
| `phases/0_harness/02_eval_code_notes.md` | 20–23/08 | four questions, each with a line reference |
| `phases/4_task2_qa/00_task2_eval_notes.md` | 09/09 | answer format, metric, extractive-viability check |
| `phases/4_task2_qa/prompts.py` | 13–14/09 | a prompt variant matching the real answer format |
| `analysis/error_analysis_phase*.md` | end of every phase | 20 categorised failures |
| `analysis/bm25_grid.md`, `hybrid_weight_sweep.md`, `ceiling_table.md` | as generated | the "Interpretation" sections |
| `phases/5_freeze/freeze_checklist.md` | 15–18/09 | every box |
| `paper/outline.md` | end of every phase | that phase's rows and paragraph |

> The generated `.md` files under `work/analysis/` all end with an **Interpretation** or
> **fill in** section. Those sections are the deliverable — the tables above them are
> just the evidence. A script can produce the table; only you can say what it means.

---

## Definition of "no blockers left"

```bash
python tools/todo.py --count
```
Exits non-zero while any blocker remains. Worth running before every submission and
before you quote any number in the paper.
