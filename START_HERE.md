# Start here

**DSC@UIT 2026 — Legal IR (Task 1) + Legal QA (Task 2).**

Public Test is **open now** (06/08 → 18/09). Under **4B parameters** per task, no APIs,
BTC data only. Task 1 returns **at most 5 document ids per question** — a 6th scores zero.

The rules are transcribed from BTC's emails, task documents and *scoring code* in
[`docs/reference/09_official_rules.md`](docs/reference/09_official_rules.md); that page
overrides everything else here.

---

## Your first hour

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python phases/0_harness/smoke_test.py      # verifies your install end-to-end
```

Then read, in this order:

0. **[`docs/python_for_cpp.md`](docs/python_for_cpp.md)** — **if you know C++ but not
   Python, start here** (25 min). It covers only what this repo uses, mapped from C++.
1. **[`docs/onboarding.md`](docs/onboarding.md)** — how the repo fits together (15 min)
2. **[`docs/walkthrough.md`](docs/walkthrough.md)** — run the entire pipeline on synthetic
   data, right now (40 min)
3. **[`phases/0_harness/README.md`](phases/0_harness/README.md)** — and begin

Keep **[`docs/glossary.md`](docs/glossary.md)** open in a tab.

> **You do not need to be fluent in Python to contribute.** Phases 0 and 1 are mostly
> running commands and reading numbers. When you are ready to write code, the places to
> do it are marked and self-contained:
> ```bash
> python tools/todo.py --yours
> ```
> Each block says what to write, gives ideas, and gives the exact command to test it.

---

## What is where

```
phases/       ← THE WORK. Six folders, numbered in the order you do them.
docs/         ← how to learn it, plus reference depth and troubleshooting
src/          ← shared library. Read it; you will rarely edit it.
tools/        ← utilities: todo list, fixture generator, error analysis
data/         ← put BTC's files in data/raw/. Never committed.
work/         ← everything the pipeline generates: runs, configs, analysis, submissions
paper/        ← assembled from work/, not written in October
```

Each **phase folder** is a lesson *and* a task list:

| Section of its README | What it is |
|---|---|
| **Part A — Learn** | the concepts, explained, with the reasoning |
| **Part B — Do** | numbered tasks, each naming its script and its definition of done |
| **Part C — Self-check** | questions, with a worked answer key in `self_check.md` |

Every script opens with `WHAT YOU NEED TO DO` and `HOW IT WORKS`.

---

## The six phases

| | Folder | Dates | You finish with |
|---|---|---|---|
| 0 | [`phases/0_harness`](phases/0_harness/) | 20–23/08 | a working scorer, a dev split, dataset stats |
| 1 | [`phases/1_bm25`](phases/1_bm25/) | 24–27/08 | your first valid leaderboard score |
| 2 | [`phases/2_dense`](phases/2_dense/) | 28/08–03/09 | a fine-tuned bi-encoder + hybrid fusion |
| 3 | [`phases/3_rerank`](phases/3_rerank/) | 04–08/09 | the full Task 1 pipeline + ceiling table |
| 4 | [`phases/4_task2_qa`](phases/4_task2_qa/) | 09–14/09 | a Task 2 baseline, submitted |
| 5 | [`phases/5_freeze`](phases/5_freeze/) | 15–18/09 | a reproduction package that actually reproduces |

**Do not skip Phase 0.** Skipping it is the most common way teams waste the next four weeks.

---

## Do today, before anything else

**Register the full candidate model list with BTC** —
[`phases/0_harness/00_model_registration.md`](phases/0_harness/00_model_registration.md).

Approval is not instant, and an unregistered model **invalidates the submission that used
it**. Register everything you might plausibly touch, not just what you plan to use; the form
can be resubmitted later to add more.

---

## Three commands worth memorising

```bash
python tools/todo.py --blockers     # what is unfinished, and what only you can do
python phases/0_harness/smoke_test.py    # is my environment sane?
python tools/error_analysis.py --run <run> --phase 1   # where did my recall go?
python -c "from src.exp_log import correlation as c; print(c())"   # is my dev split honest?
```

## The one rule most likely to cost you a submission

BTC's scorer gives a question **zero on both metrics** if it returns more than **5**
document ids, or none. Not truncation — zeroing.

```python
recall = mean([ |truth & pred| / |truth| if 0 < len(pred) <= 5 else 0 ])
```

`src/cutoff.py` clamps to 1..5 and raises if you try to raise `max_k`.
`make_submission.py` refuses to write a file that would fail. Do not defeat either.

---

## When you are stuck

| Situation | Go to |
|---|---|
| Python syntax you do not recognise | [`docs/python_for_cpp.md`](docs/python_for_cpp.md) |
| you want to write some code | `python tools/todo.py --yours` |
| something looks broken | [`docs/troubleshooting.md`](docs/troubleshooting.md) — symptoms → diagnosis |
| a term you do not know | [`docs/glossary.md`](docs/glossary.md) |
| a result surprised you | [`docs/reference/`](docs/reference/) — the depth behind each phase |
| "what is left to do?" | [`docs/todo.md`](docs/todo.md) |
| "which run produced this?" | `work/experiments/runs.csv` |

## Five rules that will save you a week

1. **Never tune on the leaderboard.** Ten submissions a day is not a tuning budget — a
   single BM25 grid is 16 runs. Tune on dev; use the leaderboard to check dev is honest.
2. **Segmentation must match the model's backbone.** PhoBERT requires segmented input;
   BGE-M3 must not have it. Neither mistake raises an error — both just score worse.
3. **Re-sweep the cutoff after every change.** The best answer-set size moves whenever the
   score distribution moves.
4. **Log every run, including the bad ones.** A negative result is a paper row. An unlogged
   run is a run you will repeat in September.
5. **`--aggregate max` when you retrieve at chunk level and the labels are documents.**
   BTC's gold labels are whole-document ids, so a chunk id like `740#dieu3` is a
   guaranteed miss. `make_submission.py` pre-flights for it.
