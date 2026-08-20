# What is left to do, and who can do it

> **TODO(TEAM/ongoing): keep this honest.** When a blocker closes, delete it here and
> in the code, so `python tools/todo.py --blockers` stays a true list.

Generated view:

```bash
python tools/todo.py              # everything, grouped by severity
python tools/todo.py --blockers   # only the hard gates
python tools/todo.py --count      # one line; exits non-zero if blockers remain
```

---

## The taxonomy

| Marker | Meaning | Consequence if left open |
|---|---|---|
| `TODO(BLOCKER/<phase>-<task>)` | Depends on the real data or on a decision only you can make. | The affected numbers are *unverified*. |
| `TODO(TEAM/<phase>-<task>)` | Analysis and writing the team produces. | Nothing crashes. The paper cannot be assembled. |
| `TODO(OPTIONAL/...)` | Worth doing if time allows. | Nothing. |

---

## RESOLVED — what used to be blocked and no longer is

These were open because they depended on BTC material we did not have. That material
has now been read (mail thread + task documents + **their actual scoring programs**,
vendored in [`phases/0_harness/btc_eval/`](../phases/0_harness/btc_eval/)):

| Was blocked on | Now |
|---|---|
| micro vs macro averaging | **macro**, confirmed in their code; `src/metrics.py` matches |
| the official scorer | vendored; `evaluate.py --cross-check` verifies ours == theirs to 1e-12 |
| the Task 1 submission format | `submission.zip` → `submission.json`, `{qid: {"answer": [...]}}` — implemented |
| the Task 2 metric | METEOR primary + ROUGE-L secondary — their code is wired into `ablate_context.py` |
| whether the answer set is capped | **yes, at 5 ids; exceeding it scores ZERO** — enforced in `src/cutoff.py` |
| raw data shapes | `context_*.json` corpus, id-keyed question objects — `ingest.py` handles both |

The full transcription, with sources, is
[`docs/reference/09_official_rules.md`](reference/09_official_rules.md). **That page
overrides every other page in this repo.**

---

## Still genuinely blocked

### 1. Verify model parameter counts and licences — `src/params.py`, `src/dense.py`
**Phase 0, task B0.** The numbers in `KNOWN` and the `segmented` / `max_seq` flags in
`dense.REGISTRY` are typed from memory of the model cards. Verify each on the card,
record the HF revision SHA, and confirm the licence permits non-commercial research.

```bash
python -c "from src.params import count_hf; print(count_hf('BAAI/bge-m3'))"
```

`segmented` is the dangerous one: wrong value costs 5–15 points and raises no exception.

**Note BTC's rule precisely:** LoRA and quantization do **not** make a >4B model legal —
they change bits-per-parameter, not parameter count. Distillation *is* fine if the
distilled model is itself under 4B.

### 2. Confirm the vendored scorer still matches what Codabench runs
**Ongoing.** We vendored the programs BTC circulated on 05/08. If they re-issue them,
re-download and re-run `--cross-check`. Cheap insurance.

### 3. Fill the reproduction package blanks — `phases/5_freeze/build_package.py`
**Phase 5, task B2.** Model revisions, the exact commands, the expected sha256. The
script prints the reminder when it runs.

---

## Team work (nothing crashes; the paper depends on it)

| Where | When | What |
|---|---|---|
| `phases/0_harness/01_schema_summary.md` | now | confirm §4 of the rules page against the files you downloaded |
| `phases/0_harness/02_eval_code_notes.md` | now | write out the scoring rule yourselves — the whole team should know the 5-cap |
| `phases/4_task2_qa/00_task2_eval_notes.md` | before Task 2 work | the parts only the real data answers |
| `work/analysis/error_analysis_phase*.md` | end of every phase | 20 categorised failures |
| `work/analysis/*.md` "Interpretation" sections | as generated | the tables are evidence; the interpretation is the deliverable |
| `phases/5_freeze/freeze_checklist.md` | 15–18/09 | every box |
| `paper/outline.md` | end of every phase | that phase's rows and paragraph |

---

## Definition of "no blockers left"

```bash
python tools/todo.py --count
```
Exits non-zero while any blocker remains. Worth running before every submission.
