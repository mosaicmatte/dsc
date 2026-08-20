# Phase 5 — Freeze, package, submit
**15–18/09 · 4 days · then Private Test 19–23/09**

> No new ideas after the 16th. This phase is about making sure the thing that works
> can be *proved* to work, by someone else, from a clean checkout.

---

## PART A — Learn

### A1. Why "no new ideas after the 16th" is a rule, not caution

An untested change made on the 17th cannot be validated before the Public Test closes. If it
is worse, you discover that during the Private Test — where you cannot fix it. You are
trading a known score for an unknown one with **no information gain**.

That is why it is a date rather than a judgement call: judgement is exactly what erodes at
11pm on the 17th.

### A2. Why you freeze the leaderboard's pick, not dev's

You have iterated against dev for four weeks. Every grid search, every cutoff sweep, every
"we kept the config that scored best" was a **selection step on a finite sample**, and
selection buys optimism that does not transfer. The leaderboard is a sample you have touched
far less — ten submissions a day is not enough to overfit meaningfully — so it is the better
estimate of the Private Test.

If they disagree by more than ~1 point, freeze the leaderboard's choice and write the
disagreement into the paper: it says something real about your dev split.

### A3. What "reproducible" actually requires

Four things, and people usually remember two:

1. **Pinned versions** — `pip freeze`, not the loose `requirements.txt`.
2. **Fixed seeds** — and *verified*, by running twice and diffing the output hash.
3. **Exact model revisions** — an HF commit SHA. `main` is not a revision; model cards get
   updated mid-competition.
4. **Commands someone else can run** — tested from a clean clone by a person who did not
   write them.

GPU kernels can be non-deterministic in the low bits of a float, which can reorder two
near-tied documents. Usually harmless — but **verify it during the freeze**, not during the
Private Test. See `docs/reference/07_hardware_runtime.md` §7.

### A4. What you may change during the Private Test

Only what makes the **frozen pipeline run** on the new data: a field name, an encoding, an
unexpected id format. These do not change the model, so the Public Test score remains a valid
estimate of what you submitted.

Not permitted, however tempting once you see the private data: re-tuning the cutoff, swapping
a checkpoint, changing depth, adjusting the fusion weight. Log every compatibility fix in
`freeze_checklist.md` §7 — documented is fine, undocumented looks like tampering.

---

## PART B — Do

### The schedule, and why it is this tight

| Date | Work | Rationale |
|---|---|---|
| **15–16/09** | Lock hyperparameters. Final Public Test submissions. | Two days of buffer before the deadline for the failure you have not thought of yet. |
| **17/09** | Build the reproduction package. | It always takes longer than expected, and doing it before the deadline means you can still fix what it reveals. |
| **18/09** | Final Public Test submissions through the registered Organization. Verify each is recorded valid. | Public Test closes today. |
| **19–23/09** | Private Test: run the **frozen** pipeline only. | No modelling changes. None. |

**Why "no new ideas after the 16th" is a real rule, not caution.** An untested change made
on the 17th cannot be validated before the Public Test closes, so if it is worse you will
not find out until the Private Test — where you cannot fix it. The expected value of a
late change is negative.

---

> Going deeper (optional):
> [`docs/reference/07_hardware_runtime.md`](../../docs/reference/07_hardware_runtime.md) §7 on the
> reproducibility caveat — GPU nondeterminism, how to tell it from a real bug, and why you
> verify it during the freeze rather than during the Private Test.

### Task B1 — Lock hyperparameters (15/09)

```bash
python phases/5_freeze/freeze_pipeline.py --run-id <best run_id> --out configs/FINAL.yaml
```

Pick the winner from `work/experiments/runs.csv` by leaderboard score, **not** dev score — dev
was for iterating, the leaderboard is the estimate of the Private Test. If dev and
leaderboard disagree about which run is best, trust the leaderboard and note the
disagreement in the paper.

Fill in [`freeze_checklist.md`](freeze_checklist.md) as you go.

### Task B2 — Build the reproduction package (17/09)

```bash
python phases/5_freeze/build_package.py --config configs/FINAL.yaml --out dist/
```

BTC accepts GitHub or a zip; Docker is optional. Required contents:

- [ ] repository with all code
- [ ] `README` with **exact step-by-step** reproduction commands
- [ ] pinned `requirements.txt` (`pip freeze`, not the loose one)
- [ ] fixed random seeds, verified by running twice
- [ ] model weights, or documented download steps with exact HF revisions
- [ ] the frozen config for the submitted run
- [ ] `work/experiments/runs.csv` — the full experiment log

**The test that matters:** clone into a fresh directory, follow your own README literally,
and confirm you reproduce the submitted file byte for byte. Have a teammate who did not
write it do this. Everything the README omits, they will find.

### Task B3 — Final submissions (18/09)

- Submit through the **registered Organization**. A personal-account submission does not
  count and cannot be retroactively transferred.
- Verify each submission is recorded as **valid**, not merely uploaded.
- Confirm every model used appears on the approved registration list.
- Record the final leaderboard scores in `work/experiments/runs.csv`.

### Task B4 — Private Test (19–23/09)

Run the frozen pipeline. Nothing else.

```bash
python phases/5_freeze/run_frozen.py --config configs/FINAL.yaml \
    --queries data/processed/queries_private_test.jsonl
```

If it crashes on the private data, the only permitted fixes are ones that make the
*frozen pipeline run* — a schema difference, a missing id, an encoding issue. Not a model
change, not a re-tuned cutoff. Write down anything you had to touch; it belongs in the
paper's limitations section.

---

## PART C — Self-check

1. Dev says run A is best; the leaderboard says run B. Which do you freeze, and why?
2. Your freeze passes every check. On 20/09 the Private Test file has a field name that does
   not exist in `ingest.py`. What are you allowed to do?
3. Two runs from the same seed produce submissions with different sha256. Is the pipeline
   broken?

Key in [`self_check.md`](self_check.md).

---

## Definition of done for Phase 5

- [ ] `configs/FINAL.yaml` frozen and git-tagged
- [ ] Reproduction package built and verified by a teammate from a clean clone
- [ ] Two identical runs from the same seed produce identical submissions
- [ ] Final Public Test submissions recorded as valid, through the Organization
- [ ] Every model used is on the approved list
- [ ] `work/experiments/runs.csv` complete, every submitted run has a leaderboard score
- [ ] Private Test executed with zero modelling changes
