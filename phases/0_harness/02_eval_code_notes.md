# Task B2 — Reading BTC's evaluation source (ANSWER IN WRITING)

> **TODO(TEAM/phase0-B2): answer every question below with a line reference into BTC's code.**

Put their published scoring script in `phases/0_harness/btc_eval/` (do not modify it —
`evaluate.py` imports it). Read it **line by line**, both tasks. Every answer below needs
a line reference into their file, not a recollection.

---

## Q1. Is the number of returned documents capped?

> ANSWER (with line ref):

If not — what stops a trivial "return the whole corpus" submission?
Possibilities to check for explicitly in their code:
- a hard truncation (`preds[:k]`)
- Precision entering the ranking so a huge set self-penalises
- a rule in the competition regulations rather than the code (note *where*)

> ANSWER:

**If nothing stops it**, that is a finding, not a licence to exploit it — write down what
the maximum-recall submission would score and treat it as an upper reference point in the
paper, then optimise honestly.

## Q2. Is Recall micro- or macro-averaged?

Look for the shape: `sum(hits)/sum(rel)` (micro) vs `mean(hits_i/rel_i)` (macro).

> ANSWER (with line ref):

**Then set** `OFFICIAL_AVERAGING` in [`../src/metrics.py`](../../src/metrics.py) to match.

## Q3. How exactly are ties resolved by Precision?

> ANSWER (with line ref):

- Is Precision computed on the same set as Recall?
- Rounding — do they round the primary score before comparing? (If they round to 4 dp, a
  1e-5 dev improvement is not a real improvement.)

> ANSWER:

## Q4. Edge cases — what happens when...

| Case | Their code does | Our `src/metrics.py` does |
|---|---|---|
| a query is missing from the submission | | counts as empty prediction (P=0, R=0) |
| a prediction contains a duplicate doc_id | | de-duplicated, order kept |
| a prediction contains an unknown doc_id | | counted in `\|predicted\|`, never a hit |
| predicted set is empty | | P=0, R=0 |
| a query has zero relevant docs | | R=0 by convention |

Any row where the two columns disagree is a bug in **our** harness. Fix `src/metrics.py`,
not their code.

## Q5. Task 2 metric

> ANSWER (with line ref): normalisation applied to answers? exact match, token-F1, or
> something else? Case/diacritic sensitivity?

---

## Verification (Task B4 gate)

```bash
python phases/0_harness/evaluate.py --pred <file> --gold <file> --cross-check
```
Must report agreement to 1e-9. Until it does, no modelling work.
