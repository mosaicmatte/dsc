# Phase 5 — self-check answer key

Phase 5 has no new modelling ideas by design, but it has failure modes that cost more than
any modelling mistake — because they surface after the Public Test has closed.

---

### 1. Dev says run A is best; the leaderboard says run B. Which do you freeze, and why?

> **Freeze B.**
>
> You have been iterating against dev for four weeks. Some of its advantage over the
> leaderboard is real signal and some is **overfitting to that particular 10% sample** —
> every cutoff sweep, every grid search, every "we picked the config that scored best on
> dev" was a selection step, and selection on a finite sample buys optimism that does not
> transfer.
>
> The leaderboard is a held-out sample you have touched far less (ten submissions a day is
> not enough to overfit meaningfully), so it is the better estimate of the Private Test.
>
> **And write the disagreement into the paper.** It says something real about your dev
> split — usually that it is too small, or that its stratification missed a dimension that
> matters. That is a limitations-section paragraph, not an embarrassment.
>
> The one exception: if the leaderboard gap is under ~1 point, it is inside the noise of
> both samples. Then prefer the simpler or cheaper system and say why.

### 2. Your freeze passes every check. On 20/09 the Private Test file has a field name that does not exist in `ingest.py`. What are you allowed to do?

> **Fix the ingest mapping. Nothing else.**
>
> The permitted category is: *changes that make the frozen pipeline run on the new data*.
> A field name, an encoding, an unexpected id format, a missing optional key. These do not
> change the model, so the Public Test score remains a valid estimate of what you are
> submitting.
>
> **Not permitted**, however tempting when you see the private data: re-tuning the cutoff,
> swapping a checkpoint, changing retrieval depth, adjusting the fusion weight. Each of
> those invalidates the estimate, and you have no way to validate the change — the Public
> Test is closed.
>
> Log the change in `freeze_checklist.md` §7 with the exact diff. It belongs in the paper's
> limitations section, and an organiser reviewing your reproduction package will see the
> commit anyway. Documented is fine; undocumented looks like tampering.

### 3. Two runs from the same seed produce submissions with different sha256. Is the pipeline broken?

> **Not necessarily — find out which kind of difference it is before deciding.**
>
> `src/config.set_seed` seeds Python, NumPy and torch, but GPU kernels can be
> non-deterministic in the low bits of a float (reduction order varies with scheduling).
> That can flip the order of two documents whose scores agree to six decimals, which changes
> the file without changing anything meaningful.
>
> Diagnose it:
> ```bash
> python -c "
> from src.io_utils import load_predictions
> a,b = load_predictions('sub1.jsonl'), load_predictions('sub2.jsonl')
> diff = {q for q in a if a[q] != b[q]}
> print(len(diff), 'queries differ')
> print({q: (a[q], b[q]) for q in list(diff)[:3]})"
> ```
> - A handful of queries, differing only in the order of near-tied documents → benign GPU
>   nondeterminism. **Document it in `freeze_checklist.md` §3** and move on.
> - Many queries, or different documents entirely → something genuinely unseeded. Look for
>   an unseeded shuffle, a `set` iteration whose order leaks into output, a sampling step in
>   generation (`--temperature > 0`), or a dict ordering assumption.
>
> Either way the fix is the same discipline: **verify, do not assume.** The place to
> discover this is during the freeze, not during the Private Test.

---

### Bonus

**Why build the reproduction package on the 17th rather than after the deadline?**
> Because building it always reveals something — an undocumented manual step, a hardcoded
> path, a model you forgot you swapped. Doing it *before* the Public Test closes means you
> can still fix what it reveals and re-submit. Afterwards you can only write it up as a
> limitation.

**Why must a teammate who did not write the README be the one to test it?**
> Because the author cannot see their own assumptions. Every step you perform automatically
> — activating a venv, having a cached model, being in the right directory — is invisible to
> you and blocking for them. That is precisely the set of gaps the organiser will hit.

**What is the expected value of a new idea on the 17th?**
> Negative. It cannot be validated before the Public Test closes, so if it is worse you find
> out during the Private Test, where you cannot fix it. You are trading a known score for an
> unknown one with no information gain. That is why the rule is a date, not a judgement call
> — judgement is exactly what gets eroded at 11pm on the 17th.
