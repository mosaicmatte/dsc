# Task B1 — Task 2 evaluation code notes (ANSWER IN WRITING)

> **TODO(TEAM/phase4-B1): answer every question below before writing any reader code.**
>
> The metric is already confirmed: **METEOR primary, ROUGE-L secondary**, macro-averaged,
> over plain `.split()` tokens (BTC's scorer has the Vietnamese tokenizer commented out).
> Their code is vendored at
> [`../0_harness/btc_eval/scoring_legalqa.py`](../0_harness/btc_eval/scoring_legalqa.py)
> and summarised in
> [`docs/reference/09_official_rules.md`](../../docs/reference/09_official_rules.md) §5.
> Use this worksheet for the parts only the real data can answer.

Same discipline as Phase 0. Line references, not recollections. **Do this before writing
any reader code** — the answer format determines whether Baseline A is even possible.

---

## Q1. What is the answer field, exactly?

```jsonc
// paste one real train record here, verbatim
```

> Field name:
> Type (span / free text / multiple choice / list of ids):
> Line ref in the data overview:

## Q2. Are gold answers verbatim substrings of the retrieved passages?

Check mechanically, do not eyeball it:
```bash
python -c "
import json,sys
d=json.load(open('data/raw/<task2_train>.json'))
hit=sum(1 for r in d if r['<answer_field>'] in r.get('<context_field>',''))
print(f'{hit}/{len(d)} answers appear verbatim in their context')"
```

> ANSWER: ___/___ verbatim
>
> *Already measured on the real data (20/08/2026), so use this to check your own
> answer rather than to skip the exercise:*
> `python phases/4_task2_qa/baseline_extractive.py --oracle-ceiling` returns
> **0/1050** on the dev split. Gold answers cite the statute and restructure it;
> none is a verbatim span. Median gold length is 309 words.
>
> **If most are verbatim → extractive (Baseline A) is viable and cheap.**
> **If few are → extraction cannot reach the gold answers; go generative.**

## Q3. What is the metric?

> ANSWER (with line ref):
> - Exact match? token-F1? ROUGE? something custom?
> - Is the answer normalised before comparison (lowercase, punctuation, diacritics)?
> - If token-F1: what tokeniser? Syllable-level or word-segmented? This changes the
>   score materially in Vietnamese.

## Q4. Output format

```jsonc
// paste the required submission format, verbatim
```
- Filename: ______   Zipped: ______
- Must every question appear? ______
- Is an empty answer allowed, and what does it score? ______
- Is a supporting-passage id also required? ______

## Q5. Is retrieval scored separately, or only the final answer?

> ANSWER:
> If only the answer is scored, retrieval quality is still your ceiling — measure it
> anyway (Task B2), you just will not be graded on it directly.

---

## Decision, based on the above

> Approach: **extractive** / **generative** / **hybrid**
> Because:
