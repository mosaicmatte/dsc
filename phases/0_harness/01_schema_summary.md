# Task B1 — Schema summary (FILL THIS IN)

> **TODO(TEAM/phase0-B1): fill in every ANSWER field below from the BTC data overview.**
>
> Much of this is already answered in
> [`docs/reference/09_official_rules.md`](../../docs/reference/09_official_rules.md) §4
> (transcribed from BTC's own task document and scoring code). Use this worksheet to
> confirm it against the actual files you downloaded, and to record anything that
> differs.

Source documents: `data/raw/DSC2026_Task1_LegalIR_Data_Overview.docx` (+ Task 2 equivalent).
Read them with:
```bash
python -c "import docx;print('\n'.join(p.text for p in docx.Document('data/raw/<file>.docx').paragraphs))"
```

**Done when:** someone who has never seen the data could produce a valid submission
file from this page alone.

---

## 1. Files shipped

| File | Rows | Purpose |
|---|---|---|
| `corpus.json` / `law.json` | | the searchable collection |
| `train.json` | | labelled queries |
| `public_test.json` | | unlabelled queries to submit for |

## 2. Corpus record — exact fields

```jsonc
// paste one real record here, verbatim
```

| Field | Type | Meaning | Notes |
|---|---|---|---|
| | | | |

**Corpus granularity — the single most important answer on this page:**
Is one record a whole **văn bản**, a single **điều**, or a **khoản**?
> ANSWER:

**What do the gold labels point at?** (must match, or `src/chunking.aggregate_to_parent`
is mandatory before scoring)
> ANSWER:

## 3. Query record — exact fields

```jsonc
// paste one real record here, verbatim
```

Distribution of relevant docs per query: run `dataset_stats.py`, paste the summary.
> ANSWER:

## 4. Submission format — exact

```jsonc
// paste the required output format, verbatim, from the BTC docs
```

- Filename required: ______
- Zipped? ______
- Is the number of returned documents capped? ______  ← cross-check with `02_eval_code_notes.md`
- Are qids required to be complete/in order? ______
- What happens to a query with an empty prediction? ______

## 5. Task 2 differences

| Aspect | Task 1 | Task 2 |
|---|---|---|
| Query field | | |
| Answer field | | |
| Answer type (span / free text / choice) | | |
| Metric | Recall (primary), Precision (tiebreak) | |

## 6. Gotchas found while reading

- 
