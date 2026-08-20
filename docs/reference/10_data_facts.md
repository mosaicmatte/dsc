# 10 — What the real data actually looks like

Every number on this page was **measured** on the Task 1 Public Test drop
(`train.json`, `public-official.json`, `selected-contexts.zip`) on 20/08/2026,
not estimated. Reproduce any of it with the snippets at the bottom.

Rules live in [`09_official_rules.md`](09_official_rules.md). This page is facts
about the data, and the consequences those facts have for what you build.

---

## 1. The shape of it

| | |
|---|---|
| corpus documents | **8,532** (`context_*.json`, ids 1–~290k, not contiguous) |
| train questions | **7,000** |
| public-test questions | **999** (`"answer": null`) |
| qid overlap train ∩ public | **0** |
| distinct documents used as gold | **3,105** of 8,532 |
| documents never gold in train | **5,427** (64%) — real distractors, not padding |
| total corpus size | 455 MB, **72.8M whitespace tokens** |

## 2. Gold-set sizes — this is the single most important table on the page

| golds per question | count | share |
|---|---|---|
| **1** | 6,447 | **92.1%** |
| 2 | 485 | 6.9% |
| 3 | 53 | 0.8% |
| 4 | 14 | 0.2% |
| 5 | 1 | 0.01% |

Mean 1.09. Never more than 5 — the same number as BTC's cap, which is surely why
the cap is 5.

### What follows from it

**For 92% of questions, recall is binary: 1.0 if the gold document is in your
five, 0.0 otherwise.** Macro-recall is therefore almost exactly *"how often is
the right document in my top 5"*.

**So always return exactly 5 document ids.** Recall is the primary metric and it
is monotonically non-decreasing in the number of documents you return, up to the
cap. Returning 3 instead of 5 can only lose recall; it can never gain any.
Precision only breaks *exact* ties on recall, which across 999 questions
essentially never happens between two different systems.

This has a blunt consequence for this repo: the adaptive-cutoff machinery in
[`src/cutoff.py`](../../src/cutoff.py) — `ratio`, `gap`, `mine` — is **a
precision instrument in a recall contest**. Keep `--cutoff fixed --k 5` for
every submission. Cutoff work is worth doing as a paper ablation (it makes a
genuine precision/recall trade-off graph), not as a way to climb the leaderboard.
`loss_cutoff` in [`src/analysis.py`](../../src/analysis.py) will read 0.0 for a
fixed-5 run, and that is correct, not a bug.

## 3. Documents are enormous

| percentile | whitespace tokens |
|---|---|
| min | 0 |
| p25 | 2,475 |
| **median** | **4,814** |
| p75 | 9,378 |
| p95 | 27,524 |
| max | **1,242,409** |

| threshold | docs above it |
|---|---|
| 256 tokens | 8,504 (99.7%) |
| **512 tokens** | **8,401 (98.5%)** |
| 1,024 tokens | 8,041 (94.2%) |
| 4,096 tokens | 4,802 (56.3%) |
| 8,192 tokens | 2,510 (29.4%) |

Questions, by contrast, are tiny: median **19** tokens, max 50.

### What follows from it

- **Chunking is not optional for dense retrieval.** A 512-token encoder sees
  ~10% of a median document and ~0.04% of the largest. Embedding a truncated
  document is embedding its letterhead: *"BỘ Y TẾ … CỘNG HÒA XÃ HỘI CHỦ NGHĨA
  VIỆT NAM … Độc lập - Tự do - Hạnh phúc"*, which every document shares. Expect
  near-random dense retrieval if you skip chunking — and it will fail *silently*,
  looking merely mediocre rather than broken.
- **Chunk, retrieve, then aggregate back to the parent document id**
  (`--aggregate max`). Gold labels are whole-document ids; chunk ids score zero.
- **BM25 over whole documents is a legitimate strong baseline** — it has no
  length limit, and its `b` parameter exists precisely to handle a 2,475 ↔
  27,524 token spread. Sweep `b` hardest.
- Budget: ~73M tokens. At 256-token chunks with stride 128 that is roughly
  570k chunks to embed. That is the real cost driver of Phase 2, so measure the
  encode rate on 100 documents before launching the full run.

## 4. Landmines

### 4.1 Integer ids vs string ids — silent total zero

`context_*.json` carries `"id": 740` (**int**). `train.json` carries
`"answer": ["177504"]` (**str**). BTC's scorer intersects raw sets, and
`{740} & {"740"} == set()`. An all-integer submission scores **0.0 recall, 0.0
precision, no error, no warning**.

`ingest.py` casts with `str()`; `metrics.check_submittable()` now rejects
non-strings. Do not defeat either.

### 4.2 Empty passages — 9 train questions are unwinnable

BTC confirmed on 20/08 that train contains empty and duplicated passages while
public/private test answers do not.

| | count |
|---|---|
| documents with empty `passage` | 20 |
| of those, gold for ≥1 train question | 6 |
| **train questions whose every gold is empty** | **9** |
| exact-duplicate passage groups | 4 (9 documents, 4 questions affected) |

So your **local train recall ceiling is 0.9987**, not 1.0. `ingest.py --validate`
prints this and writes `data/processed/quarantine_queries_train.jsonl`. Exclude
those 9 when mining positive pairs — a positive pair pointing at an empty
document teaches the encoder noise — and remember the ceiling when a local number
refuses to reach 1.0.

Duplicates are rare enough (4 groups) to ignore, but if a gold document has an
identical twin, both ids are equally "correct" by content and you cannot tell
them apart. Returning both costs 2 of your 5 slots to guarantee 1 hit.

### 4.3 ROUGE-L on Vietnamese is mostly noise (Task 2)

BTC's vendored `rouge_score` tokenizer is the stock Google one:

```python
NON_ALPHANUM_PATTERN = r"[^a-z0-9]+"     # applied AFTER .lower()
```

Vietnamese letters are not in `a-z`. Every diacritic — and `đ` — is replaced by a
space. Measured with BTC's own code:

```
REFERENCE : Theo Điều 37 Nghị định 153/2020/NĐ-CP, doanh nghiệp phát hành trái phiếu phải công bố thông tin.
ROUGE SEES: ['theo','i','u','37','ngh','nh','153','2020','n','cp','doanh','nghi','p',
             'ph','t','h','nh','tr','i','phi','u','ph','i','c','ng','b','th','ng','tin']
```

| hypothesis | ROUGE-L | METEOR |
|---|---|---|
| identical to reference | 1.0000 | 0.9999 |
| same sentence, diacritics stripped | 0.2857 | 0.1176 |
| **unrelated** Vietnamese sentence | **0.2692** | 0.0000 |
| random latin letters, no meaning | 0.1923 | 0.0000 |

An unrelated sentence scores 0.269 ROUGE-L. The metric has a noise floor around
0.2–0.27 for *any* Vietnamese text of similar length, so it barely discriminates.

**METEOR is the metric that means anything**, and it is the primary one anyway —
it tokenizes with plain `.split()`, so it sees real Vietnamese words. Two
consequences: optimise for METEOR and treat ROUGE-L as a rounding error; and
**never strip diacritics from a Task 2 answer** — it costs 88% of METEOR while
ROUGE-L barely notices.

## 5. What this data does to a baseline

Measured on the **1,049-question dev split** produced by
`python phases/0_harness/build_dev_split.py` with its defaults (seed 42,
dev_frac 0.15, stratified by gold count, unanswerable questions dropped). BM25
with stock `k1=1.2, b=0.75`, no word segmentation, top-5:

| corpus granularity | Recall@5 | Precision | recall@10 | recall@50 | recall@100 | MRR@10 |
|---|---|---|---|---|---|---|
| whole document | 0.3680 | 0.0784 | 0.4471 | 0.6548 | 0.7271 | 0.2594 |
| **article chunks + `--aggregate max`** | **0.7410** | 0.1575 | 0.8240 | 0.9252 | 0.9395 | 0.5796 |

**Chunking alone doubles recall: +0.3730.** No model was trained, no parameter
tuned. This is the single largest intervention measured so far, and it is a
*data-processing* change — precisely the kind BTC said in the 08/08 email counts
as a contribution.

The mechanism is §3: a median document is 4,814 tokens, so BM25's length
normalisation drowns a 20-token question's signal in a document that is 250×
longer. Cutting the document into its `Điều` gives the query something its own
size to match. `avgdl` drops from 8,438 to 419.

### Where the remaining 0.2666 goes

`tools/error_analysis.py` on the article run:

```
0.1985  ranking     RERANKER / FUSION — retrieved but ranked below position 5
0.0605  retrieval   RETRIEVER — the document is not in the run at all
0.0000  cutoff      (we always return exactly 5, so this is structurally zero)
```

**77% of what is missing is a ranking problem, not a retrieval problem.** The
gold document is already in BM25's top-100 for 94.0% of questions; it is just
not in the top 5. That is what a cross-encoder reranker is for, and it bounds
the payoff: a *perfect* reranker over this candidate set would reach 0.9395, and
a perfect retriever improvement at fixed depth could add at most 0.0605. Phase 3
is where the points are.

The whole-document run tells the same story from the other end: its recall@100
is only 0.7271, so *no* reranker could have taken it past 0.73. Chunking did not
just add recall, it raised the ceiling every later stage inherits.

## 6. Reproduce these numbers

```bash
python phases/0_harness/ingest.py \
    --raw-corpus  data/raw/selected-contexts.zip \
    --raw-queries data/raw/train.json \
    --raw-test    data/raw/public-official.json
python phases/0_harness/ingest.py --validate
python phases/0_harness/build_dev_split.py
python tools/data_facts.py
```

`tools/data_facts.py` regenerates every table above from
`data/processed/`, so when BTC ships Private Test you re-run it rather than
trusting this page.

---

## Check yourself

1. Your teammate proposes an adaptive cutoff that returns 2 documents when the
   retriever is confident and 5 when it is not. Using §2, say what that does to
   Recall and to Precision, and whether it can ever improve your leaderboard
   position.
2. A dense run scores recall@5 = 0.11 — barely above chance. Before you blame
   the model, which single number from §3 would you check, and what would you
   expect to see in the top-5 documents if that were the cause?
3. Your local dev recall is 0.94 and refuses to go higher no matter what you
   try. Two facts on this page bound it. Name them and give the arithmetic.
4. Your submission scores 0.0000 recall on Codabench but 0.71 locally, and the
   scorer printed no error. What is the first thing you check, and why does it
   produce exactly zero rather than a degraded score?
5. For Task 2, a teammate suggests stripping diacritics from generated answers
   "to make matching easier". Using the table in §4.3, quantify how bad that
   idea is on each metric, and explain why the two metrics disagree so violently.
