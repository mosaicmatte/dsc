# 09 — Official rules and specs, with sources

Everything on this page is transcribed from BTC's own emails, task documents and
**scoring code**, not inferred. Each item names its source so you can re-check it.
Where this page and any other page in the repo disagree, **this page wins.**

Verified 20/08/2026 against the DSC@UIT mail thread (through the 20/08 Q&A
email), the vendored scoring programs in
[`phases/0_harness/btc_eval/`](../../phases/0_harness/btc_eval/) — byte-identical
to BTC's published copies — and the real Task 1 Public Test data.

---

## 1. Timeline

| Round | Dates | Source |
|---|---|---|
| Warm-up | from 31/07 | kickoff emails, 31/07 |
| **Public Test (both tasks)** | **06/08 → end of 18/09/2026** | Public Test emails, 06/08 |
| Private Test | 19–23/09 | competition rules |

> The Public Test has been **open since 06/08**. The data below is already
> downloadable — there is nothing to wait for.

## 2. Links

| What | Where |
|---|---|
| Task 1 Codabench | https://www.codabench.org/competitions/17715/ |
| Task 2 Codabench | https://www.codabench.org/competitions/17716/ |
| Task 1 data (Public Test) | Drive folder `1e4XctfiDz9TNPuxYtNJ3Uoaz0vQ9gB1t` |
| Task 1 scoring program | Drive file `12QTJfS_GlilTibz4k3jV1c8q8_BU0V36` — vendored |
| Task 2 scoring program | Drive file `1HS5SqEZIoWiOqzNwtUdzvAnug8zNdXsJ` — vendored |
| Task 1 overview doc | Google Doc `1ZibamHVM21GvwlZjVL0FlGlpe8mI-Yod` |
| Task 2 overview doc | Google Doc `1TGzgRSQvTqefMtlgwVYQ98lsth_j-iWt` |
| Journal (for the paper) | Tạp chí Phát triển KH&CN ĐHQG-HCM |

Contact `dsc@uit.edu.vn`; system faults go on the Codabench forum.

## 3. Submissions

- **10 per day per team.** Leaderboard shows each team's **best of that day**.
- Must be submitted through the **registered Organization** on Codabench. Only
  Organization submissions are valid.
- Both tasks: a `submission.zip` containing exactly one `submission.json`.

## 4. Task 1 — LegalIR

### Data

| File | Content |
|---|---|
| `train.json` | training questions |
| `public-official.json` | Public Test questions (note the **hyphen**) |
| `selected-contexts.zip` | corpus — many `context_*.json` |
| `DSC2026_Task1_LegalIR_Data_Overview.docx` | the task document |

Corpus record (`context_*.json`):
```json
{"link": "https://thuvienphapluat.vn/...", "name": "Quyet-dinh-5868-QD-BYT-...",
 "passage": "BỘ Y TẾ\r\n\n...", "id": 740}
```
Questions (`train.json`) — a JSON **object keyed by question id**:
```json
{"147194": {"question": "...", "answer": ["177504"]}}
```
So gold labels are **whole-document ids**. Retrieve at any granularity you like,
but you must aggregate back to the document id before submitting
(`--aggregate max`).

**The id type trap.** In `context_*.json` the id is a JSON **integer**
(`"id": 740`). In `train.json` the gold answers are JSON **strings**
(`["177504"]`). BTC's scorer intersects the two raw sets, and
`{740} & {"740"} == set()`. Submit integer ids and every question scores **0.0
on both metrics with no error message at all**. Always `str()` your doc_ids —
`ingest.py` does it, and `metrics.check_submittable()` now refuses a submission
that contains non-strings.

**Empty and duplicate passages** (source: BTC email, 20/08). BTC confirms the
**train** corpus contains some documents with an empty `passage` and some
duplicated passages; **public test and private test answers contain neither**.
Teams are told to preprocess train carefully. Measured on the Public Test drop:

| | count | effect |
|---|---|---|
| documents with an empty passage | 20 | unretrievable by content |
| of those, gold for some train question | 6 | |
| train questions with only empty gold | 9 | local train recall ceiling **0.9987** |
| exact-duplicate passage groups | 4 (9 docs) | affects 4 train questions |

`ingest.py --validate` reports both and writes
`data/processed/quarantine_queries_train.jsonl`. Exclude those questions when
mining training pairs — a positive pair whose document is empty teaches noise —
and subtract them before comparing your local recall against a leaderboard score.

### Submission format
```json
{"147194": {"answer": ["177504", "740"]}}
```

### Scoring — the exact code

From [`btc_eval/scoring_legalir.py`](../../phases/0_harness/btc_eval/scoring_legalir.py):

```python
recall    = mean([ |truth&pred| / |truth|   if 0 < len(pred) <= 5 else 0 ])
precision = mean([ |truth&pred| / len(pred) if 0 < len(pred) <= 5 else 0 ])
```

| Fact | Consequence |
|---|---|
| **≤ 5 document ids per question** | 6 ids ⇒ that question scores **0 on both metrics**. Not truncation — zeroing. |
| **Macro averaging** (`.mean()` over questions) | every question weighs the same |
| Empty prediction ⇒ 0 | never submit an empty answer list |
| Precision denominator is `len(pred)`, **not** `len(set(pred))` | a duplicate id lowers precision *and* burns one of the five slots |
| `if len(ids_preds) != len(ids_truth): raise` | a missing or extra question makes the **whole submission fail**, not score badly |
| **Recall primary, Precision tiebreak** | confirmed 02/08 — an earlier email said the reverse and was corrected |

> The "return everything" exploit is closed by the 5-id cap. The real game is
> spending five slots well: 1 id where the retriever is confident (precision 1.0
> at no recall cost), 5 where it is not.

## 5. Task 2 — LegalQA

Same corpus. Questions are an object keyed by id; the `answer` is **long prose**:

```json
{"id": {"question": "Trách nhiệm của tổ chức đấu thầu...",
        "answer": "Theo Điều 37 Nghị định 153/2020/NĐ-CP, được sửa đổi bởi khoản 26 Điều 1 Nghị định 65/2022/NĐ-CP quy định cụ thể:\n- Tuân thủ quy định...\n- Thực hiện chế độ báo cáo..."}}
```

### Submission format
```json
{"9001": {"answer": "Theo Điều 37 ... quy định cụ thể: - ..."}}
```

### Scoring

From [`btc_eval/scoring_legalqa.py`](../../phases/0_harness/btc_eval/scoring_legalqa.py):

| Metric | Role | Implementation |
|---|---|---|
| **METEOR** | **primary** | `nltk.translate.meteor_score`, macro-averaged |
| ROUGE-L | secondary | `rouge_score.RougeScorer(['rougeL'], use_stemmer=False)` |

Two details that shape the whole approach:

1. **No word segmentation.** The `ViTokenizer` call is *commented out* in their
   source; scoring is over plain `.split()` whitespace tokens.
2. **METEOR is recall-weighted** (NLTK's default α=0.9) and penalises
   fragmentation. Long answers that cover the reference's content **in the
   reference's order** score well. Terse answers are punished — which is why the
   `concise` prompt is a control here, not a favourite, and why `mimic` (which
   reproduces the reference's "Theo … quy định: - …" shape) usually wins.

Same key-count rule: a missing question makes the submission fail.

## 6. Model and data rules

Source: BTC's consolidated Q&A email, 02/08.

- **< 4 billion parameters total per task**, summed over **every component**,
  explicitly **including the embedding layer**.
- **Distillation is fine** — if the distilled model is itself under 4B, it is legal.
- **LoRA and quantization do NOT make a >4B model legal.** BTC states this
  directly: those techniques change bits-per-parameter or trainable-parameter
  count, not the parameter count. Using a >4B base means more pretrained
  information, which is the unfairness the rule exists to prevent.
- **No APIs at all** — including free/non-commercial ones. Only open-weight
  models you hold and control locally.
- **No commercial products.** Licences must permit non-profit research/education.
- **BTC data only. No data augmentation. No external data.**
  - Clarification: a pretrained model's own training corpus does **not** count as
    external data — you are using an estimator, not the corpus.
- **No cross-task data use** (source: BTC email, **20/08** — newest rule).
  > "Do hai tác vụ được triển khai độc lập, không giao nhau về câu hỏi cũng như
  > context nên các nhóm **không được** sử dụng dữ liệu của tác vụ này cho tác vụ kia."

  The two tasks are independent and share neither questions nor contexts, so
  Task 1 data may not be used for Task 2 and vice versa. What this rules out in
  this repo, concretely:

  | | verdict |
  |---|---|
  | Phase 2/3 bi-encoder or reranker **checkpoint** fine-tuned on Task 1 pairs, reused for Task 2 | **forbidden** — the weights are Task 1 data |
  | Retrieving Task 2 questions over Task 1's `corpus_*.jsonl` | **forbidden** — Task 2 ships its own `selected-contexts.zip` |
  | The same off-the-shelf pretrained model, loaded fresh and fine-tuned on Task 2 data only | allowed |
  | The same *method* — chunking scheme, k1/b, fusion weights, cutoff rule | allowed, a recipe is not data |

  `phases/4_task2_qa/retrieval_stage.py` enforces the two detectable cases and
  exits 2. Keep Task 1 and Task 2 checkpoints in separate directories.
- **Packaging:** Docker is optional. GitHub or a zip is fine. Downloading weights
  from the internet at run time is fine, provided they are open/non-commercial.
  What matters is a README with step-by-step reproduction that BTC can follow.

## 7. The paper

Source: BTC email, 08/08. Ranked teams are invited to submit to *Tạp chí Phát
triển Khoa học và Công nghệ ĐHQG-HCM* (peer-reviewed).

BTC asks each paper to have:
- a **hypothesis/assumption** about how to solve the task, tested experimentally;
- **enough scenarios** to actually test it;
- **numbers and analysis** for every method tried: why was this one not good
  enough, where exactly did it fall short, and what weakness did the next method fix?

Their research question, verbatim in substance:

> With limited compute (systems under 4B parameters) and modest data (~10k points
> per task), how do pure deep-learning methods compare against methods that
> leverage the power of large language models?

And explicitly: **the approach is not limited to models/systems — it extends to
data-processing strategies**, within the no-augmentation / no-external-data rules.
That makes the chunking-granularity ablation a first-class contribution, not
plumbing.
