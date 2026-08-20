# Phase 4 — Task 2, LegalQA
**09–14/09 · 6 days**

> Goal: a submitted Task 2 baseline, with the retrieval stage inherited from Task 1 and
> the reader chosen by measurement rather than by assumption.

---

## PART A — Learn

### A1. RAG anatomy: retrieve → rerank → read

```
question ──► [retriever] ──► top-100 ──► [reranker] ──► top-k ──► [reader] ──► answer
```

**The reader is only as good as what the retriever hands it.** If the gold passage is not
in the top-k, no reader can produce a grounded answer — it will hallucinate something
plausible instead, and you will spend two days blaming the generator.

So: **measure retrieval quality on the Task 2 data separately, before touching the reader.**
Task 2's questions are not the same distribution as Task 1's queries — they may be longer,
more conversational, or reference a scenario rather than a statute. A retriever tuned on
Task 1 may transfer poorly, and you need to know that as a number, not a suspicion.
That is Task B2 below, and it is not optional.

### A2. Extractive vs abstractive answering

| | Extractive | Abstractive (generative) |
|---|---|---|
| Output | a span copied from a passage | free text |
| Can it hallucinate? | no — output is a substring | yes |
| Handles multi-passage synthesis | poorly | yes |
| Handles rephrasing | no | yes |
| Params | ~135M (QA head on PhoBERT) | 0.5–3B |

**BTC's answers are long, structured prose**, e.g. *"Theo Điều 37 Nghị định
153/2020/NĐ-CP … quy định cụ thể: - … - …"*. They are not spans, so a pure extractive head
cannot reach them.

**The metric is METEOR (primary) and ROUGE-L (secondary)**, macro-averaged, computed on
plain whitespace tokens — BTC's scorer has the Vietnamese tokenizer commented out. METEOR
is **recall-weighted**, so covering the reference's content in the reference's order is
what scores; brevity is punished. See
[`docs/reference/09_official_rules.md`](../../docs/reference/09_official_rules.md) §5.

### A3. Grounding

Forcing the answer to cite (or be constrained to) the retrieved passage reduces
hallucination and usually improves the metric. Practical forms, cheapest first:
- prompt instruction: *"Only answer using the provided passages. If they do not contain
  the answer, say so."*
- output format that names the source article, so a wrong citation is visibly wrong
- constrained decoding / extraction from the passage

### A4. Context size is not monotonic

More retrieved passages in the prompt is **not** monotonically better:
- more passages → higher chance the gold passage is present (good)
- more passages → more distractors, and small models attend badly over long contexts;
  the "lost in the middle" effect means a gold passage at position 4 of 5 may be
  effectively invisible (bad)

For a 1.5B model the optimum is often top-3. **Ablate it** (Task B5) — it is one of the
cheapest measurable wins in the phase.

### A5. Budget — check before you train anything

Task 2 total = retriever + reranker + generator, all under **4B**.

| Configuration | Total | Fits? |
|---|---|---|
| BGE-M3 embedder (0.57B) + BGE-M3 reranker (0.57B) + Qwen2.5-1.5B (1.54B) | **2.68B** | yes |
| BGE-M3 embedder + BGE-M3 reranker + Qwen2.5-3B (3.09B) | **4.23B** | **NO** |
| PhoBERT embedder (0.14B) + PhoRanker (0.14B) + Qwen2.5-3B | **3.37B** | yes |

```bash
python src/params.py
```

**LoRA does not help you here.** A LoRA-tuned Qwen2.5-3B still counts as 3.09B parameters
for eligibility. LoRA saves training memory, not budget.

---

### Going deeper (optional)

[`docs/reference/06_rag_task2.md`](../../docs/reference/06_rag_task2.md) — the blame-assignment
decomposition, why Task 2 retrieval differs from Task 1, the extractive/generative decision as
a measurement, grounding techniques ranked by cost, and the budget trade that buys you a 3B
generator.

---

## PART B — Do

### Task B1 — Read the Task 2 evaluation code FIRST
→ fill in [`00_task2_eval_notes.md`](00_task2_eval_notes.md)
Exactly as in Phase 0. The answer format determines the entire approach; **do not guess it.**
**Done when:** you can state the metric, the required output shape, and whether answers
are spans or free text, each with a line reference.

### Task B2 — Port the Task 1 retriever and measure it on Task 2 data
```bash
python phases/4_task2_qa/retrieval_stage.py --queries data/processed/task2_dev.jsonl \
    --model <task1 best> --reranker <task1 best>
```
**Done when:** you know Task 2's recall@1/@3/@5. That number is the ceiling on every
reader below, and it belongs in the log before any reader work starts.

### Task B3 — Baseline A: extractive
```bash
python phases/4_task2_qa/baseline_extractive.py --top-k 3
```
Only if answers are spans. Skip with a note in the log if they are not.

### Task B4 — Baseline B: generative
```bash
python phases/4_task2_qa/baseline_generative.py --model Qwen/Qwen2.5-1.5B-Instruct --top-k 3
```
**Done when:** total system parameters confirmed under 4B and logged in the `n_params` column.

### Task B5 — Ablate context size
```bash
python phases/4_task2_qa/ablate_context.py --model <chosen> --top-k 1 3 5
```
**Done when:** the top-1/3/5 table is in `work/analysis/` — including the case where more
context made it worse, which is the interesting result.

### Task B6 — Ablate the prompt / answer format
```bash
python phases/4_task2_qa/ablate_context.py --model <chosen> --prompts all
```
**Done when:** the prompt variants table is in `work/analysis/`.

### Task B7 — (if time permits) LoRA fine-tune the generator
```bash
python phases/4_task2_qa/train_generator_lora.py --model <chosen> --epochs 1
```
Permitted, but it does **not** reduce the base model's parameter count for eligibility.

---

### Task B8 — Write your own code  ← `TODO(YOU/phase4)`

[`phases/4_task2_qa/prompts.py`](prompts.py) — the `"yours"` template. Prompt wording is
the cheapest lever in this phase, far cheaper than fine-tuning. Read 20 real gold answers
first, then copy their structure; `mimic` is the variant to beat.

```bash
python phases/4_task2_qa/ablate_context.py --model <m> --prompts yours mimic grounded
```

---

## PART C — Self-check

1. Your generator produces fluent, confident, wrong answers. Name the two distinct root
   causes and the one measurement that separates them.
2. Top-5 context scores worse than top-3. Give the mechanism.
3. Why does LoRA not help you fit under the 4B ceiling?

Key in [`self_check.md`](self_check.md).

---

## Definition of done for Phase 4

- [ ] `00_task2_eval_notes.md` answered with line references
- [ ] Task 2 retrieval recall@1/@3/@5 measured and logged **before** any reader work
- [ ] At least one reader baseline submitted to Codabench
- [ ] Context-size ablation table in `work/analysis/`
- [ ] Prompt-format ablation table in `work/analysis/`
- [ ] `python src/params.py` confirms the full Task 2 stack is under 4B
- [ ] 20 dev failures categorised in `analysis/error_analysis_phase4.md`
