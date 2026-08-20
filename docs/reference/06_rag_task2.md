# 06 — RAG and Task 2

Prerequisite: `phases/4_task2_qa/README.md` Part A.

---

## 1. The decomposition that assigns blame

```
question → [retrieve] → [rerank] → top-k passages → [read] → answer
```

Final accuracy is bounded above by retrieval:

```
Accuracy ≤ recall@k
```

because a passage that is not in the context cannot be read. So **any Task 2 debugging
starts by splitting dev failures into two buckets**:

| bucket | test | fix |
|---|---|---|
| gold passage **not** in context | `recall@k` at your chosen k | retrieval — prompting cannot help |
| gold passage **in** context, answer still wrong | the remainder | reader: prompt, context size, format |

Mixing these is how a week disappears tuning prompts to fix a retrieval bug.
`retrieval_stage.py` prints `recall@1/@3/@5` explicitly, and warns if `recall@5 < 0.70`.

## 2. Why Task 2 retrieval is not Task 1 retrieval

Task 2 questions are usually a different distribution:

| | Task 1 query | Task 2 question |
|---|---|---|
| length | shorter, keyword-ish | longer, conversational |
| quotes statute numbers | often | less often |
| shape | "find the provision" | "what happens if…" scenario |

A retriever tuned on Task 1 may transfer poorly. **Measure it before assuming.** If Task 2
recall is much worse, the likely fixes are: lower the BM25 weight in fusion (questions
paraphrase more, so lexical matching helps less), and consider finer chunking (a scenario
question matches a specific khoản, not a whole điều).

## 3. Extractive vs generative — decided by the data, not by taste

Run the mechanical check first (`00_task2_eval_notes.md` Q2, and
`baseline_extractive.py --oracle-ceiling`): what fraction of gold answers appear **verbatim**
in a retrieved passage?

| verbatim rate | implication |
|---|---|
| high (> 0.8) | extraction is viable and cheap; a generative paraphrase may score **zero** under exact match |
| middling | hybrid: extract when possible, generate otherwise |
| low (< 0.4) | answers are synthesised; extraction has a hard ceiling it can never pass |

`--oracle-ceiling` reports exactly this number, and it is the maximum any extractive reader
can score. Getting it takes two minutes and can save two days.

**Do also run the zero-parameter `--mode passage` baseline.** If returning the whole top-1
passage is competitive, that tells you the metric rewards recall over precision in the
answer string — which changes what the generator should be optimised for, and frees budget.

## 4. Context size is not monotonic

Two opposing forces as `k` grows:

- **up:** `recall@k` increases — the gold passage is present more often
- **down:** more distractors; attention degrades over long inputs, and the extra passages are
  the *lower-ranked* ones, hence the most topically-adjacent-but-wrong

The marginal recall from ranks 4–5 is small (they rarely contain the gold passage) while the
distraction cost applies to every question. So the curve typically turns down. For a 1.5B
model the optimum is often top-3.

The position effect is documented in the literature as "lost in the middle": models attend
best to the beginning and end of a long context and worst to the middle. Practical
implication — **passage order matters**, and putting the highest-scoring passage first is
not merely conventional.

Ablate it (`ablate_context.py --top-k 1 3 5`). Report the turning point; it is a finding.

## 5. Grounding

In descending order of cost and effect:

1. **Prompt instruction** — "only use the provided passages; if they do not contain the
   answer, say so". Free. Helps most when retrieval is weak, because it converts confident
   hallucinations into abstentions.
2. **Numbered passages + required citation** — makes a wrong citation visible, which is worth
   points if the metric rewards grounding and worth error-analysis signal regardless.
3. **Answer-length constraint** — matters under token-F1, where a long correct answer is
   penalised on precision against a short gold string.
4. **Constrained decoding / extraction** — strongest guarantee, most work.

`prompts.py` implements 1–3 as the `grounded`, `cited` and `concise` variants, plus a
`minimal` control. **Always run the control**: if the elaborate prompts do not beat it, they
are noise, and you should say so rather than shipping complexity.

Prompts are in Vietnamese deliberately — instruction-following degrades when the prompt
language does not match the content language.

## 6. Budget arithmetic

Task 2 total = retriever + reranker + generator, under 4B.

| Configuration | Total | Fits |
|---|---|---|
| BGE-M3 embed (0.57) + BGE-M3 rerank (0.57) + Qwen2.5-1.5B (1.54) | 2.68B | yes |
| BGE-M3 embed + BGE-M3 rerank + Qwen2.5-3B (3.09) | 4.23B | **no** |
| PhoBERT embed (0.14) + PhoRanker (0.14) + Qwen2.5-3B | 3.37B | yes |

Note the third row: **dropping to PhoBERT-class retrieval buys you the 3B generator.**
Whether that trade is worth it is an empirical question and a good ablation — it directly
addresses BTC's research question about where deep learning stops paying off relative to
LLM-based approaches.

**LoRA does not change any number in that table.** A LoRA-tuned 3B model loads all 3.09B base
parameters plus ~30M of adapter — slightly *more*, not less. LoRA saves training memory.
Pick the base model that fits first, then decide whether to LoRA it.

## 7. When LoRA is actually the right call

Only after context-size and prompt-format ablations. Those are far cheaper and often move
the metric more on ~10k examples.

- Error analysis says the model retrieves correctly and answers in the **wrong format** →
  LoRA is the right fix.
- Error analysis says the model **hallucinates** → fix retrieval or grounding. Fine-tuning on
  10k examples will not teach it the law.

And mask the loss to answer tokens only. Training on the prompt tokens too wastes capacity
teaching the model to reproduce passages it is already being shown —
`train_generator_lora.py` masks with `-100`.

---

## Check yourself

1. `recall@3 = 0.58` and your Task 2 score is 0.44. How much headroom does prompt engineering
   have, at most?
2. The `--mode passage` baseline (return the whole top-1 passage, zero parameters) scores
   within 3 points of your 1.5B generator. What does that tell you?
3. Why is greedy decoding the default in `baseline_generative.py`?

<details><summary>answers</summary>

1. **At most 14 points** (0.58 − 0.44), and only if the reader becomes perfect on every
   question whose gold passage is present. The other 42% of questions have no gold passage in
   context and are unreachable by any prompt. If you want more than 14 points, the work is in
   retrieval. This subtraction should be the first thing you compute on any Task 2 result.
2. That the metric **rewards recall over precision in the answer string** — a long passage
   containing the answer scores nearly as well as the answer itself. That reshapes the whole
   approach: prefer the `concise` prompt only if it wins empirically, consider extraction over
   generation, and question whether the generator's parameter budget is buying anything. It is
   also a strong paper finding about the metric.
3. **Reproducibility**, which the Phase 5 package requires — a sampled run does not reproduce
   its own Public Test score. And for an extraction-shaped task, sampling adds variance
   without adding accuracy. If you do sample, fix and log the seed.
</details>
