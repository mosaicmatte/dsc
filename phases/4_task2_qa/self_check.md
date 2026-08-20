# Phase 4 — self-check answer key

### 1. Fluent, confident, wrong answers. Two root causes and the one measurement that separates them.

> **Cause A — retrieval failure.** The gold passage is not in the context at all.
> The model has nothing to ground on, so it falls back on parametric knowledge
> and produces something plausible. This is the most common cause and it is not a
> generator problem.
>
> **Cause B — reader failure.** The gold passage *is* in the context and the model
> still answers wrongly: it attends to a distractor, misreads a negated condition,
> or answers a subtly different question.
>
> **The measurement: `recall@k` at your chosen context size.** It is printed by
> `retrieval_stage.py` and again by `baseline_generative.py`.
>
> If recall@3 = 0.62, then 38% of questions cannot be answered correctly no
> matter what the generator does — that is cause A, and the fix is retrieval, not
> prompting. If recall@3 = 0.95 and you are scoring 0.60, the passages are there
> and cause B dominates: fix the prompt, the context size, or the answer format.
>
> Concretely: split your dev failures into "gold passage was in context" and "was
> not". Those two buckets need completely different work, and mixing them is how
> teams spend a week tuning prompts to fix a retrieval bug.

### 2. Top-5 scores worse than top-3. Give the mechanism.

> Two forces pull opposite ways as k rises, and past some point the second wins.
>
> **Up:** recall@5 ≥ recall@3, so the gold passage is present more often.
>
> **Down:** two extra distractor passages enter the context. A 1.5B model has
> limited capacity to attend over long inputs, and attention degrades with
> position — the "lost in the middle" effect means a gold passage sitting at
> position 4 of 5 can be effectively invisible even though it is technically
> present. Additionally, the extra passages are the *lower-ranked* ones, so they
> are the most likely to be topically adjacent but wrong: exactly the distractors
> that mislead.
>
> Net effect: the recall gain from ranks 4–5 is small (those ranks contain the
> gold passage rarely), while the distraction cost applies to every question. So
> the curve turns down.
>
> This is why the ablation exists and why "more context is better" is not a safe
> default. Report the curve. The turning point is a real finding.

### 3. Why does LoRA not help you fit under the 4B ceiling?

> Because the ceiling counts **the parameters that run at inference**, and LoRA
> does not remove any of them. A LoRA-tuned Qwen2.5-3B still loads all 3.09B base
> parameters and then adds ~30M adapter parameters on top — it is 3.12B, slightly
> *more* than the base model, not less.
>
> What LoRA actually saves is **training memory**: only the adapter weights get
> gradients and optimiser states, so you can fine-tune a 3B model on a GPU that
> could not hold the full-fine-tuning optimiser state. That is a real and useful
> benefit. It has nothing to do with eligibility.
>
> Practical consequence: pick the base model that fits the budget *first*
> (`python src/params.py`), then decide whether to LoRA it. Never the reverse.

---

### Bonus

**Why is greedy decoding (`--temperature 0`) the default?**
> Reproducibility, which the Phase 5 package requires, and because for an
> extraction-shaped task sampling adds variance without adding accuracy. If you
> do sample, fix the seed and log it, or your frozen pipeline will not reproduce
> its own Public Test score.

**The `passage` extractive baseline costs zero parameters. Why run it at all?**
> Because if it is competitive, that is a major finding: it means the metric
> rewards recall over precision in the answer string, and your entire generator
> budget could be spent elsewhere. It takes five minutes and it calibrates every
> number that follows.
