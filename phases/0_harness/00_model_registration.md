# Task B0 — Model registration (BLOCKING — do today, 20/08)

**Why this is first.** Approval is not instant. A model that is not on the approved list
invalidates every submission that used it — including a Private Test run you cannot redo.
The form can be resubmitted later to add models, so **over-register**: list every
candidate you might plausibly touch, not just the one you intend to use.

## What to submit

| Role | Candidate | Params (verify!) | Backbone | Segmented input? | Licence |
|---|---|---|---|---|---|
| Segmenter | `pyvi` | rule/statistical | — | — | |
| Segmenter | `underthesea` | | | — | |
| Segmenter | VnCoreNLP | register if model-based | | — | |
| Bi-encoder | `bkai-foundation-models/vietnamese-bi-encoder` | ~135M | PhoBERT-base | **YES** | |
| Bi-encoder | `AITeamVN/Vietnamese_Embedding` | ~568M | BGE-M3 | no | |
| Bi-encoder | `dangvantuan/vietnamese-embedding` | ~135M | PhoBERT-base | **YES** | |
| Bi-encoder | `BAAI/bge-m3` | ~568M | XLM-R-large | no | |
| Cross-encoder | `AITeamVN/Vietnamese_Reranker` | ~568M | BGE-M3 | no | |
| Cross-encoder | `itdainb/PhoRanker` | ~135M | PhoBERT-base | **YES** | |
| Cross-encoder | `namdp-ptit/ViRanker` | ~135M | PhoBERT-base | **YES** | |
| Cross-encoder | `BAAI/bge-reranker-v2-m3` | ~568M | XLM-R-large | no | |
| Generator | `Qwen/Qwen2.5-1.5B-Instruct` | ~1.54B | — | no | |
| Generator | `Qwen/Qwen2.5-3B-Instruct` | ~3.09B | — | no | |

## Before you tick each row

1. **Verify the parameter count yourself** on the model card — the table above is
   planning-grade, not authoritative:
   ```bash
   python -c "from src.params import count_hf; print(count_hf('BAAI/bge-m3'))"
   ```
2. **Verify the licence permits research/educational use.** Note it in the table.
   A permissive-looking model with a restrictive licence is a disqualification risk.
3. **Record the exact revision** you will use (HF commit SHA), not just the repo name.
   Model cards get updated mid-competition.

## Budget sanity check before registering the generator

```bash
python src/params.py
```
Task 2 must fit retriever + reranker + generator under 4B **in total**.
BGE-M3-class embedder (0.57B) + bge-reranker-v2-m3-class (0.57B) = ~1.14B, leaving
~2.86B for the generator — Qwen2.5-3B (3.09B) does **not** fit alongside both.
Either use the 1.5B generator, or drop to a PhoBERT-class reranker for Task 2.

## Status log

| Date | Action | Outcome |
|---|---|---|
| 20/08 | initial bulk registration submitted | _pending_ |
