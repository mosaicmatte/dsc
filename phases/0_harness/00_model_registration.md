# Registration and paperwork — what to do, step by step

> **TODO(TEAM/phase0-B0): work top to bottom. Item 1 blocks your data access;
> item 2 blocks your submissions being counted.**

There are exactly **two forms**. Neither is hard. Both are blocking.

---

## 1. Data-usage commitment (Cam kết sử dụng bộ dữ liệu) — BLOCKS DATA ACCESS

**Without this, BTC does not give you the Public/Private Test data at all.**
One per team, covers both tasks, submitted by the **team leader** from the
**email registered with BTC**.

| Step | Do this |
|---|---|
| 1 | Download the template: https://docs.google.com/document/d/1ExjVZcL0tGNaOI3wNisUw0hM6qPDTvA1g4ct5Vc8viE/edit |
| 2 | Fill in the team details, sign it (an electronic signature is fine), save as **PDF or DOCX** |
| 3 | Submit it here: https://forms.gle/bmyjfc9Bng9NtfNW9 |

Notes from BTC:
- **Team leader only**, using the registered email.
- **One per team** — if you entered both tasks, you still submit only once.
- Already submitted and confirmed? Do not resubmit.
- If it is incomplete, BTC will contact you to redo it.

> **Check first:** if you already have the Public Test data (`train.json`,
> `public_official.json`, `selected-contexts.zip`), this is already done and you
> can skip to item 2.

---

## 2. Model registration — BLOCKS YOUR SUBMISSIONS BEING VALID

**Registration form:** https://forms.gle/HWE7tcxzWq63Kxv28
**Public approved list:** https://docs.google.com/spreadsheets/d/1c5jzsYezWho1WGLRfMKWOaFPLIk_GTnXP5vV8AOWM2Q/edit

**Window: 06/08 → 18/09/2026.** You may submit the form **as many times as you
like** to add models later.

### How it actually works

1. BTC keeps one **public list of approved models**, shared by all teams.
2. **If a model is already on that list, you do not register it.** Just use it.
3. If a model is *not* on the list, submit the form and wait for BTC to approve
   and add it.
4. **A submission that used an unapproved model is not recognised.** This is the
   part that hurts: you find out after the fact.

### The good news — almost everything this repo uses is already approved

Checked against the list on 20/08/2026. **No action needed for any of these:**

| Role | Model | Status |
|---|---|---|
| Bi-encoder | `AITeamVN/Vietnamese_Embedding` | ✅ approved |
| Bi-encoder | `AITeamVN/Vietnamese_Embedding_v2` | ✅ approved (newer — worth testing) |
| Bi-encoder | `BAAI/bge-m3` | ✅ approved |
| Bi-encoder | `bkai-foundation-models/vietnamese-bi-encoder` | ✅ approved |
| Cross-encoder | `AITeamVN/Vietnamese_Reranker` | ✅ approved |
| Cross-encoder | `BAAI/bge-reranker-v2-m3` | ✅ approved |
| Cross-encoder | `itdainb/PhoRanker` | ✅ approved |
| Cross-encoder | `namdp-ptit/ViRanker` | ✅ approved |
| Generator | `Qwen/Qwen2.5-1.5B-Instruct` | ✅ approved |
| Generator | `Qwen/Qwen2.5-3B-Instruct` | ✅ approved |
| Segmenter | `underthesea` | ✅ approved |
| Segmenter | `VnCoreNLP` | ✅ approved |
| Backbone | `vinai/phobert-base`, `phobert-base-v2` | ✅ approved |

### ⚠️ Two things in this repo that are NOT approved

| Item | Problem | What to do |
|---|---|---|
| **`pyvi`** | **Not on the list.** It is this repo's *default* segmenter. | Either register it, or switch to `underthesea` (already approved). Registering is one form; switching is one flag. |
| `dangvantuan/vietnamese-embedding` | Not on the list | Drop it, or register it. It is a spare candidate, not load-bearing. |

**Do this now:** decide whether to register `pyvi` or switch to `underthesea`.
Every script takes `--segmenter underthesea`, and `src/dense.REGISTRY` records
which models need segmentation at all.

### Legal-domain models already approved — worth a look

These were registered by other teams and are already legal for you to use. Several
are Vietnamese **legal-domain** models, which is unusually well-matched to this task:

| Model | Why it might matter |
|---|---|
| `bqbbao6/vietnamese-legal-embedding` | legal-domain embedding |
| `huyydangg/DEk21_hcmute_embedding_v2` | Vietnamese embedding |
| `ntphuc149/ViLegalBERT` | legal-domain BERT |
| `ntphuc149/ViLegalQwen2.5-1.5B-Base`, `ViLegalQwen3-1.7B-Base` | legal-domain generators |
| `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2` | legal, RL-tuned |
| `mainguyen9/vietlegal-harrier-0.6b` | small legal model |
| `AITeamVN/Vi-Qwen2-1.5B-RAG`, `Vi-Qwen2-3B-RAG` | Vietnamese RAG-tuned generators — a strong Task 2 candidate |
| `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B` | small and modern; leaves budget for Task 2 |

Add a zero-shot row for two or three of these in Phase 2 — it costs one command
each and they may beat the general-purpose models on legal text.

### ⚠️ The "4B" name trap

The approved list contains `Qwen/Qwen3-4B`, `Qwen/Qwen3-Embedding-4B`,
`Qwen/Qwen3.5-4B`, `codefuse-ai/F2LLM-4B`, `Octen/Octen-Embedding-4B`.

**Being on the approved list does not exempt a model from the parameter ceiling.**
They are separate rules. The limit is *strictly under* 4 billion (`dưới 4 tỷ`),
and a model named "-4B" almost certainly has ≥4.0B parameters — and would consume
your entire budget on its own, leaving nothing for a retriever or reranker.

Verify before using any of them:
```bash
python -c "from src.params import count_hf; print(count_hf('Qwen/Qwen3-4B'))"
```

Likewise `*-GGUF` entries: GGUF is a quantization format, and BTC states plainly
that quantization does **not** change the parameter count. A GGUF of a 3B model is
fine because the base is 3B; a GGUF of a 7B model is not.

---

## 3. Before you tick any model off

1. **Verify the parameter count yourself** — `src/params.KNOWN` is planning-grade:
   ```bash
   python -c "from src.params import count_hf; print(count_hf('BAAI/bge-m3'))"
   ```
2. **Record the HF revision (commit SHA)** you will use, not just the repo name.
   Model cards get updated mid-competition; the freeze in Phase 5 needs the SHA.
3. **Check the licence** permits non-commercial research/education use.
4. **Check the budget for the whole pipeline**, not one model:
   ```bash
   python src/params.py
   ```

## 4. The rules that decide what is even registrable

From BTC's model-registration email, 05/08:

- **< 4 billion parameters total per task**, across **every component** — generator,
  embedder, reranker, anything else in the pipeline.
- **Distillation is fine** if the distilled model is itself under 4B.
- **LoRA, quantization, GPTQ, AWQ, GGUF do NOT reduce the parameter count.** A >4B
  model is illegal no matter how you compress it.
- **No APIs at all**, commercial or not. Open weights you download and run yourself.
- Non-commercial / research / education licences are acceptable.

---

## 5. Status log — fill this in

| Date | Action | Outcome |
|---|---|---|
| | Data-usage commitment submitted | |
| | Decided: register `pyvi` **or** switch to `underthesea` | |
| | Extra models registered (list them) | |
| | Re-checked the approved list | |

> Re-check the public list every week or so — BTC updates it as other teams
> register, and something you want may already be approved.
