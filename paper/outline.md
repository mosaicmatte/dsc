# Paper outline — assembled, not written

> **TODO(TEAM/ongoing): fill in each section at the END OF ITS PHASE, not in October.**

> The paper BTC expects (hypothesis → scenarios → failure analysis) is not written in
> October. It is **assembled** from the log you keep starting today. Every section below
> names the artefact that fills it. If the artefact does not exist, the section cannot be
> written, and that is the signal to go make the artefact.

---

## 1. Introduction

- Task, dataset, the 4B ceiling, no-API constraint.
- **The research question, stated as BTC framed it:** under a 4B parameter ceiling and
  ~10k training examples, where does pure deep learning stop paying off and where does an
  LLM-based approach begin to?
- Our answer in one sentence. (Write this last; it is the ablation ladder's conclusion.)

*Artefact: none — but the answer must be defensible from §4.*

## 2. Data

- Corpus and query statistics, granularity, relevant-documents-per-query distribution.
- **The Precision/Recall consequence**: because most queries have few relevant documents,
  a variable-length answer set is not an optimisation, it is a requirement.

*Artefact: `analysis/dataset_stats.md` — the table goes in verbatim.*

## 3. Method

3.1 Preprocessing — Unicode NFC, tone-placement normalisation, segmentation, and the
    PhoBERT/BGE-M3 input-format split.
3.2 Chunking — document vs điều, and the parent-aggregation step.
3.3 Lexical retrieval — BM25, tuned k1/b.
3.4 Dense retrieval — bi-encoder, contrastive fine-tuning, the three negative-mining rounds.
3.5 Hybrid fusion — RRF vs weighted score normalisation.
3.6 Reranking — cross-encoder, depth, training negatives from our own retriever's errors.
3.7 **Answer-set cutoff as a model component** — this is our distinctive contribution to
    the write-up. Argue it from the metric definition (see `phases/0_harness/self_check.md`
    Q2), not from empirical convenience.
3.8 Task 2 reader.

*Artefacts: `configs/FINAL.yaml`, the phase READMEs.*

## 4. Experiments — the ablation ladder

One table, one fixed dev split, one cutoff procedure:

| System | dev R | dev P | leaderboard | Δ |
|---|---|---|---|---|
| BM25 | | | | — |
| + tuned k1/b, chunking | | | | |
| Dense zero-shot | | | | |
| Dense fine-tuned (in-batch) | | | | |
| + BM25 hard negatives | | | | |
| + self-mined hard negatives | | | | |
| + hybrid fusion | | | | |
| + cross-encoder rerank | | | | |
| + LLM reader (Task 2) | | | | |

Supporting figures:
- cutoff sweep (Precision/Recall/score vs set size) — `analysis/fig_cutoff_sweep.png`
- hybrid weight sweep — `analysis/hybrid_weight_sweep.md`
- retrieval ceiling table — `analysis/ceiling_table.md`

*Artefact: `work/experiments/runs.csv`. If a row is missing, the experiment did not happen.*

## 5. Analysis — why each method was insufficient

This is the section BTC asked for explicitly, and the one most teams skip.

For each rung: 20 categorised dev failures, the dominant category, and what the next rung
fixed. Structure it as a chain:

> BM25 failed on `lexical-mismatch` (n/20) → dense retrieval fixed those but introduced
> `numeric` failures (n/20) → hybrid fusion recovered them → the remaining failures were
> `multi-article` and `negation`, which reranking addressed / did not address because …

**The retrieval-ceiling argument belongs here.** State plainly where the ceiling bound the
result and what would have been needed to raise it. Negative results count: "we mined
harder negatives at round 3 and it stopped helping" is a finding, not a failure.

*Artefacts: `analysis/error_analysis_phase*.md`, `analysis/ceiling_table.md`.*

## 6. Limitations

- Dev-split size and variance; how much of a 1-point difference is noise.
- Label noise found during error analysis, with an estimated rate.
- Anything changed during the Private Test for compatibility
  (`phases/5_freeze/freeze_checklist.md` §7).
- What the 4B ceiling prevented us from testing.

## 7. Reproducibility

Seeds, frozen config, exact model revisions, the reproduction package.

*Artefact: `phases/5_freeze/freeze_checklist.md`, `dist/`.*

---

## Writing schedule

| When | What |
|---|---|
| End of each phase | fill in that phase's §4 rows and §5 paragraph — **while you remember why** |
| 17–18/09 | §2 and §7 (both are pure artefact transcription) |
| after Private Test | §1 conclusion, §6 |

Writing §5 in October from memory produces a worse paper than writing it in August from
a worksheet you filled in the same afternoon.
