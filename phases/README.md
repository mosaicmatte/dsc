# The phases

Six folders, numbered in the order you do them. **Work through them in sequence.**

Each folder is a lesson *and* a task list:

- **Part A — Learn**: the concepts, with the reasoning, not just the recipe
- **Part B — Do**: numbered tasks, each naming its script and its definition of done
- **Part C — Self-check**: questions, answer key in that folder's `self_check.md`
- **Definition of done**: the checklist that says you may move on

| | Folder | Dates | Objective | Deliverable |
|---|---|---|---|---|
| 0 | [`0_harness`](0_harness/) | 20–23/08 | ground truth & instrumentation | working scorer, dev split, dataset stats |
| 1 | [`1_bm25`](1_bm25/) | 24–27/08 | lexical baseline | first valid submission; dev↔leaderboard confirmed |
| 2 | [`2_dense`](2_dense/) | 28/08–03/09 | dense retrieval | fine-tuned bi-encoder, hybrid fusion, 4+ ablation rows |
| 3 | [`3_rerank`](3_rerank/) | 04–08/09 | reranking | full Task 1 pipeline + retrieval-ceiling table |
| 4 | [`4_task2_qa`](4_task2_qa/) | 09–14/09 | Task 2 LegalQA | Task 2 baseline submitted |
| 5 | [`5_freeze`](5_freeze/) | 15–18/09 | freeze & package | reproduction package; frozen pipeline |

**Phase 0 produces no score. Do not skip it.** Everything after it is unmeasurable without
a scorer you trust and a dev split that predicts the leaderboard.

## At the end of every phase

```bash
python tools/error_analysis.py --run work/experiments/runs/<best>.jsonl --phase <n> -n 20
```

Twenty categorised failures. BTC explicitly asked for analysis of *why* a method
underperformed and what the next one fixed — this is that section, and it is far easier to
write the same afternoon than in October.

## Splitting work across the team

Phases are sequential; inside a phase the tasks parallelise:

| Phase | Splits into |
|---|---|
| 0 | (a) read BTC's eval code + wrap it · (b) ingest + dev split + stats |
| 1 | (a) chunking + BM25 grid · (b) cutoff sweep + submission pipeline |
| 2 | (a) zero-shot comparison + hybrid · (b) fine-tuning rounds 1–3 |
| 3 | (a) zero-shot reranker comparison · (b) cross-encoder fine-tuning |
| 4 | (a) Task 2 retrieval + eval notes · (b) reader baselines + ablations |

Whoever is not on the critical path does the error analysis. It is the most commonly
skipped and most directly requested deliverable.
