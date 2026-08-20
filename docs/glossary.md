# Glossary

Terms used across the phase READMEs. Vietnamese legal terms are kept in Vietnamese
because that is how they appear in the data.

## Legal document structure

| Term | Meaning |
|---|---|
| **văn bản** | a whole legal document (a Luật, Nghị định, Thông tư…) |
| **chương** | chapter |
| **mục** | section |
| **điều** | article — usually the unit that actually answers a question |
| **khoản** | clause, numbered `1.`, `2.` inside an điều |
| **điểm** | point, lettered `a)`, `b)` inside a khoản |
| **Nghị định 100/2019/NĐ-CP** | a decree identifier — verbatim in both query and source, and the single highest-IDF token type in the corpus |

Nesting: `văn bản > chương > mục > điều > khoản > điểm`.

## Retrieval

| Term | Meaning |
|---|---|
| **corpus / collection** | the searchable set of documents |
| **granularity** | which nesting level is one indexed unit. Ablation #1 |
| **chunk** | one indexed unit after splitting; `meta.parent_id` points at its văn bản |
| **run** | a full ranking per query: `{qid, ranked:[[doc_id, score], …]}` |
| **prediction** | the variable-length answer set after a cutoff: `{qid, predicted:[…]}` |
| **qrels** | the gold labels: which documents are relevant to which query |
| **depth** | how many candidates retrieval returns; sets the ceiling for everything after |
| **cutoff rule** | how a ranking becomes an answer set. A model component, not formatting |
| **retrieval ceiling** | `recall@depth` of the retriever — no reranker over that list can exceed it |

## Metrics

| Term | Definition | Note |
|---|---|---|
| **Precision** | `\|pred ∩ rel\| / \|pred\|` | falls as you return more |
| **Recall** | `\|pred ∩ rel\| / \|rel\|` | **primary metric**; rises monotonically as you return more |
| **F1 / F2** | harmonic mean of P and R; F2 weights recall 2× | |
| **Recall@k** | recall of the top-k of a ranking | the ceiling metric |
| **MRR** | mean of 1/(rank of first hit) | "how fast to one answer" |
| **MAP** | mean of averaged precision at each hit | overall ranking quality |
| **nDCG@k** | position-discounted gain | ranking quality with log discount |
| **micro-average** | pool all queries, then one ratio | many-answer queries dominate |
| **macro-average** | per-query, then average | every query weighs the same |

## Models

| Term | Meaning |
|---|---|
| **BM25** | lexical scoring: IDF × saturated term frequency ÷ length normalisation |
| **k1** | BM25 term-frequency saturation. 0 = binary presence; large = raw counts |
| **b** | BM25 length normalisation. 0 = none; 1 = full |
| **IDF** | inverse document frequency — how surprising a term is |
| **bi-encoder** | encodes query and document separately; corpus embeddable offline |
| **cross-encoder** | encodes the pair jointly; far more accurate, one pass per candidate |
| **retrieve-then-rerank** | cheap model narrows to ~50–100, expensive model reorders those |
| **InfoNCE / MNRL** | contrastive loss: softmax over one positive and many negatives |
| **in-batch negatives** | other examples in the same batch used as negatives, free |
| **temperature (τ) / scale** | sharpness of the contrastive softmax; `scale = 1/τ` |
| **hard negative** | a document that looks relevant and is not — the useful kind |
| **false negative** | a mined "negative" that is actually relevant but unlabelled. Damages training |
| **RRF** | Reciprocal Rank Fusion: `Σ w/(K + rank)`. Scale-free |
| **PhoBERT backbone** | **requires** word-segmented input (`người_lao_động`), 256-token limit |
| **BGE-M3 / XLM-R backbone** | **must not** get segmented input; 8192-token limit |

## Task 2

| Term | Meaning |
|---|---|
| **RAG** | retrieve → rerank → read |
| **extractive** | the answer is a span copied from a passage. Cannot hallucinate |
| **abstractive / generative** | the answer is generated text. Can hallucinate |
| **grounding** | forcing the answer to come from (or cite) the retrieved passage |
| **lost in the middle** | attention degrades for passages in the middle of a long context |
| **LoRA** | low-rank adapters. Saves **training memory**, not parameter budget |

## Process

| Term | Meaning |
|---|---|
| **BTC** | Ban Tổ Chức — the organisers |
| **dev split** | our held-out 10% of train; the real feedback loop |
| **stratified** | split preserving the distribution of relevant-docs-per-query |
| **ablation** | measuring one component's contribution by removing/changing only it |
| **ablation ladder** | the ordered chain of systems that answers BTC's research question |
| **run_id** | date + config hash; identical configs produce identical run_ids |
| **freeze** | lock config + seeds + revisions so the run reproduces exactly |
